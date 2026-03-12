import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers import logging as hf_logging

from src.callbacks import AggregatorCallback, SampleGenerationCallback
from src.dataset.collate_fn import DataCollator
from src.dataset.hf_dataset import HFDataset
from src.model.audio.audio_adapter import AudioAdapter
from src.model.audio.audio_encoder import AudioEncoder
from src.model.audio.audio_feature_extractor import AudioFeatureExtractor
from src.model.audio.audio_fusion import AudioFusion
from src.model.audio.audio_postprocessing import AudioPostprocessing
from src.model.wrapped_llms.qwen3 import (
    Qwen3AudioWrappedConfig,
    Qwen3AudioWrappedForCausalLM,
)
from src.utils import ROOT_PATH


def load_pretrained_adapters(model, save_dir, pretrained_adapters_names):
    save_dir = (ROOT_PATH / save_dir).resolve()
    adapters = model.get_audio_adapter().adapters
    for encoder_name in adapters.keys():
        if (
            pretrained_adapters_names is None
            or encoder_name in pretrained_adapters_names
        ):
            adapter_weights_path = save_dir / f"{encoder_name}.pth"
            weights = torch.load(adapter_weights_path)
            model.get_audio_adapter().adapters[encoder_name].load_state_dict(weights)
            print(f"Loaded weights for {encoder_name} adapter")


def init_model(model, config):
    llm_dict = AutoModelForCausalLM.from_pretrained(
        config.model.llm, torch_dtype="auto"
    ).state_dict()
    encoder_dict = AudioEncoder(config.model.config.audio_encoder_configs).state_dict()
    encoder_dict = {f"model.audio_encoder.{k}": v for k, v in encoder_dict.items()}
    # adapter_dict = AudioAdapter(config.model.config.audio_adapter_configs).state_dict()
    # fusion_dict = AudioFusion(config.model.config.audio_fusion_config).state_dict()
    # postprocessing_dict = AudioPostprocessing(
    #     config.model.config.audio_postprocessing_config
    # ).state_dict()
    adapter_dict = model.get_audio_adapter().state_dict()
    fusion_dict = model.get_audio_fusion().state_dict()
    postprocessing_dict = model.get_audio_postprocessing().state_dict()
    adapter_dict = {f"model.audio_adapter.{k}": v for k, v in adapter_dict.items()}
    fusion_dict = {f"model.audio_fusion.{k}": v for k, v in fusion_dict.items()}
    postprocessing_dict = {
        f"model.audio_postprocessing.{k}": v for k, v in postprocessing_dict.items()
    }

    combined_dict = {
        **llm_dict,
        **encoder_dict,
        **adapter_dict,
        **fusion_dict,
        **postprocessing_dict,
    }

    if not config.model.config.use_explicit_audio_tokens:
        audio_start_emb, audio_end_emb = model.get_audio_sep_embed()
        audio_sep_dict = {
            "model.audio_start_emb": audio_start_emb.detach(),
            "model.audio_end_emb": audio_end_emb.detach(),
        }
        combined_dict.update(**audio_sep_dict)

    model.load_state_dict(combined_dict, assign=True)


@hydra.main(
    version_base=None,
    config_path=str(ROOT_PATH / "src" / "configs"),
    config_name="main",
)
def train(config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model.llm)
    # hf_logging.set_verbosity_error()
    if config.model.from_pretrained is not None:
        model = Qwen3AudioWrappedForCausalLM.from_pretrained(
            config.model.from_pretrained
        )
    else:
        model_config = Qwen3AudioWrappedConfig.from_pretrained(config.model.llm)
        updated_model_config = OmegaConf.to_container(config.model.config, resolve=True)
        for k, v in updated_model_config.items():
            print(f"Resetting config kwarg: {k}...")
            setattr(model_config, k, v)
        # setattr(model_config, "attn_implementation", "flash_attention_2")
        model = Qwen3AudioWrappedForCausalLM(model_config)
        init_model(model, config)

    if config.pretrained_adapters_dir is not None:
        load_pretrained_adapters(
            model, config.pretrained_adapters_dir, config.pretrained_adapters_names
        )

    # turn of caching during training
    model.get_config().use_cache = False

    hf_logging.set_verbosity_info()

    # set max model len
    if config.model_max_length is not None:
        tokenizer.model_max_length = config.model_max_length

    model.set_tokenizer_arguments(tokenizer)
    model.freeze_llm()  # freeze all llm layers + head
    model.get_audio_encoder().freeze_encoder()  # freeze audio encoder

    if config.freeze_adapters:
        # used to train audio fusion only
        model.get_audio_adapter().freeze_adapter(config.frozen_adapters_names)

    if config.freeze_fusion:
        # used to train audio fusion only
        model.get_audio_fusion().freeze_fusion()

    # model = AutoModelForCausalLM.from_pretrained(config.model.llm)

    training_args = OmegaConf.to_container(config.training_args, resolve=True)
    training_args = TrainingArguments(**training_args)
    feature_extractor = AudioFeatureExtractor(config.model.feature_extractor_configs)

    if config.alternative_data_dir is not None:
        dataset_data_dir = config.alternative_data_dir
    else:
        dataset_data_dir = "data"

    train_dataset = instantiate(
        config.dataset.train,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        use_explicit_audio_tokens=config.model.config.use_explicit_audio_tokens,
        data_dir=dataset_data_dir,
    )

    collator = DataCollator(tokenizer=tokenizer)

    # aggregate eval metrics
    config_groups = config.dataset.groups
    if config_groups is None:
        groups = None
    else:
        groups = {}
        for k, v in config_groups.items():
            group_names = set(v)
            groups[k] = group_names

    print("GROUPS", groups)

    aggregator_callback = AggregatorCallback(groups)

    callbacks = [aggregator_callback]

    # monitor train generation quality
    if config.sample_generations:
        train_sample_generation_callback = SampleGenerationCallback(
            tokenizer, train_dataset, collator, dataset_tag="train/"
        )
        callbacks.append(train_sample_generation_callback)

    if training_args.eval_strategy != "no":
        eval_dataset_dict = {}
        for key in config.dataset.keys():
            if key == "train" or key == "groups":
                continue
            eval_dataset = instantiate(
                config.dataset[key],
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                use_explicit_audio_tokens=config.model.config.use_explicit_audio_tokens,
                data_dir=dataset_data_dir,
            )
            eval_dataset_dict[key] = eval_dataset
            # monitor eval generation quality
            if config.sample_generations:
                eval_sample_generation_callback = SampleGenerationCallback(
                    tokenizer, eval_dataset, collator, dataset_tag=f"eval/{key}_"
                )
                callbacks.append(eval_sample_generation_callback)
    else:
        eval_dataset = None

    print("Example from a dataset:")
    # https://github.com/huggingface/lerobot/issues/2488
    # for multiprocessing, we need to be careful with calling audio decoder
    # elem = train_dataset[0]
    elem = train_dataset.dataset.select_columns(
        ["context", "llm_answer", "audio_description"]
    )[0]
    elem = train_dataset.extract_data_from_dict(
        elem, return_audio=False  # to avoid calling audio decoder
    )
    elem = train_dataset.prepare_data(elem, return_audio=False)
    print(elem["inputs_start"] + elem["inputs_main"] + elem["outputs"])

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_dict,
        callbacks=callbacks,
    )

    if config.eval_mode:
        trainer.evaluate()
        return None

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    trainer.save_model(training_args.output_dir)
    trainer.evaluate()


if __name__ == "__main__":
    train()
