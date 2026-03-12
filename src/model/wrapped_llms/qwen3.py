from collections import defaultdict
from typing import Optional, Union

import torch
from torch import nn
from transformers import Cache, Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from src.model.audio.audio_feature_extractor import AudioFeatureExtractor
from src.model.llm_audio_wrapper import LLMAudioWrapperForCausalLM, LLMAudioWrapperModel


class Qwen3AudioWrappedConfig(Qwen3Config):
    model_type = "qwen3_audio"

    def __init__(
        self,
        audio_encoder_configs=None,
        audio_adapter_configs=None,
        audio_fusion_config=None,
        audio_postprocessing_config=None,
        audio_sep_d_embed=1024,
        use_explicit_audio_tokens=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.audio_sep_d_embed = audio_sep_d_embed
        self.audio_encoder_configs = audio_encoder_configs
        self.audio_adapter_configs = audio_adapter_configs
        self.audio_fusion_config = audio_fusion_config
        self.audio_postprocessing_config = audio_postprocessing_config
        self.use_explicit_audio_tokens = use_explicit_audio_tokens


class Qwen3AudioWrappedModel(LLMAudioWrapperModel, Qwen3Model):
    config_class = Qwen3AudioWrappedConfig

    def __init__(self, config):
        super().__init__(config)


class Qwen3AudioWrappedForCausalLM(Qwen3ForCausalLM, LLMAudioWrapperForCausalLM):
    config_class = Qwen3AudioWrappedConfig

    def __init__(self, config):
        # we want to setup model ourselves
        # so we call super for the "next" object in MRO
        super(Qwen3ForCausalLM, self).__init__(config)

        self.model = Qwen3AudioWrappedModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_head(self):
        return self.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        inputs_system: Optional[torch.LongTensor] = None,
        inputs_start: Optional[torch.LongTensor] = None,
        inputs_main: Optional[torch.LongTensor] = None,
        outputs: Optional[torch.LongTensor] = None,
        audio: Optional[dict[torch.FloatTensor]] = None,
        audio_attention: Optional[dict[torch.LongTensor]] = None,
        audio_lengths: Optional[dict[torch.LongTensor]] = None,
        return_labels_and_loss_mask: bool = False,
        **kwargs
    ):
        if inputs_embeds is None:
            if input_ids is not None:
                inputs_embeds = self.get_model().embed_tokens(input_ids)
            else:
                (
                    inputs_embeds,
                    labels,
                    attention_mask,
                ) = self.prepare_multimodal_inputs(
                    inputs_system=inputs_system,
                    inputs_start=inputs_start,
                    inputs_main=inputs_main,
                    outputs=outputs,
                    audio=audio,
                    audio_attention=audio_attention,
                    audio_lengths=audio_lengths,
                )

        result = super().forward(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if not return_labels_and_loss_mask:
            return result

        loss_mask = labels != -100

        return result, labels, loss_mask

    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        inputs_system: Optional[torch.LongTensor] = None,
        inputs_start: Optional[torch.LongTensor] = None,
        inputs_main: Optional[torch.LongTensor] = None,
        outputs: Optional[torch.LongTensor] = None,
        audio: Optional[torch.FloatTensor] = None,
        audio_attention: Optional[torch.LongTensor] = None,
        audio_lengths: Optional[torch.LongTensor] = None,
        return_inputs_length: bool = False,
        **kwargs
    ):
        if inputs_start is not None:
            (
                inputs_embeds,
                labels,
                attention_mask,
            ) = self.prepare_multimodal_inputs(
                inputs_system=inputs_system,
                inputs_start=inputs_start,
                inputs_main=inputs_main,
                outputs=None,  # we do not want to share outputs
                audio=audio,
                audio_attention=audio_attention,
                audio_lengths=audio_lengths,
                padding_side="left",  # for generation only
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        result = super().generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )
        if return_inputs_length:
            return result, inputs_embeds.shape[1]

        return result


class Qwen3AudioWrappedFeatureExtractor(nn.Module):
    def __init__(self, model_config, checkpoint_name, tokenizer):
        super().__init__()
        self.model = Qwen3AudioWrappedForCausalLM.from_pretrained(checkpoint_name)
        self.feature_extractor = AudioFeatureExtractor(
            model_config.feature_extractor_configs
        )

        self.model.set_tokenizer_arguments(tokenizer)

        self.model.eval()
        self.freeze()
        self.device = "cpu"

        self.model_config = model_config

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def to(self, device):
        self.model.get_model().embed_tokens.to(device)
        self.model.get_audio_adapter().to(device)
        self.model.get_audio_encoder().to(device)
        self.model.get_audio_fusion().to(device)
        self.model.get_audio_postprocessing().to(device)
        if not self.model.get_config().use_explicit_audio_tokens:
            audio_start_emb = self.model.get_model().audio_start_emb.to(device)
            self.model.get_model().audio_start_emb = nn.Parameter(audio_start_emb)
            audio_end_emb = self.model.get_model().audio_end_emb.to(device)
            self.model.get_model().audio_end_emb = nn.Parameter(audio_end_emb)
        self.device = device

    def prepare_audio(self, audio_list):
        audio_lengths = defaultdict(list)
        audio = defaultdict(list)
        audio_attention = defaultdict(list)

        for elem in audio_list:
            features = self.feature_extractor(elem)
            for encoder_name in features["audio"].keys():
                audio[encoder_name].append(features["audio"][encoder_name])
                audio_lengths[encoder_name].append(
                    features["audio_lengths"][encoder_name]
                )
                audio_attention[encoder_name].append(
                    features["audio_attention"][encoder_name]
                )

        processed_audio = {}
        processed_attention = {}
        processed_lengths = {}

        for encoder_name in audio.keys():
            processed_lengths[encoder_name] = torch.tensor(
                audio_lengths[encoder_name]
            ).to(self.device)
            processed_audio[encoder_name] = torch.nn.utils.rnn.pad_sequence(
                audio[encoder_name], batch_first=True
            ).to(self.device)
            processed_attention[encoder_name] = torch.nn.utils.rnn.pad_sequence(
                audio_attention[encoder_name], batch_first=True
            ).to(self.device)

        return processed_audio, processed_attention, processed_lengths

    def get_text_embeds(self, input_ids):
        input_ids = [elem.to(self.device) for elem in input_ids]
        inputs_embeds = [
            self.model.get_model().embed_tokens(elem) for elem in input_ids
        ]
        return inputs_embeds

    def forward(self, inputs_system, inputs_start, inputs_main, audio):
        audio, audio_attention, audio_lengths = self.prepare_audio(audio)

        if inputs_system is not None:
            inputs_system = [elem.to(self.device) for elem in inputs_system]
        inputs_start = [elem.to(self.device) for elem in inputs_start]
        inputs_main = [elem.to(self.device) for elem in inputs_main]

        with torch.no_grad():
            (
                inputs_embeds,
                _,
                _,
            ) = self.model.prepare_multimodal_inputs(
                inputs_system=inputs_system,
                inputs_start=inputs_start,
                inputs_main=inputs_main,
                outputs=None,  # we do not want to share outputs
                audio=audio,
                audio_attention=audio_attention,
                audio_lengths=audio_lengths,
                return_batch=False,
            )
        return inputs_embeds
