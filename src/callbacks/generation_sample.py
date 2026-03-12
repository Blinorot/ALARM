import torch
from tqdm.auto import tqdm
from transformers.integrations import TensorBoardCallback
from transformers.utils import logging as hf_logging

from src.metrics import BLEU, METEOR, ROUGE, BERTScore

logger = hf_logging.get_logger("custom_callbacks")


METRICS = {
    "bertscore": None,
    "meteor": None,
    "bleu": None,
    "rouge": None,
}


def get_metric_funcs(n):
    if METRICS["bertscore"] is not None:
        return METRICS

    METRICS["bertscore"] = BERTScore(batch_size=n, device="cpu")
    METRICS["meteor"] = METEOR()
    METRICS["bleu"] = BLEU()
    METRICS["rouge"] = ROUGE()
    return METRICS


def gen_eval_samples_preds(
    model,
    dataset,
    collator,
    tokenizer,
    n=5,
    max_length=512,
    autocast_type=torch.bfloat16,
    batch_size=4,
):
    model.eval()
    examples = []

    dataset_with_context = dataset.with_context_dataset
    dataset_no_context = dataset.no_context_dataset

    for i in range(min(n, len(dataset_with_context))):
        examples.append(dataset_with_context[i])
    for i in range(min(n, len(dataset_no_context))):
        examples.append(dataset_no_context[i])

    device = model.device

    results = []
    generated_responses = []
    target_responses = []

    inputs_list = []

    for data_dict in examples:
        elem = dataset.extract_data_from_dict(data_dict, dataset.return_audio)
        elem = dataset.prepare_data(elem, dataset.return_audio)
        inputs_list.append(elem)

    n_batches = len(inputs_list) // batch_size
    if len(inputs_list) % batch_size != 0:
        n_batches += 1

    for index in range(n_batches):
        index_start = index * batch_size
        index_end = (index + 1) * batch_size
        inputs = inputs_list[index_start:index_end]
        inputs = collator(inputs)
        inputs.pop("audio_description")
        inputs.pop("context")
        inputs.pop("reference")
        inputs.pop("rejected_outputs")
        inputs_device = {}
        for k, v in inputs.items():
            if isinstance(v, list):
                inputs_device[k] = [v_elem.to(device) for v_elem in v]
            elif isinstance(v, dict):
                inputs_device[k] = {
                    k_elem: v_elem.to(device) for k_elem, v_elem in v.items()
                }
            elif v is not None:
                inputs_device[k] = v.to(device)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=autocast_type):
                output_ids = model.generate(
                    **inputs_device,
                    max_new_tokens=max_length,
                    # return_inputs_length=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

        for elem_index in range(len(inputs["inputs_start"])):
            generated_text = tokenizer.decode(
                output_ids[elem_index], skip_special_tokens=True
            )
            generated_responses.append(generated_text)

    for elem, generated_text in zip(inputs_list, generated_responses):
        context = elem["audio_description"] + "\n" + elem["context"]
        final_text = (
            f"Input:\n\n{context}\n\nGenerated Response:\n\n{generated_text}\n\n"
        )
        if "llm_answer" in elem.keys():
            target_text = elem["llm_answer"]
            final_text += f"Ground-Truth Responce:\n\n{target_text}"
            target_responses.append(target_text)
        else:
            target_responses.append("")
        results.append(final_text)

    metrics = get_metric_funcs(n)
    final_metrics = {}
    for metric_name, metric in metrics.items():
        metric_outputs = metric(generated_responses, target_responses)
        for k, v in metric_outputs.items():
            final_metrics[f"{metric_name}_{k}"] = v

    return results, final_metrics


class SampleGenerationCallback(TensorBoardCallback):
    """
    A callback that generates and logs sample predictions.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        collator,
        num_samples=5,
        max_length=512,
        dataset_tag="eval",
        autocast_type=torch.bfloat16,
    ):
        super().__init__()
        self.dataset = dataset  # will be passed later
        self.dataset_tag = dataset_tag
        self.collator = collator
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length
        self.autocast_type = autocast_type

    def on_save(self, args, state, control, **kwargs):
        """
        Called after each evaluation.
        We use 'on_save' because otherwise callback will be
        re-called for each dataset and we will re-run callbacks.

        save_strategy is always equal to eval strategy, so it is fine.
        """
        joint_text, final_metrics = self.get_text(**kwargs)

        # to avoid DDP deadlock, we do generate on all processes
        if not state.is_world_process_zero:
            return

        self.tb_writer.add_text(
            tag=f"{self.dataset_tag}sample_generation",
            text_string=joint_text,
            global_step=state.global_step,
        )

        for k, v in final_metrics.items():
            dataset_k = f"{self.dataset_tag}{k}"
            self.tb_writer.add_scalar(
                tag=dataset_k,
                scalar_value=v,
                global_step=state.global_step,
            )

        self.tb_writer.flush()

    def get_text(self, **kwargs):
        model = kwargs["model"]
        # TODO: fix mixed precision for Deepspeed

        logger.info("Sampling generation outputs...")

        model.get_config().use_cache = True
        results, final_metrics = gen_eval_samples_preds(
            model=model,
            dataset=self.dataset,
            tokenizer=self.tokenizer,
            collator=self.collator,
            n=self.num_samples,
            max_length=self.max_length,
            autocast_type=self.autocast_type,
        )
        model.get_config().use_cache = False
        joint_text = "\n\n=========\n\n".join(results)
        return joint_text, final_metrics
