import re
from collections import defaultdict

import torch
from transformers.integrations import TensorBoardCallback
from transformers.utils import logging as hf_logging

logger = hf_logging.get_logger("custom_callbacks")


class AggregatorCallback(TensorBoardCallback):
    """
    A callback that aggregates metrics across the eval datasets.
    """

    def __init__(
        self,
        groups=None,
    ):
        """
        Args:
        groups (dict[str, set[str]]): dict with dataset names
            for group-wise aggregation
        """
        super().__init__()
        self.groups = groups
        self.metrics = {}

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        Called after each evaluation. This call is run for each dataset
        separately. Here, we gather metrics across all datasets.
        """
        if not state.is_world_process_zero:
            return
        self.metrics.update(**metrics)

    def on_save(self, args, state, control, **kwargs):
        """
        Called after each evaluation.
        We use 'on_save' because otherwise callback will be
        re-called for each dataset and we will re-run callbacks.

        save_strategy is always equal to eval strategy, so it is fine.
        """
        if not state.is_world_process_zero:
            return

        loss_metrics = defaultdict(list)

        for k, v in self.metrics.items():
            dataset_name = self.extract_dataset_name(k)
            if dataset_name is None:
                continue

            loss_metrics["eval/all_loss"].append(v)

            if self.groups is None:
                continue

            for group_name, group_elems in self.groups.items():
                if dataset_name in group_elems:
                    loss_metrics[f"eval/{group_name}_loss"].append(v)

        for k, v in loss_metrics.items():
            v = torch.tensor(v).mean()
            self.tb_writer.add_scalar(
                tag=k,
                scalar_value=v,
                global_step=state.global_step,
            )

        self.tb_writer.flush()
        self.metrics = {}  # reset

    def extract_dataset_name(self, metric_name):
        m = re.fullmatch(r"eval_(.+)_loss", metric_name)
        return m.group(1) if m else None
