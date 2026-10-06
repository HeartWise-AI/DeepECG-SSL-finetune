"""Masked multi-label BCE for multi-source, multi-task fine-tuning.

Targets are float tensors where ``NaN`` means "label not available for this
sample" (e.g. incident-AF labels on EchoNext rows, or echo labels on an MHI ECG
that has no echo within the linkage window). Missing entries contribute neither
to the loss nor to the metrics. Everything else (sample_size convention, fp16,
output stores for ``fairseq-hydra-inference``) matches
``binary_cross_entropy_with_logits`` so existing configs only need
``criterion._name=masked_bce``.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from fairseq_signals import metrics
from fairseq_signals.criterions import BaseCriterion, register_criterion
from fairseq_signals.dataclass import ChoiceEnum, Dataclass
from fairseq_signals.logging.meters import Meter, safe_round
from fairseq_signals.tasks import Task
from fairseq_signals.utils import utils

SAMPLE_SIZE_CHOICES = ChoiceEnum(["positives", "valid", "signals"])


class MaskedAUCMeter(Meter):
    """Accumulates (y_true, y_score) over every validation batch and reports the AUROC or AUPRC of
    the whole set. NaN targets are ignored. With 2-D inputs the value is the macro average over the
    columns that contain both classes; with 1-D inputs it is the metric of that single column.

    fairseq-signals reduces metrics once per validation batch, so a scalar logged per batch would be
    averaged across batches; accumulating here makes ``auroc`` (and ``best_checkpoint_metric``)
    the full-set value.
    """

    def __init__(self, metric: str = "auroc", round: Optional[int] = 4):
        assert metric in ("auroc", "auprc")
        self.metric = metric
        self.round = round
        self.reset()

    def reset(self):
        self.targets = []
        self.scores = []

    def update(self, y_true, y_score):
        self.targets.append(np.asarray(y_true, dtype=np.float32))
        self.scores.append(np.asarray(y_score, dtype=np.float32))

    def state_dict(self):
        return {"metric": self.metric, "round": self.round, "targets": self.targets, "scores": self.scores}

    def load_state_dict(self, state_dict):
        self.metric = state_dict["metric"]
        self.round = state_dict.get("round", None)
        self.targets = state_dict["targets"]
        self.scores = state_dict["scores"]

    @staticmethod
    def _score(metric, yt, ys):
        from sklearn.metrics import average_precision_score, roc_auc_score

        m = ~np.isnan(yt)
        yt, ys = yt[m], ys[m]
        if len(yt) == 0 or yt.min() == yt.max():
            return None
        return float(roc_auc_score(yt, ys) if metric == "auroc" else average_precision_score(yt, ys))

    @property
    def value(self):
        if not self.targets:
            return float("nan")
        y_true = np.concatenate(self.targets)
        y_score = np.concatenate(self.scores)
        if y_true.ndim == 1:
            v = self._score(self.metric, y_true, y_score)
            return float("nan") if v is None else v
        vals = [self._score(self.metric, y_true[:, j], y_score[:, j]) for j in range(y_true.shape[1])]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def smoothed_value(self) -> float:
        val = self.value
        if self.round is not None:
            val = safe_round(val, self.round)
        return val


@dataclass
class MaskedBinaryCrossEntropyWithLogitsCriterionConfig(Dataclass):
    threshold: float = field(
        default=0.5, metadata={"help": "probability threshold for accuracy / precision / recall"}
    )
    report_auc: bool = field(
        default=True, metadata={"help": "report macro AUROC / AUPRC over valid labels at validation"}
    )
    pos_weight: Optional[List[float]] = field(
        default=None,
        metadata={"help": "per-label positive-class weight (length = num_labels)"},
    )
    label_weights: Optional[List[float]] = field(
        default=None,
        metadata={"help": "per-label multiplicative loss weight (length = num_labels)"},
    )
    label_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "optional label names, used for per-label metric keys"},
    )
    log_per_label: bool = field(
        default=True, metadata={"help": "kept for config compatibility; per-label AUROC is always logged"}
    )
    sample_size_mode: SAMPLE_SIZE_CHOICES = field(
        default="positives",
        metadata={
            "help": "denominator used by the trainer to normalise gradients: "
            "'positives' = number of valid positive labels (same convention as "
            "binary_cross_entropy_with_logits, keeps v6 learning-rate semantics), "
            "'valid' = number of valid label entries, 'signals' = batch size"
        },
    )


@register_criterion(
    "masked_bce", dataclass=MaskedBinaryCrossEntropyWithLogitsCriterionConfig
)
class MaskedBinaryCrossEntropyWithLogitsCriterion(BaseCriterion):
    def __init__(self, cfg: MaskedBinaryCrossEntropyWithLogitsCriterionConfig, task: Task):
        super().__init__(task)
        self.threshold = cfg.threshold
        self.report_auc = cfg.report_auc
        self.pos_weight = None if cfg.pos_weight is None else torch.tensor(list(cfg.pos_weight), dtype=torch.float32)
        self.label_weights = None if cfg.label_weights is None else torch.tensor(list(cfg.label_weights), dtype=torch.float32)
        self.label_names = None if cfg.label_names is None else list(cfg.label_names)
        self.log_per_label = cfg.log_per_label
        self.sample_size_mode = str(cfg.sample_size_mode)

    def forward(self, model, sample, reduce=True, save_outputs=False):
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output).float()

        if save_outputs:
            # keep NaN in the stored targets so downstream evaluation can re-mask
            self.store(logits, target)

        valid = ~torch.isnan(target)
        tgt = torch.nan_to_num(target, nan=0.0)

        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)

        loss_el = F.binary_cross_entropy_with_logits(
            input=logits, target=tgt, pos_weight=pos_weight, reduction="none"
        )
        loss_el = loss_el * valid.float()
        if self.label_weights is not None:
            loss_el = loss_el * self.label_weights.to(logits.device)

        loss = loss_el.sum() if reduce else loss_el

        n_valid = int(valid.sum().item())
        n_pos = int((tgt * valid).sum().item())
        if self.sample_size_mode == "signals":
            sample_size = sample["id"].numel()
        elif self.sample_size_mode == "valid":
            sample_size = max(n_valid, 1)
        else:
            sample_size = n_pos if n_pos > 0 else max(n_valid, 1)

        logging_output = {
            "loss": loss.item() if reduce else loss.detach(),
            "nsignals": sample["id"].numel(),
            "sample_size": sample_size,
            "n_valid": n_valid,
        }

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            pred = probs > self.threshold
            correct = ((pred == tgt.bool()) & valid).sum().item()
            tp = (pred & tgt.bool() & valid).sum().item()
            fp = (pred & ~tgt.bool() & valid).sum().item()
            fn = (~pred & tgt.bool() & valid).sum().item()
            logging_output.update(
                {"correct": correct, "count": n_valid, "tp": tp, "fp": fp, "fn": fn}
            )
            if not self.training and self.report_auc:
                logging_output["_y_true"] = target.cpu().numpy()  # NaN = missing
                logging_output["_y_score"] = probs.cpu().numpy()
                if self.label_names is not None:
                    logging_output["_label_names"] = self.label_names

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nsignals = utils.item(sum(log.get("nsignals", 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))

        metrics.log_scalar("loss", loss_sum / (sample_size or 1) / math.log(2), sample_size, round=3)
        metrics.log_scalar("nsignals", nsignals)

        correct = sum(log.get("correct", 0) for log in logging_outputs)
        total = sum(log.get("count", 0) for log in logging_outputs)
        tp = sum(log.get("tp", 0) for log in logging_outputs)
        fp = sum(log.get("fp", 0) for log in logging_outputs)
        fn = sum(log.get("fn", 0) for log in logging_outputs)
        metrics.log_scalar("_correct", correct)
        metrics.log_scalar("_total", total)
        metrics.log_scalar("_tp", tp)
        metrics.log_scalar("_fp", fp)
        metrics.log_scalar("_fn", fn)
        if total > 0:
            metrics.log_derived(
                "accuracy",
                lambda m: safe_round(m["_correct"].sum / m["_total"].sum, 5) if m["_total"].sum > 0 else float("nan"),
            )
            metrics.log_derived(
                "precision",
                lambda m: safe_round(m["_tp"].sum / (m["_tp"].sum + m["_fp"].sum), 5)
                if (m["_tp"].sum + m["_fp"].sum) > 0 else float("nan"),
            )
            metrics.log_derived(
                "recall",
                lambda m: safe_round(m["_tp"].sum / (m["_tp"].sum + m["_fn"].sum), 5)
                if (m["_tp"].sum + m["_fn"].sum) > 0 else float("nan"),
            )

        if any("_y_true" in log for log in logging_outputs):
            y_true = np.concatenate([log["_y_true"] for log in logging_outputs if "_y_true" in log])
            y_score = np.concatenate([log["_y_score"] for log in logging_outputs if "_y_score" in log])
            # accumulated over the whole validation set (see MaskedAUCMeter); 'auroc' is the recommended
            # checkpoint.best_checkpoint_metric (maximize)
            metrics.log_custom(lambda: MaskedAUCMeter("auroc"), "auroc", y_true, y_score)
            metrics.log_custom(lambda: MaskedAUCMeter("auprc"), "auprc", y_true, y_score)
            names = None
            for log in logging_outputs:
                if "_label_names" in log:
                    names = log["_label_names"]
                    break
            for j in range(y_true.shape[1]):
                key = names[j] if names is not None and j < len(names) else f"label{j}"
                key = key.replace(" ", "_").replace("/", "_")
                metrics.log_custom(lambda: MaskedAUCMeter("auroc"), f"auroc_{key}", y_true[:, j], y_score[:, j])

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return False
