"""ECG transformer classifier that also consumes per-ECG auxiliary scalars
(age, sex, ...) alongside the waveform.

The waveform path is identical to ``ecg_transformer_classifier`` (WCR encoder ->
masked mean pooling). The auxiliary vector is passed through a small MLP and
concatenated with the pooled embedding before the final linear head. Missing
auxiliary values are encoded as NaN by the dataset; they are zero-filled and,
when ``aux_missing_indicator`` is set, a per-feature missingness flag is appended
so the model can learn to ignore absent demographics (about half of MHI ECGs
have no recorded age).

Manifest: add ``aux_path:/path/to/{split}_aux.npy`` (shape (N, aux_dim),
float32, NaN = missing) next to ``x_path`` / ``y_path``.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from fairseq_signals.models import register_model
from fairseq_signals.models.classification.ecg_transformer_classifier import (
    ECGTransformerClassificationConfig,
    ECGTransformerClassificationModel,
)
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel


@dataclass
class ECGTransformerAuxClassificationConfig(ECGTransformerClassificationConfig):
    aux_dim: int = field(
        default=2, metadata={"help": "number of auxiliary scalar inputs per ECG (e.g. age_z, sex)"}
    )
    aux_hidden_dim: int = field(
        default=64, metadata={"help": "hidden size of the auxiliary MLP (0 = concatenate raw aux)"}
    )
    aux_dropout: float = field(default=0.1, metadata={"help": "dropout inside the auxiliary MLP"})
    aux_missing_indicator: bool = field(
        default=True,
        metadata={"help": "append a 0/1 missingness flag per auxiliary feature (NaN in input)"},
    )


@register_model("ecg_transformer_aux_classifier", dataclass=ECGTransformerAuxClassificationConfig)
class ECGTransformerAuxClassificationModel(ECGTransformerClassificationModel):
    def __init__(self, cfg: ECGTransformerAuxClassificationConfig, encoder):
        super().__init__(cfg, encoder)
        self.aux_dim = cfg.aux_dim
        self.aux_missing_indicator = cfg.aux_missing_indicator
        aux_in = cfg.aux_dim * (2 if cfg.aux_missing_indicator else 1)

        if cfg.aux_hidden_dim and cfg.aux_hidden_dim > 0:
            self.aux_proj = nn.Sequential(
                nn.Linear(aux_in, cfg.aux_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.aux_dropout),
            )
            aux_out = cfg.aux_hidden_dim
        else:
            self.aux_proj = nn.Identity()
            aux_out = aux_in

        # replace the waveform-only head built by the parent class
        self.proj = nn.Linear(cfg.encoder_embed_dim + aux_out, cfg.num_labels)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 0.0)

    def _prepare_aux(self, aux, batch_size, device, dtype):
        if aux is None:
            aux = torch.full((batch_size, self.aux_dim), float("nan"), device=device, dtype=dtype)
        aux = aux.to(device=device, dtype=dtype)
        missing = torch.isnan(aux)
        aux = torch.nan_to_num(aux, nan=0.0)
        if self.aux_missing_indicator:
            aux = torch.cat([aux, missing.to(dtype)], dim=-1)
        return aux

    def forward(self, source, padding_mask=None, aux=None, **kwargs):
        res = ECGTransformerFinetuningModel.forward(self, source=source, padding_mask=padding_mask)

        x = res["x"]
        padding_mask = res["padding_mask"]

        x = self.final_dropout(x)
        if padding_mask is not None and padding_mask.any():
            x[padding_mask] = 0
        x = torch.div(x.sum(dim=1), (x != 0).sum(dim=1))

        aux = self._prepare_aux(aux, x.size(0), x.device, x.dtype)
        h = self.aux_proj(aux)
        out = self.proj(torch.cat([x, h], dim=-1))

        return {
            "encoder_out": res["x"].detach(),
            "padding_mask": padding_mask,
            "out": out,
        }
