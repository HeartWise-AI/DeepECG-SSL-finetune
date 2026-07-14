#!/usr/bin/env python3
"""DeepECG-ACS 3-head — PyTorch model definition + loader (for fine-tuning / refining).

Requires the fairseq-signals framework (DeepECG-SSL-finetune) and a WCRv2 backbone
checkpoint (any ecg_transformer_classifier checkpoint serves as the architecture donor;
its weights are overwritten by best_model.pt). See README for setup.
"""
import torch, torch.nn as nn
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel


class ThreeHead(nn.Module):
    """One shared WCRv2 backbone + 3 heads: ACS, ACCO gate, culprit vessel (4)."""
    def __init__(self, base, d=768):
        super().__init__()
        self.base = base                              # ecg_transformer_classifier (backbone)
        self.h_acs = nn.Linear(d, 1)
        self.h_acco = nn.Linear(d, 1)
        self.h_ves = nn.Linear(d, 4)

    def forward(self, source):                        # source: (B, 12, 2500) in mV
        res = ECGTransformerFinetuningModel.forward(self.base, source=source)
        x = res["x"]; pad = res["padding_mask"]
        x = self.base.final_dropout(x)
        if pad is not None and pad.any():
            x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        f = x.sum(1) / (x != 0).sum(1).clamp(min=1)   # mean-pool over time
        return self.h_acs(f), self.h_acco(f), self.h_ves(f)   # logits


def load_model(best_model_pt, wcrv2_backbone_ckpt, device="cuda"):
    """Rebuild the architecture from a WCRv2 backbone checkpoint, then load fine-tuned weights.
    wcrv2_backbone_ckpt: a WCRv2 ecg_transformer_classifier checkpoint (architecture donor).
    Returns (model, meta) where meta = {gate, vessels, scale}."""
    base = checkpoint_utils.load_model_and_task(wcrv2_backbone_ckpt)[0]
    base = base[0] if isinstance(base, list) else base
    base.num_updates = 10**9                          # ensure backbone is trainable
    model = ThreeHead(base)
    ck = torch.load(best_model_pt, map_location="cpu")
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    return model, {"gate": ck["gate"], "vessels": ck["vessels"], "scale": ck["scale"]}
