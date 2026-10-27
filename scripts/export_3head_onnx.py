#!/usr/bin/env python3
"""Re-export the trained 3-head model to ONNX using the legacy (dynamo=False) exporter
(avoids the onnxscript dependency). Outputs: acs_probability, acco_probability, vessel_probability."""
import torch, torch.nn as nn
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel

A = "/volume/DeepECG-SSL-finetune/data/acs"
OUT = f"{A}/checkpoints/acs_3head"

class ThreeHead(nn.Module):
    def __init__(self, base, d=768):
        super().__init__()
        self.base = base
        self.h_acs = nn.Linear(d, 1); self.h_acco = nn.Linear(d, 1); self.h_ves = nn.Linear(d, 4)
    def pooled(self, source):
        res = ECGTransformerFinetuningModel.forward(self.base, source=source)
        x = res["x"]; pad = res["padding_mask"]
        x = self.base.final_dropout(x)
        if pad is not None and pad.any(): x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        return x.sum(1) / (x != 0).sum(1).clamp(min=1)
    def forward(self, source):
        f = self.pooled(source)
        return self.h_acs(f), self.h_acco(f), self.h_ves(f)

class Export(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, ecg_12lead):
        la, lc, lv = self.m(ecg_12lead)
        return torch.sigmoid(la), torch.sigmoid(lc), torch.sigmoid(lv)

base = checkpoint_utils.load_model_and_task(f"{A}/checkpoints/acs/checkpoint_best.pt")[0]
base = base[0] if isinstance(base, list) else base
base.num_updates = 10**9
model = ThreeHead(base)
ckpt = torch.load(f"{OUT}/best_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"]); model.eval().cuda()
print("gate:", ckpt["gate"], "vessels:", ckpt["vessels"], "scale:", ckpt["scale"])

dummy = torch.randn(1, 12, 2500, device="cuda")
torch.onnx.export(Export(model).eval(), dummy, f"{OUT}/acs_acco_vessel_3head.onnx",
                  input_names=["ecg_12lead"],
                  output_names=["acs_probability", "acco_probability", "vessel_probability"],
                  dynamic_axes={"ecg_12lead": {0: "B"}, "acs_probability": {0: "B"},
                                "acco_probability": {0: "B"}, "vessel_probability": {0: "B"}},
                  opset_version=14, dynamo=False)
print("saved", f"{OUT}/acs_acco_vessel_3head.onnx")
