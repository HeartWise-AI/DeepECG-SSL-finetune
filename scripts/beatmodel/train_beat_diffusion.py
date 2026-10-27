#!/usr/bin/env python3
"""Train the class-conditional beat diffusion model: P(beat | territory).

Classes are drawn UNIFORMLY (25% each) rather than by natural prevalence, because occluded
beats are ~3% of the cohort and a prevalence-weighted sampler would give the three occlusion
classes almost no gradient. Class-conditional generation is the goal, not density estimation of
the cohort, so balanced exposure is the right choice; the cost is that the model's implicit
class prior is uniform and must not be read as a prevalence.

Per-lead standardisation statistics are computed on the training beats and stored in the
checkpoint, so sampling can invert them exactly.

Validation each epoch samples a small batch per class and scores it with the frozen beat
classifier -- the generator is useful only if its ACO-positive samples actually read as
occlusions of the intended territory.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beat_diffusion import (BEAT_LEN, N_CLASSES, NULL_CLASS, PAD_LEN, BeatUNet,  # noqa: E402
                            Schedule, ddim_sample)

from fairseq_signals.utils import checkpoint_utils  # noqa: E402
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel  # noqa: E402

A = "/volume/DeepECG-SSL-finetune/data/acs"
BASE = "/volume/DeepECG-SSL-finetune/data/acs_final"
CACHE = f"{BASE}/beat_cache_nk"
CLF = f"{BASE}/checkpoints/beat_3head_nk/beat_classifier.pt"
OUT = f"{BASE}/checkpoints/beat_diffusion"
CLASSES = ["normal", "LEFT", "RCA", "LCX"]
DEV = "cuda"
EPOCHS = int(os.environ.get("DIFF_EPOCHS", 120))
BATCH = int(os.environ.get("DIFF_BATCH", 256))
LR = float(os.environ.get("DIFF_LR", 2e-4))
BASE_W = int(os.environ.get("DIFF_WIDTH", 96))
P_DROP = 0.1                       # class dropout for classifier-free guidance
STEPS_PER_EPOCH = int(os.environ.get("DIFF_SPE", 400))
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)
np.random.seed(0)


class ThreeHeadSoftmax(nn.Module):
    def __init__(self, base, d=768, n_cls=3):
        super().__init__()
        self.base = base
        self.h_acs = nn.Linear(d, 1)
        self.h_acco = nn.Linear(d, 1)
        self.h_ves = nn.Linear(d, n_cls)

    def forward(self, source):
        res = ECGTransformerFinetuningModel.forward(self.base, source=source)
        x = res["x"]
        pad = res["padding_mask"]
        x = self.base.final_dropout(x)
        if pad is not None and pad.any():
            x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        f = x.sum(1) / (x != 0).sum(1).clamp(min=1)
        return self.h_acs(f), self.h_acco(f), self.h_ves(f)


def load_classifier():
    base = checkpoint_utils.load_model_and_task(f"{A}/checkpoints/acs/checkpoint_best.pt")[0]
    base = base[0] if isinstance(base, list) else base
    base.num_updates = 10 ** 9
    m = ThreeHeadSoftmax(base)
    m.load_state_dict(torch.load(CLF, map_location="cpu")["model_state_dict"])
    m.eval().to(DEV)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def class_index(meta):
    """0 = normal (non-ACS), 1..3 = single-territory complete occlusion, -1 = unused."""
    y = np.full(len(meta), -1, np.int64)
    y[(meta.ACS == 0).to_numpy()] = 0
    sel = (meta.single_terr_acco == 1).to_numpy()
    y[sel] = meta.territory.to_numpy()[sel] + 1
    return y


def main():
    Xtr = np.load(f"{CACHE}/train_beats.npy").astype(np.float32)
    mtr = pd.read_parquet(f"{CACHE}/train_beats.parquet")
    ytr = class_index(mtr)
    keep = ytr >= 0
    Xtr, ytr = Xtr[keep], ytr[keep]
    idx_by_class = [np.where(ytr == c)[0] for c in range(N_CLASSES)]
    print("[data] " + ", ".join(f"{CLASSES[c]} {len(idx_by_class[c])}" for c in range(N_CLASSES)),
          flush=True)

    mean = Xtr.mean((0, 2), keepdims=True)
    std = Xtr.std((0, 2), keepdims=True) + 1e-6
    print(f"[norm] per-lead std range {std.min():.3f}-{std.max():.3f} mV", flush=True)
    Xn = torch.from_numpy((Xtr - mean) / std)
    Xn = F.pad(Xn, (0, PAD_LEN - BEAT_LEN))
    mean_t = torch.from_numpy(mean).to(DEV)
    std_t = torch.from_numpy(std).to(DEV)

    clf = load_classifier()
    sched = Schedule(device=DEV)
    print(f"[sched] t_max {sched.t_max}, alpha_bar range "
          f"{sched.ab.min():.2e}-{sched.ab.max():.4f}", flush=True)
    model = BeatUNet(base=BASE_W).to(DEV)
    print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
    rng = np.random.default_rng(0)

    def sample_batch(n):
        per = n // N_CLASSES
        ids = np.concatenate([rng.choice(idx_by_class[c], per, replace=True)
                              for c in range(N_CLASSES)])
        return Xn[ids].to(DEV, non_blocking=True), torch.from_numpy(ytr[ids]).to(DEV)

    @torch.no_grad()
    def eval_samples(n_per=32, guidance=3.0, steps=60):
        bak = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: v for k, v in ema.items()})
        model.eval()
        y = torch.arange(N_CLASSES, device=DEV).repeat_interleave(n_per)
        x = ddim_sample(model, sched, len(y), y, device=DEV, steps=steps, guidance=guidance)
        xb = x * std_t + mean_t
        _, lc, lv = clf(xb)
        p_acco = torch.sigmoid(lc)[:, 0].cpu().numpy()
        pv = torch.softmax(lv, 1).cpu().numpy()
        model.load_state_dict(bak)
        model.train()
        y = y.cpu().numpy()
        out = {"acco_normal": float(p_acco[y == 0].mean()),
               "amp_p99": float(xb.abs().flatten().kthvalue(int(0.99 * xb.numel()))[0])}
        hit = []
        for c in range(1, N_CLASSES):
            s = y == c
            out[f"acco_{CLASSES[c]}"] = float(p_acco[s].mean())
            out[f"p_{CLASSES[c]}"] = float(pv[s][:, c - 1].mean())
            hit.append(float((pv[s].argmax(1) == (c - 1)).mean()))
        out["territory_top1"] = float(np.mean(hit))
        return out

    steps = STEPS_PER_EPOCH
    hist = []
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        tot = 0.0
        for _ in range(steps):
            x0, y = sample_batch(BATCH)
            y = torch.where(torch.rand(len(y), device=DEV) < P_DROP,
                            torch.full_like(y, NULL_CLASS), y)
            t = torch.randint(0, sched.t_max + 1, (len(x0),), device=DEV)
            noise = torch.randn_like(x0)
            xt = sched.q_sample(x0, t, noise)
            loss = F.mse_loss(model(xt, t, y), noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema[k].mul_(0.999).add_(v.float(), alpha=0.001)
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / steps, "sec": time.time() - t0}
        if ep % 5 == 4 or ep == EPOCHS - 1:
            rec.update(eval_samples())
            print(f"ep{ep:03d} loss {rec['loss']:.4f} | samples: P(ACCO) normal "
                  f"{rec['acco_normal']:.3f} LEFT {rec['acco_LEFT']:.3f} "
                  f"RCA {rec['acco_RCA']:.3f} LCX {rec['acco_LCX']:.3f} | "
                  f"territory top-1 {rec['territory_top1']:.3f} | "
                  f"amp p99 {rec['amp_p99']:.2f} mV ({rec['sec']:.0f}s)", flush=True)
            torch.save({"ema": ema, "model": model.state_dict(), "mean": mean, "std": std,
                        "epoch": ep, "base_width": BASE_W, "classes": CLASSES, "metrics": rec},
                       f"{OUT}/beat_diffusion.pt")
        else:
            print(f"ep{ep:03d} loss {rec['loss']:.4f} ({rec['sec']:.0f}s)", flush=True)
        hist.append(rec)
        json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    print(f"[done] -> {OUT}/beat_diffusion.pt", flush=True)


if __name__ == "__main__":
    main()
