#!/usr/bin/env python3
"""Sample ACO-positive beats per territory from the conditional diffusion model, and audit them.

The point of a conditional generator is that an ACO-positive example is a SAMPLE from
P(beat | territory) rather than a normal beat hill-climbed until a classifier is satisfied. So
the audit asks three separate questions, and the third is the one the morph failed:

  1. Does the classifier read the samples as occlusions of the intended territory?
  2. Are they physiologically plausible (amplitude, per-lead dispersion) and not memorised
     copies of training beats?
  3. Does the sampled class contrast reproduce the REAL ischemic ST/T change for that territory?
     This is the same cosine used for the morph, so the two are directly comparable.

Sweeps classifier-free guidance, because guidance trades sample fidelity against diversity and
the right setting is not knowable in advance.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from beat_diffusion import N_CLASSES, BeatUNet, Schedule, ddim_sample  # noqa: E402

from fairseq_signals.utils import checkpoint_utils  # noqa: E402
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel  # noqa: E402

A = "/volume/DeepECG-SSL-finetune/data/acs"
BASE = "/volume/DeepECG-SSL-finetune/data/acs_final"
CACHE = f"{BASE}/beat_cache_nk"
CLF = f"{BASE}/checkpoints/beat_3head_nk/beat_classifier.pt"
DIFF = f"{BASE}/checkpoints/beat_diffusion/beat_diffusion.pt"
OUT = f"{BASE}/explainability"
CLASSES = ["normal", "LEFT", "RCA", "LCX"]
TERR = ["LEFT", "RCA", "LCX"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DEV = "cuda"
N_PER = int(os.environ.get("N_PER", 256))
GUIDANCES = [float(g) for g in os.environ.get("GUIDANCES", "0,1,2,3,5").split(",")]
DDIM_STEPS = int(os.environ.get("DDIM_STEPS", 100))
FS, PRE = 250, 70
ST0, ST1 = PRE + int(0.08 * FS), PRE + int(0.16 * FS)
T0, T1 = PRE + int(0.16 * FS), PRE + int(0.40 * FS)
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


def st_t(b):
    return b[..., ST0:ST1].mean(-1), b[..., T0:T1].max(-1)


def delta_vec(pos_mean, neg_mean):
    s1, t1 = st_t(pos_mean)
    s0, t0 = st_t(neg_mean)
    return np.concatenate([s1 - s0, t1 - t0])


@torch.no_grad()
def score(clf, x, bs=256):
    pa, pv = [], []
    for i in range(0, len(x), bs):
        xb = torch.as_tensor(x[i:i + bs], dtype=torch.float32, device=DEV)
        _, lc, lv = clf(xb)
        pa.append(torch.sigmoid(lc)[:, 0].cpu().numpy())
        pv.append(torch.softmax(lv, 1).cpu().numpy())
    return np.concatenate(pa), np.concatenate(pv)


def main():
    ck = torch.load(DIFF, map_location="cpu", weights_only=False)
    mean = torch.from_numpy(ck["mean"]).to(DEV)
    std = torch.from_numpy(ck["std"]).to(DEV)
    model = BeatUNet(base=ck["base_width"]).to(DEV)
    model.load_state_dict({k: v for k, v in ck["ema"].items()})
    model.eval()
    print(f"[diffusion] epoch {ck['epoch']}, EMA weights", flush=True)
    clf = load_classifier()
    sched = Schedule(device=DEV)

    # ---- real reference, held-out test beats
    Xte = np.load(f"{CACHE}/test_beats.npy").astype(np.float32)
    mte = pd.read_parquet(f"{CACHE}/test_beats.parquet")
    real_norm = Xte[(mte.ACS == 0).to_numpy()]
    real_norm = real_norm[np.random.default_rng(0).choice(len(real_norm), 2000, replace=False)]
    real_terr = {c: Xte[((mte.single_terr_acco == 1) & (mte.territory == i)).to_numpy()]
                 for i, c in enumerate(TERR)}
    real_delta = {c: delta_vec(real_terr[c].mean(0), real_norm.mean(0)) for c in TERR}
    p_acco_real, pv_real = score(clf, np.concatenate([real_norm[:512]] +
                                                     [real_terr[c] for c in TERR]))
    n_split = [512] + [len(real_terr[c]) for c in TERR]
    off = np.cumsum([0] + n_split)
    print(f"[real] P(ACCO) normal {p_acco_real[:512].mean():.3f}; " +
          ", ".join(f"{c} {p_acco_real[off[i+1]:off[i+2]].mean():.3f}"
                    for i, c in enumerate(TERR)), flush=True)

    # ---- training beats for the memorisation check
    Xtr = np.load(f"{CACHE}/train_beats.npy").astype(np.float32)
    mtr = pd.read_parquet(f"{CACHE}/train_beats.parquet")
    tr_terr = {c: Xtr[((mtr.single_terr_acco == 1) & (mtr.territory == i)).to_numpy()]
               for i, c in enumerate(TERR)}

    results, best = {}, None
    for w in GUIDANCES:
        y = torch.arange(N_CLASSES, device=DEV).repeat_interleave(N_PER)
        xs = []
        for i in range(0, len(y), 256):
            xs.append(ddim_sample(model, sched, len(y[i:i + 256]), y[i:i + 256], device=DEV,
                                  steps=DDIM_STEPS, guidance=w))
        x = torch.cat(xs) * std + mean
        xn = x.cpu().numpy()
        yv = y.cpu().numpy()
        p_acco, pv = score(clf, xn)

        r = {"guidance": w, "n_per_class": N_PER,
             "acco_normal": float(p_acco[yv == 0].mean()),
             "amp_p99": float(np.percentile(np.abs(xn), 99)),
             "amp_max": float(np.abs(xn).max())}
        samp_norm = xn[yv == 0]
        cos, hits = {}, []
        for i, c in enumerate(TERR):
            s = yv == i + 1
            r[f"acco_{c}"] = float(p_acco[s].mean())
            r[f"p_{c}"] = float(pv[s][:, i].mean())
            hits.append(float((pv[s].argmax(1) == i).mean()))
            dm = delta_vec(xn[s].mean(0), samp_norm.mean(0))
            cos[c] = float(dm @ real_delta[c] /
                           (np.linalg.norm(dm) * np.linalg.norm(real_delta[c])))
            # memorisation: distance to the nearest real training beat of the same class
            a = torch.from_numpy(xn[s][:64].reshape(64, -1)).to(DEV)
            b = torch.from_numpy(tr_terr[c].reshape(len(tr_terr[c]), -1)).to(DEV)
            d_sample = torch.cdist(a, b).min(1).values.mean().item()
            bb = b[:512]
            d_real = torch.cdist(bb, b).kthvalue(2, dim=1).values.mean().item()
            r[f"nn_dist_{c}"] = d_sample
            r[f"nn_dist_real_{c}"] = d_real
        r["territory_top1"] = float(np.mean(hits))
        r["cosine_vs_real"] = cos
        results[str(w)] = r
        print(f"w={w:<4} P(ACCO) normal {r['acco_normal']:.3f} | "
              + " ".join(f"{c} {r['acco_'+c]:.3f}" for c in TERR)
              + f" | top-1 {r['territory_top1']:.3f} | cosine "
              + " ".join(f"{c} {cos[c]:+.2f}" for c in TERR)
              + f" | amp p99 {r['amp_p99']:.2f} mV", flush=True)
        # Save every guidance setting; the reference for figures is chosen below on
        # DISTRIBUTIONAL FAITHFULNESS -- the sampled P(occlusion) should match real occluded
        # beats, not exceed them. Guidance sharpens toward the class mode and inflates apparent
        # severity, which is the same failure the gradient morph has.
        real_acco = {c: float(p_acco_real[off[i + 1]:off[i + 2]].mean())
                     for i, c in enumerate(TERR)}
        r["real_acco"] = real_acco
        r["acco_gap"] = float(np.mean([abs(r[f"acco_{c}"] - real_acco[c]) for c in TERR]))
        if True:
            np.savez_compressed(f"{OUT}/conditional_samples_w{w:g}.npz", x=xn, y=yv, guidance=w,
                                p_acco=p_acco, p_terr=pv,
                                real_norm_mean=real_norm.mean(0),
                                **{f"real_{c}_mean": real_terr[c].mean(0) for c in TERR},
                                classes=np.array(CLASSES))
    ok = [k for k in results if results[k]["territory_top1"] >= 0.80]
    best = min(ok, key=lambda k: results[k]["acco_gap"])
    import shutil
    shutil.copy(f"{OUT}/conditional_samples_w{float(best):g}.npz",
                f"{OUT}/conditional_samples.npz")
    results["best_guidance"] = best
    results["selection_rule"] = ("smallest mean |sampled P(occlusion) - real P(occlusion)| "
                                 "among settings with territory top-1 >= 0.80")
    json.dump(results, open(f"{OUT}/conditional_samples.json", "w"), indent=1)
    print(f"\nbest guidance {best} -> {OUT}/conditional_samples.npz", flush=True)


if __name__ == "__main__":
    main()
