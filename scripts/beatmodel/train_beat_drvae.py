#!/usr/bin/env python3
"""Discriminatively regularised beat VAE (DR-VAE) — the generative half of the beat model.

The Nature 2026 main text says only "a variational auto-encoder (VAE) with 512-dimension latent
space". Their released code (github.com/alexmschubert/ECG-SCD, 02_Morphing/s05_utils_vae.py) is
more specific -- the objective is

    loss = loss_rec + alpha * loss_kl + beta * dr_reg,
    dr_reg(x, x_hat) = (pred_fn(x) - pred_fn(x_hat)) ** 2

with `pred_fn` the frozen predictive model's probability for the outcome being morphed, alpha
annealed up to 0.01 and beta = 0.01, z_dim = 512. The discriminative term is what keeps the
autoencoder from smoothing away exactly the morphology the classifier reads -- without it the
decoder is free to reconstruct a plausible beat that has lost the ST-T detail.

Two deviations, both deliberate and reported:
  1. `dr_reg` here is a VECTOR discrepancy over [P(ACCO), P(LEFT), P(RCA), P(LCX)], not their
     single scalar. Their task is binary; ours is a 3-way territory softmax, and a scalar
     regulariser would preserve only the occlusion dimension and let territory be destroyed.
  2. The decoder is the convolutional BeatVAE from DeepECG-PNG rather than their 4x100 MLP.

The DR weight is auto-balanced on the first batch so the term starts at DR_TARGET of the
reconstruction loss, which makes the setting reproducible across unit conventions (they train on
raw ADC counts with a summed reconstruction loss; we train on millivolts with a mean one).
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
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/volume/DeepECG-PNG/scripts/explainability/beatmodel")
from beat_vae import BeatVAE, to_units  # noqa: E402

from fairseq_signals.utils import checkpoint_utils  # noqa: E402
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel  # noqa: E402

A = "/volume/DeepECG-SSL-finetune/data/acs"
BASE = "/volume/DeepECG-SSL-finetune/data/acs_final"
CACHE = f"{BASE}/beat_cache_nk"
CLF = f"{BASE}/checkpoints/beat_3head_nk/beat_classifier.pt"
OUT = f"{BASE}/checkpoints/beat_drvae"
CLASSES = ["LEFT", "RCA", "LCX"]
DEV = "cuda"

EPOCHS = int(os.environ.get("VAE_EPOCHS", 60))
BATCH = int(os.environ.get("VAE_BATCH", 128))
LR = float(os.environ.get("VAE_LR", 2e-4))
BETA_KL = float(os.environ.get("VAE_BETA_KL", 5e-4))     # on mean-KL; annealed in
DR_TARGET = float(os.environ.get("VAE_DR_TARGET", 0.2))  # DR term as a fraction of rec at start
WIDTH = int(os.environ.get("VAE_WIDTH", 96))
Z = 512
WARM = 0.1                                               # fraction of steps for the KL warm-up
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
    ck = torch.load(CLF, map_location="cpu")
    m.load_state_dict(ck["model_state_dict"])
    m.eval().to(DEV)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck


def probs(clf, x):
    """-> (B,4): [P(ACCO), P(LEFT), P(RCA), P(LCX)]. Differentiable w.r.t. x."""
    _, lc, lv = clf(x)
    return torch.cat([torch.sigmoid(lc), torch.softmax(lv, 1)], 1)


def main():
    Xtr = torch.from_numpy(np.load(f"{CACHE}/train_beats.npy")).float()
    Xva_np = np.load(f"{CACHE}/val_beats.npy")
    mva = pd.read_parquet(f"{CACHE}/val_beats.parquet")
    sel = np.random.default_rng(0).choice(len(Xva_np), min(6144, len(Xva_np)), replace=False)
    Xva = torch.from_numpy(Xva_np[sel]).float()
    yva_acco = mva.ACCO.to_numpy()[sel]
    yva_terr = mva.territory.to_numpy()[sel]
    mva_terr = mva.single_terr_acco.to_numpy()[sel].astype(bool)
    print(f"[data] train {tuple(Xtr.shape)}  val {tuple(Xva.shape)}", flush=True)

    clf, ck = load_classifier()
    print(f"[clf] {CLF}", flush=True)
    m = BeatVAE(z_dim=Z, width=WIDTH).to(DEV)
    print(f"[vae] z={Z} width={WIDTH} {sum(p.numel() for p in m.parameters())/1e6:.1f}M params",
          flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    steps = len(Xtr) // BATCH
    total = steps * EPOCHS
    Xva_d = Xva.to(DEV)

    beta_dr = None
    best, hist, gstep = float("inf"), [], 0
    for ep in range(EPOCHS):
        m.train()
        t0 = time.time()
        acc = {"rec": 0.0, "kl": 0.0, "dr": 0.0}
        perm = torch.randperm(len(Xtr))
        for s in range(steps):
            x = Xtr[perm[s * BATCH:(s + 1) * BATCH]].to(DEV, non_blocking=True)
            xr, mu, lv = m(x)
            rec = F.mse_loss(to_units(xr), to_units(x))
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean()
            with torch.no_grad():
                p_real = probs(clf, x)
            dr = F.mse_loss(probs(clf, xr), p_real)
            if beta_dr is None:                      # auto-balance once, then fix
                beta_dr = float(DR_TARGET * rec.item() / max(dr.item(), 1e-8))
                print(f"[dr] beta_dr = {beta_dr:.4g}  (rec {rec.item():.4f}, dr {dr.item():.5f})",
                      flush=True)
            a_kl = BETA_KL * min(1.0, gstep / max(1.0, WARM * total))
            loss = rec + a_kl * kl + beta_dr * dr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            gstep += 1
            acc["rec"] += rec.item()
            acc["kl"] += kl.item()
            acc["dr"] += dr.item()

        # ---- validation: reconstruction fidelity AND signal fidelity
        m.eval()
        with torch.no_grad():
            xr_all, pr_all, pv_all = [], [], []
            for i in range(0, len(Xva_d), 256):
                xb = Xva_d[i:i + 256]
                xrb, _, _ = m(xb)
                xr_all.append(xrb)
                pr_all.append(probs(clf, xb))
                pv_all.append(probs(clf, xrb))
            xr = torch.cat(xr_all)
            p_real = torch.cat(pr_all).cpu().numpy()
            p_rec = torch.cat(pv_all).cpu().numpy()
            vrec = F.mse_loss(to_units(xr), to_units(Xva_d)).item()
            a = xr.cpu().numpy().reshape(len(xr), -1)
            b = Xva.numpy().reshape(len(Xva), -1)
            r_beat = float(np.nanmean([np.corrcoef(a[i], b[i])[0, 1]
                                       for i in range(0, len(a), 4)]))
        corr_acco = float(np.corrcoef(p_real[:, 0], p_rec[:, 0])[0, 1])
        au_real = roc_auc_score(yva_acco, p_real[:, 0])
        au_rec = roc_auc_score(yva_acco, p_rec[:, 0])
        if mva_terr.sum() > 10:
            t_real = float(np.mean([roc_auc_score((yva_terr[mva_terr] == i).astype(int),
                                                  p_real[mva_terr][:, i + 1]) for i in range(3)]))
            t_rec = float(np.mean([roc_auc_score((yva_terr[mva_terr] == i).astype(int),
                                                p_rec[mva_terr][:, i + 1]) for i in range(3)]))
        else:
            t_real = t_rec = float("nan")
        hist.append({"epoch": ep, "rec": acc["rec"] / steps, "kl": acc["kl"] / steps,
                     "dr": acc["dr"] / steps, "val_rec": vrec, "val_corr_per_beat": r_beat,
                     "acco_prob_corr": corr_acco, "acco_auroc_real": au_real,
                     "acco_auroc_recon": au_rec, "terr_macro_real": t_real,
                     "terr_macro_recon": t_rec})
        print(f"ep{ep:03d} rec {acc['rec']/steps:.4f} kl {acc['kl']/steps:.1f} "
              f"dr {acc['dr']/steps:.5f} | val rec {vrec:.4f} corr {r_beat:.4f} | "
              f"P(ACCO) corr {corr_acco:.3f}  ACCO AUROC real {au_real:.3f} / recon {au_rec:.3f} | "
              f"terr macro real {t_real:.3f} / recon {t_rec:.3f} ({time.time()-t0:.0f}s)",
              flush=True)
        if vrec < best:
            best = vrec
            torch.save({"model": m.state_dict(), "z_dim": Z, "width": WIDTH, "epoch": ep,
                        "val_rec": vrec, "val_corr_per_beat": r_beat, "beta_kl": BETA_KL,
                        "beta_dr": beta_dr, "acco_prob_corr": corr_acco,
                        "acco_auroc_recon": au_rec, "terr_macro_recon": t_rec},
                       f"{OUT}/beat_drvae_best.pt")
        json.dump(hist, open(f"{OUT}/history.json", "w"), indent=1)
    print(f"[done] best val_rec {best:.4f} -> {OUT}/beat_drvae_best.pt", flush=True)


if __name__ == "__main__":
    main()
