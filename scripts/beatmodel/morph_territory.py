#!/usr/bin/env python3
"""Counterfactual beat morphing toward acute occlusion in a named territory.

Nature 2026 protocol (s41586-026-10674-6): "we identify the gradient of predicted risk around
each beat, and perturb its latent vector to follow the gradient. This produces a higher-risk
vector, which we then pass through the decoder to reconstruct, resulting in a counterfactual,
higher-risk ECG waveform. The new vector is the starting point for another round of perturbation
and reconstruction. We repeat 2,000 times, or until the risk of the generated synthetic beat
reaches the 90th risk percentile."

Their released loop (02_Morphing/s10_morph_ecgs.py) ascends the raw pre-sigmoid logit by plain
fixed-step gradient ascent with no manifold constraint, and gates beats before morphing on
reconstruction error. The 90th-percentile stop is in the paper text but not in the released code
(MORPH_YHAT_UPPER_BOUND = 1.0 there, so every beat runs the full 2,000 steps and saturates).
It is implemented here, because saturation is exactly what made the previous attempt on this
project uninterpretable.

Multiclass adaptation: their outcome is one scalar risk. Ours is "complete occlusion, in
territory k", so the ascended score is

    logit_ACCO  +  [ logit_k - logsumexp_{j != k} logit_j ]

i.e. occlusion plus the one-vs-rest territory margin. Ascending a raw softmax logit alone would
let all three territory logits inflate together.

-> data/acs_final/explainability/morphs.npz   (per territory: host, t0, tfinal, trajectories)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/volume/DeepECG-PNG/scripts/explainability/beatmodel")
from beat_vae import BeatVAE, to_units  # noqa: E402

from fairseq_signals.utils import checkpoint_utils  # noqa: E402
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel  # noqa: E402

A = "/volume/DeepECG-SSL-finetune/data/acs"
BASE = "/volume/DeepECG-SSL-finetune/data/acs_final"
CACHE = f"{BASE}/beat_cache_nk"
CLF = f"{BASE}/checkpoints/beat_3head_nk/beat_classifier.pt"
VAE = f"{BASE}/checkpoints/beat_drvae/beat_drvae_best.pt"
OUT = f"{BASE}/explainability"
CLASSES = ["LEFT", "RCA", "LCX"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DEV = "cuda"

N_HOSTS = int(os.environ.get("N_HOSTS", 64))      # paper: 56
STEPS = int(os.environ.get("STEPS", 2000))
LR = float(os.environ.get("MORPH_LR", 1e-2))      # their fixed step
MIN_RECON_CORR = 0.90                             # pre-morph gate (their recon-error gate)
FS, PRE = 250, 70
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


def load_models():
    base = checkpoint_utils.load_model_and_task(f"{A}/checkpoints/acs/checkpoint_best.pt")[0]
    base = base[0] if isinstance(base, list) else base
    base.num_updates = 10 ** 9
    clf = ThreeHeadSoftmax(base)
    ck = torch.load(CLF, map_location="cpu")
    clf.load_state_dict(ck["model_state_dict"])
    clf.eval().to(DEV)
    for p in clf.parameters():
        p.requires_grad_(False)
    # The backbone runs its conv feature extractor under torch.no_grad() whenever
    # feature_grad_mult == 0 (ecg_transformer.py:140-147), which silently severs the graph
    # between the input and the loss -- fatal for any input-gradient method. The weights stay
    # frozen (requires_grad False above); this only lets the graph through them.
    clf.base.encoder.feature_grad_mult = 1.0

    vck = torch.load(VAE, map_location="cpu")
    vae = BeatVAE(z_dim=vck["z_dim"], width=vck["width"]).to(DEV)
    vae.load_state_dict(vck["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    print(f"[vae] epoch {vck['epoch']} val_rec {vck['val_rec']:.4f} "
          f"corr/beat {vck['val_corr_per_beat']:.3f} "
          f"recon ACCO AUROC {vck.get('acco_auroc_recon', float('nan')):.3f}", flush=True)
    return clf, vae, ck["stop_percentiles"]


def score_parts(clf, x, k):
    """-> (score to ascend, P(ACCO), P(territory k), territory logits). x: (B,12,250) mV."""
    _, lc, lv = clf(x)
    logit_acco = lc[:, 0]
    other = torch.cat([lv[:, j:j + 1] for j in range(3) if j != k], 1)
    margin = lv[:, k] - torch.logsumexp(other, 1)
    return logit_acco + margin, torch.sigmoid(logit_acco), torch.softmax(lv, 1)[:, k], lv


def main():
    clf, vae, stop = load_models()
    print("[stop] 90th-percentile thresholds from real beats:",
          {k: round(v, 4) for k, v in stop.items()}, flush=True)

    Xte = np.load(f"{CACHE}/test_beats.npy")
    mte = pd.read_parquet(f"{CACHE}/test_beats.parquet")

    # hosts: real NON-ACS beats from the held-out split, one beat per record
    pool = mte.index[(mte.ACS == 0)].to_numpy()
    first = mte.loc[pool].groupby("rec").head(1).index.to_numpy()
    rng = np.random.default_rng(0)
    cand = rng.choice(first, size=min(6 * N_HOSTS, len(first)), replace=False)

    xb = torch.from_numpy(Xte[cand].astype(np.float32)).to(DEV)
    with torch.no_grad():
        mu, _ = vae.encode(xb)
        xr = vae.decode(mu)
        a = xr.cpu().numpy().reshape(len(xr), -1)
        b = xb.cpu().numpy().reshape(len(xb), -1)
        corr = np.array([np.corrcoef(a[i], b[i])[0, 1] for i in range(len(a))])
    keep = np.where(corr >= MIN_RECON_CORR)[0][:N_HOSTS]
    print(f"[hosts] {int((corr >= MIN_RECON_CORR).sum())}/{len(cand)} candidate normal beats "
          f"pass the reconstruction gate (corr >= {MIN_RECON_CORR}); median corr "
          f"{np.median(corr):.3f}; using {len(keep)}", flush=True)
    hosts = xb[torch.from_numpy(keep).to(DEV)]
    host_rec = mte.loc[cand[keep], "rec"].to_numpy()

    # real occluded beats per territory, for the reference trace and the latent-distance check
    real_terr = {}
    for i, c in enumerate(CLASSES):
        sel = mte.index[(mte.single_terr_acco == 1) & (mte.territory == i)].to_numpy()
        xr_ = torch.from_numpy(Xte[sel].astype(np.float32)).to(DEV)
        with torch.no_grad():
            mu_r, _ = vae.encode(xr_)
        real_terr[c] = {"x": Xte[sel].astype(np.float32), "mu": mu_r, "n": len(sel)}
        print(f"[real] {c}: {len(sel)} held-out single-territory occlusion beats", flush=True)

    results = {}
    for k, c in enumerate(CLASSES):
        with torch.no_grad():
            z0, _ = vae.encode(hosts)
            x_t0 = vae.decode(z0)
            s0, p_acco0, p_terr0, _ = score_parts(clf, x_t0, k)
        # Occlusion: the paper's rule, the 90th percentile of real occluded beats (0.71 here).
        # Territory: NOT a percentile. The territory softmax is saturated on real beats
        # (median P(LEFT) and P(RCA) are both 1.000), so its 90th percentile is 0.9998 and a
        # percentile stop would demand exactly the saturation this rule exists to prevent.
        # The interpretable equivalent is "the model now calls this territory": argmax == k
        # with at least half the probability mass.
        thr_acco = stop["acco_p90_real_acco_beats"]
        thr_terr = 0.5

        z = z0.clone()
        done = torch.zeros(len(z), dtype=torch.bool, device=DEV)
        steps_taken = torch.full((len(z),), STEPS, device=DEV, dtype=torch.long)
        traj = []
        for t in range(STEPS):
            z_ = z.detach().requires_grad_(True)
            x = vae.decode(z_)
            s, p_acco, p_terr, lv_all = score_parts(clf, x, k)
            is_argmax = lv_all.detach().argmax(1) == k
            reached = (p_acco >= thr_acco) & (p_terr >= thr_terr) & is_argmax
            newly = reached & (~done)
            steps_taken[newly] = t
            done = done | reached
            if t % 25 == 0 or done.all():
                traj.append({"step": t, "p_acco": float(p_acco.mean()),
                             "p_terr": float(p_terr.mean()), "done": int(done.sum())})
            if done.all():
                break
            g = torch.autograd.grad(s.sum(), z_)[0]
            z = (z + LR * g * (~done).float().unsqueeze(1)).detach()

        with torch.no_grad():
            x_final = vae.decode(z)
            s1, p_acco1, p_terr1, _ = score_parts(clf, x_final, k)
            dz = (z - z0).norm(dim=1)
            d_real = torch.cdist(z0, real_terr[c]["mu"]).min(1).values      # normal -> real occl.
            frac = (dz / d_real).cpu().numpy()

        results[c] = {
            "host": hosts.cpu().numpy(), "t0": x_t0.cpu().numpy(),
            "tfinal": x_final.cpu().numpy(), "host_rec": host_rec,
            "p_acco0": p_acco0.cpu().numpy(), "p_acco1": p_acco1.cpu().numpy(),
            "p_terr0": p_terr0.cpu().numpy(), "p_terr1": p_terr1.cpu().numpy(),
            "steps": steps_taken.cpu().numpy(), "done": done.cpu().numpy(),
            "dz": dz.cpu().numpy(), "d_to_real": d_real.cpu().numpy(), "frac_of_way": frac,
            "real_mean": real_terr[c]["x"].mean(0), "traj": traj,
            "thr_acco": thr_acco, "thr_terr": thr_terr,
        }
        print(f"\n=== morph -> {c} ===")
        print(f"  P(ACCO)      {p_acco0.mean():.3f} -> {p_acco1.mean():.3f}  "
              f"(stop threshold {thr_acco:.3f})")
        print(f"  P({c})".ljust(16) + f"{p_terr0.mean():.3f} -> {p_terr1.mean():.3f}  "
              f"(stop threshold {thr_terr:.3f})")
        print(f"  reached the 90th-percentile stop: {int(done.sum())}/{len(done)} beats, "
              f"median {int(np.median(steps_taken.cpu().numpy()))} steps")
        print(f"  |dz| {dz.mean():.2f}; distance to the nearest real {c} beat "
              f"{d_real.mean():.2f}; travelled {100*np.nanmean(frac):.0f}% of the way",
              flush=True)

    np.savez_compressed(f"{OUT}/morphs.npz",
                        **{f"{c}_{k}": v for c, r in results.items() for k, v in r.items()
                           if isinstance(v, np.ndarray)},
                        classes=np.array(CLASSES), leads=np.array(LEADS))
    summary = {c: {k: (float(np.mean(v)) if isinstance(v, np.ndarray) else v)
                   for k, v in r.items()
                   if k in ("p_acco0", "p_acco1", "p_terr0", "p_terr1", "steps", "dz",
                            "d_to_real", "frac_of_way", "thr_acco", "thr_terr", "traj")}
               for c, r in results.items()}
    for c in CLASSES:
        summary[c]["n_reached_stop"] = int(results[c]["done"].sum())
        summary[c]["n_hosts"] = int(len(results[c]["done"]))
    json.dump(summary, open(f"{OUT}/morph_summary.json", "w"), indent=1, default=float)
    print(f"\n-> {OUT}/morphs.npz and morph_summary.json", flush=True)


if __name__ == "__main__":
    main()
