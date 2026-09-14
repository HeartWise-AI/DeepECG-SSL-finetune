#!/usr/bin/env python3
"""Explainability for the softmax-territory DeepECG-ACS model — generator-free.

Follows the protocol in Obermeyer/Schubert et al., Nature 2026 (s41586-026-10674-6,
in DeepECG-PNG/manuscript/) as adapted in DeepECG-PNG/scripts/explainability/HANDOVER.md,
with that handover's own guidance applied:

  §2  "fix the direction using real data and measure the model. Do not let the model
       choose the direction and then interpret what it chose."  -> no VAE, no gradient
       morph; the perturbation is the measured mean ischemic change of real patients.
  §6  "Do not pool culprit vessels."                            -> everything is
       stratified by territory (LEFT = LAD|Left_Main, RCA, LCX).
  §8/§9 limitations: LCX added, several host seeds, and the injection is applied at the
       real R peaks of intact 10-s records, so there is no beat-tiling artefact.

Analyses
  A. Dose-response: add s x (mean real occluded beat - mean real normal beat) for each
     territory, measured on the TRAIN split, into real normal TEST records. Report ACS,
     ACCO and the territory softmax against dose, over several host samples.
  B. Territory double dissociation: the full 3x3 (injected territory x softmax head)
     matrix at s=0 and s=3.
  C. Negative control: a rectangular ST offset (ST raised, T untouched) at matched ST
     magnitude must NOT move the ACCO head.
  D. Magnitude calibration: the ST deviation delivered at s=1 vs the ST deviation of real
     occlusions in the same leads.
  E. Nature validation step: hand-computable ST/T proxies on REAL held-out records carry
     the same territory signal, per territory, with bootstrap CIs.

Outputs JSON + a figure to data/acs_final/explainability/.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "/volume/DeepECG-PNG/scripts/explainability/beatmodel")
from beats import FS, PRE, BEAT_LEN, detect_r_peaks, median_beat  # noqa: E402

from fairseq_signals.utils import checkpoint_utils  # noqa: E402
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel  # noqa: E402

SCALE = 0.00488
A = "/volume/DeepECG-SSL-finetune/data/acs"
BASE = "/volume/DeepECG-SSL-finetune/data/acs_final"
CKPT = f"{BASE}/checkpoints/acs_3head_softmax/best_model.pt"
SP = "/media/data1/datasets/DeepECG/acs_v6_final_labels.csv"
OUT = f"{BASE}/explainability"
CLASSES = ["LEFT", "RCA", "LCX"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
# classical territory leads (the ones a cardiologist reads for each artery)
TERR_LEADS = {"LEFT": ["V2", "V3", "V4"], "RCA": ["II", "III", "aVF"], "LCX": ["V5", "V6", "I"]}
DOSES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
N_HOSTS = 64
SEEDS = [0, 1, 2]
DEV = "cuda"
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ model
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


def load_model():
    base = checkpoint_utils.load_model_and_task(f"{A}/checkpoints/acs/checkpoint_best.pt")[0]
    base = base[0] if isinstance(base, list) else base
    base.num_updates = 10 ** 9
    m = ThreeHeadSoftmax(base)
    ck = torch.load(CKPT, map_location="cpu")
    m.load_state_dict(ck["model_state_dict"])
    m.eval().to(DEV)
    print(f"loaded {CKPT}  gate={ck['gate']:.6g}  classes={ck['classes']}", flush=True)
    assert ck["classes"] == CLASSES
    return m, float(ck["gate"])


@torch.no_grad()
def score(model, recs, bs=64):
    """recs: (N,12,2500) mV -> (p_acs, p_acco, p_territory[N,3]). One sigmoid/softmax, once."""
    pa, pc, pv = [], [], []
    for i in range(0, len(recs), bs):
        xb = torch.from_numpy(np.ascontiguousarray(recs[i:i + bs], dtype=np.float32)).to(DEV)
        with torch.cuda.amp.autocast():
            la, lc, lv = model(xb)
        pa.append(torch.sigmoid(la.float())[:, 0].cpu().numpy())
        pc.append(torch.sigmoid(lc.float())[:, 0].cpu().numpy())
        pv.append(torch.softmax(lv.float(), 1).cpu().numpy())
    return np.concatenate(pa), np.concatenate(pc), np.concatenate(pv)


# ------------------------------------------------------------------ data
df = pd.read_csv(SP, low_memory=False)
df = df[df.in_database_A == 1].reset_index(drop=True)
df["LEFT"] = ((df.LAD == 1) | (df.Left_Main == 1)).astype(int)


def split_frame(split):
    return df[df.Split == split].reset_index(drop=True)


def territory_index(sub):
    onehot = sub[CLASSES].to_numpy(np.float32)
    n = onehot.sum(1)
    single = (sub.ACCO == 1).to_numpy() & (n == 1)
    idx = np.where(n == 1, onehot.argmax(1), -1)
    return single, idx


def load_mv(split):
    X = np.load(f"{BASE}/arrays/X_{split}.npy", mmap_mode="r")
    return X


def median_beats(X, rows, desc=""):
    """rows: indices into X. -> (n_ok,12,BEAT_LEN) median beats in mV, and the kept rows."""
    out, kept = [], []
    for j, r in enumerate(rows):
        sig = np.asarray(X[r], dtype=np.float32).T * SCALE          # (12,2500) mV
        mb = median_beat(sig)
        if mb is not None:
            out.append(mb)
            kept.append(r)
        if desc and (j + 1) % 500 == 0:
            print(f"  {desc}: {j+1}/{len(rows)}", flush=True)
    return (np.stack(out) if out else np.zeros((0, 12, BEAT_LEN), np.float32)), np.array(kept)


# ST / T proxy windows, referenced to R at index PRE
ST0, ST1 = PRE + int(0.08 * FS), PRE + int(0.16 * FS)
T0, T1 = PRE + int(0.16 * FS), PRE + int(0.40 * FS)


def proxy(beats):
    """(N,12,L) -> dict of (N,12) ST level and T amplitude."""
    return {"st": beats[:, :, ST0:ST1].mean(2), "t": beats[:, :, T0:T1].max(2)}


def inject(rec_mv, delta, s, peaks=None):
    """Add s*delta at every detected R peak of an intact 10-s record. rec (12,2500), delta (12,L).

    `peaks` is detected once on the clean host and reused at every dose, so the doses differ
    only in the amount of injected morphology.
    """
    out = rec_mv.copy()
    for p in (detect_r_peaks(rec_mv) if peaks is None else peaks):
        a, b = p - PRE, p - PRE + delta.shape[1]
        if a < 0 or b > out.shape[1]:
            continue
        out[:, a:b] += s * delta
    return out


def rect_st(rec_mv, leads, mv):
    """Negative control: raise the ST segment only, leaving the T wave untouched."""
    out = rec_mv.copy()
    li = [LEADS.index(L) for L in leads]
    for p in detect_r_peaks(rec_mv):
        a, b = p + int(0.08 * FS), p + int(0.16 * FS)
        if a < 0 or b > out.shape[1]:
            continue
        out[li, a:b] += mv
    return out


def boot_mean(v, n=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = np.arange(len(v))
    b = np.array([v[rng.choice(idx, len(idx), True)].mean() for _ in range(n)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_auroc(y, p, n=1000, rng=None):
    rng = rng or np.random.default_rng(0)
    idx = np.arange(len(y))
    aus = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), True)
        if 0 < y[b].sum() < len(b):
            aus.append(roc_auc_score(y[b], p[b]))
    return float(roc_auc_score(y, p)), float(np.percentile(aus, 2.5)), float(np.percentile(aus, 97.5))


def main():
    model, gate = load_model()
    RES = {"gate": gate, "classes": CLASSES, "doses": DOSES, "n_hosts": N_HOSTS, "seeds": SEEDS,
           "territory_leads": TERR_LEADS}

    # ---------------- ischemic patterns, measured on TRAIN (never on the hosts' split)
    tr = split_frame("train")
    Xtr = load_mv("train")
    single_tr, idx_tr = territory_index(tr)
    normal_rows = np.where((tr.ACS == 0).to_numpy())[0]
    rng0 = np.random.default_rng(0)
    normal_rows = rng0.choice(normal_rows, 2000, replace=False)
    print(f"train: {int(single_tr.sum())} single-territory ACCO, 2000 sampled normals", flush=True)
    nb, _ = median_beats(Xtr, normal_rows, "train normals")
    nor_mean = nb.mean(0)
    RES["n_train_normal_beats"] = int(len(nb))

    deltas, real_occ_st = {}, {}
    for i, c in enumerate(CLASSES):
        rows = np.where(single_tr & (idx_tr == i))[0]
        ob, _ = median_beats(Xtr, rows, f"train {c}")
        deltas[c] = ob.mean(0) - nor_mean
        li = [LEADS.index(L) for L in TERR_LEADS[c]]
        p_occ, p_nor = proxy(ob), proxy(nb)
        real_occ_st[c] = float(p_occ["st"][:, li].mean() - p_nor["st"][:, li].mean())
        RES.setdefault("pattern", {})[c] = {
            "n_train_occlusions": int(len(ob)),
            "real_ST_in_territory_mV": real_occ_st[c],
            "dST_by_lead_mV": {L: float(p_occ["st"][:, j].mean() - p_nor["st"][:, j].mean())
                               for j, L in enumerate(LEADS)},
            "dT_by_lead_mV": {L: float(p_occ["t"][:, j].mean() - p_nor["t"][:, j].mean())
                              for j, L in enumerate(LEADS)},
        }
        print(f"pattern {c}: n={len(ob)}  real ST in {TERR_LEADS[c]} = {real_occ_st[c]:+.3f} mV",
              flush=True)
    np.savez(f"{OUT}/territory_patterns.npz", **{c: deltas[c] for c in CLASSES},
             normal_mean_beat=nor_mean)

    # HANDOVER §6: show the patterns do NOT survive pooling
    pooled = np.mean([deltas[c] for c in CLASSES], 0)
    RES["pooling_check"] = {
        "note": "mean of the three territory patterns; per-lead ST cancels when pooled",
        "pooled_dST_by_lead_mV": {L: float(pooled[j, ST0:ST1].mean()) for j, L in enumerate(LEADS)},
        "corr_LEFT_RCA_dST": float(np.corrcoef(
            deltas["LEFT"][:, ST0:ST1].mean(1), deltas["RCA"][:, ST0:ST1].mean(1))[0, 1]),
        "corr_LEFT_LCX_dST": float(np.corrcoef(
            deltas["LEFT"][:, ST0:ST1].mean(1), deltas["LCX"][:, ST0:ST1].mean(1))[0, 1]),
        "corr_RCA_LCX_dST": float(np.corrcoef(
            deltas["RCA"][:, ST0:ST1].mean(1), deltas["LCX"][:, ST0:ST1].mean(1))[0, 1]),
    }

    # ---------------- hosts: real normal TEST records
    te = split_frame("test")
    Xte = load_mv("test")
    single_te, idx_te = territory_index(te)
    host_pool = np.where((te.ACS == 0).to_numpy())[0]

    # A + B: dose-response and the territory matrix, over seeds
    RES["dose_response"] = {}
    for c in CLASSES:
        RES["dose_response"][c] = {"dose": DOSES, "acs": [], "acco": [],
                                   "territory": {k: [] for k in CLASSES}, "st_territory": [],
                                   "acs_sd": [], "acco_sd": [], "st_territory_sd": [],
                                   "territory_sd": {k: [] for k in CLASSES}}
    for c in CLASSES:
        li = [LEADS.index(L) for L in TERR_LEADS[c]]
        per_seed = {s: {"acs": [], "acco": [], "ves": [], "st": []} for s in DOSES}
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            rows = rng.choice(host_pool, N_HOSTS * 2, replace=False)
            hosts, peaks = [], []
            for r in rows:
                h = np.asarray(Xte[r], np.float32).T * SCALE
                pk = detect_r_peaks(h)
                if len(pk) >= 3 and median_beat(h) is not None:
                    hosts.append(h)
                    peaks.append(pk)
                if len(hosts) == N_HOSTS:
                    break
            hosts = np.stack(hosts)
            base_st = None
            for s in DOSES:
                recs = np.stack([inject(h, deltas[c], s, pk) for h, pk in zip(hosts, peaks)])
                pa, pc, pv = score(model, recs)
                mbs = np.stack([mb for mb in (median_beat(r) for r in recs) if mb is not None])
                st_now = proxy(mbs)["st"][:, li].mean()
                if s == 0.0:
                    base_st = st_now
                per_seed[s]["acs"].append(pa.mean())
                per_seed[s]["acco"].append(pc.mean())
                per_seed[s]["ves"].append(pv.mean(0))
                per_seed[s]["st"].append(float(st_now - base_st))
            print(f"[{c}] seed {seed} done", flush=True)
        for s in DOSES:
            d = RES["dose_response"][c]
            d["acs"].append(float(np.mean(per_seed[s]["acs"])))
            d["acs_sd"].append(float(np.std(per_seed[s]["acs"])))
            d["acco"].append(float(np.mean(per_seed[s]["acco"])))
            d["acco_sd"].append(float(np.std(per_seed[s]["acco"])))
            d["st_territory"].append(float(np.mean(per_seed[s]["st"])))
            d["st_territory_sd"].append(float(np.std(per_seed[s]["st"])))
            v = np.stack(per_seed[s]["ves"])
            for k, cl in enumerate(CLASSES):
                d["territory"][cl].append(float(v[:, k].mean()))
                d["territory_sd"][cl].append(float(v[:, k].std()))
        d = RES["dose_response"][c]
        print(f"\n=== injected {c} pattern (mean of {len(SEEDS)} host samples of {N_HOSTS}) ===")
        print(f"{'s':>4} {'dST terr':>9} {'ACS':>7} {'ACCO':>7} " +
              " ".join(f"{'P('+cl+')':>9}" for cl in CLASSES))
        for j, s in enumerate(DOSES):
            print(f"{s:4.1f} {d['st_territory'][j]:+9.3f} {d['acs'][j]:7.3f} {d['acco'][j]:7.3f} " +
                  " ".join(f"{d['territory'][cl][j]:9.3f}" for cl in CLASSES), flush=True)
        RES["pattern"][c]["delivered_ST_at_s1_mV"] = d["st_territory"][DOSES.index(1.0)]

    # ---------------- C: negative control, rectangular ST offset
    RES["negative_control"] = {}
    rng = np.random.default_rng(0)
    rows = rng.choice(host_pool, N_HOSTS * 2, replace=False)
    nc_hosts = []
    for r in rows:
        h = np.asarray(Xte[r], np.float32).T * SCALE
        if len(detect_r_peaks(h)) >= 3:
            nc_hosts.append(h)
        if len(nc_hosts) == N_HOSTS:
            break
    nc_hosts = np.stack(nc_hosts)
    for c in CLASSES:
        hosts = nc_hosts
        out = {"mv": [], "acco": [], "acs": []}
        for mv in (0.0, 0.10, 0.20, 0.30):
            recs = np.stack([rect_st(h, TERR_LEADS[c], mv) for h in hosts])
            pa, pc, _ = score(model, recs)
            out["mv"].append(mv)
            out["acco"].append(float(pc.mean()))
            out["acs"].append(float(pa.mean()))
        RES["negative_control"][c] = out
        print(f"neg control {c}: ACCO {out['acco']} at ST offsets {out['mv']} mV", flush=True)

    # ---------------- E: interpretable proxy on REAL held-out records
    print("\nproxy validation on real test records ...", flush=True)
    occ_rows = np.where(single_te)[0]
    norm_rows = np.random.default_rng(0).choice(np.where((te.ACS == 0).to_numpy())[0], 1500, False)
    ob_te, ob_keep = median_beats(Xte, occ_rows, "test occlusions")
    nb_te, _ = median_beats(Xte, norm_rows, "test normals")
    y_terr = idx_te[ob_keep]
    p_occ, p_nor = proxy(ob_te), proxy(nb_te)
    RES["proxy_validation"] = {}
    for i, c in enumerate(CLASSES):
        li = [LEADS.index(L) for L in TERR_LEADS[c]]
        # (i) territory vs normal, using that territory's own ST+T leads
        f_occ = p_occ["st"][y_terr == i][:, li].mean(1) + p_occ["t"][y_terr == i][:, li].mean(1)
        f_nor = p_nor["st"][:, li].mean(1) + p_nor["t"][:, li].mean(1)
        y = np.r_[np.ones(len(f_occ)), np.zeros(len(f_nor))]
        au, lo, hi = boot_auroc(y, np.r_[f_occ, f_nor])
        # (ii) this territory vs the other occluded territories (localisation, not detection)
        f_all = p_occ["st"][:, li].mean(1) + p_occ["t"][:, li].mean(1)
        yl = (y_terr == i).astype(int)
        aul, lol, hil = boot_auroc(yl, f_all)
        RES["proxy_validation"][c] = {
            "n_occlusions": int((y_terr == i).sum()), "leads": TERR_LEADS[c],
            "vs_normal_auroc": au, "vs_normal_ci": [lo, hi],
            "vs_other_territories_auroc": aul, "vs_other_territories_ci": [lol, hil]}
        print(f"  {c:5s} n={int((y_terr==i).sum()):3d}  ST+T proxy vs normal "
              f"AUROC {au:.2f} ({lo:.2f}-{hi:.2f});  vs other territories "
              f"{aul:.2f} ({lol:.2f}-{hil:.2f})", flush=True)

    # ---------------- model's own territory discrimination on the same real records
    recs = np.stack([np.asarray(Xte[r], np.float32).T * SCALE for r in occ_rows])
    pa, pc, pv = score(model, recs)
    RES["model_on_real"] = {"n": int(len(occ_rows))}
    for i, c in enumerate(CLASSES):
        y = (idx_te[occ_rows] == i).astype(int)
        au, lo, hi = boot_auroc(y, pv[:, i])
        RES["model_on_real"][c] = {"auroc": au, "ci": [lo, hi], "n_pos": int(y.sum())}
        print(f"  model {c:5s} AUROC {au:.2f} ({lo:.2f}-{hi:.2f})", flush=True)

    with open(f"{OUT}/explainability_results.json", "w") as f:
        json.dump(RES, f, indent=1)
    print(f"\n-> {OUT}/explainability_results.json", flush=True)
    return RES


if __name__ == "__main__":
    main()
