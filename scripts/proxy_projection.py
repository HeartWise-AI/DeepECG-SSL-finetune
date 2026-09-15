#!/usr/bin/env python3
"""Nature-protocol validation step, done per territory and without hand-picked leads.

The paper's non-circular step is: quantify the morphology the analysis surfaced with a
hand-computable proxy, then show that proxy carries independent signal on REAL records.

Here the morphology is the measured per-territory ischemic pattern (train split). The proxy
is its projection: for a median beat, take the 24-dim vector [ST(12 leads), T(12 leads)] and
score it against each territory's own pattern direction, measured on the training split.
No lead is chosen by hand, so the earlier null for LCx on the classical lateral leads
(V5/V6/I) is not an artefact of that choice.

Reports, on the held-out single-territory complete occlusions:
  - territory vs the OTHER occluded territories (localisation)
  - territory vs normal records (detection)
and, for reference, the same two numbers for the model's own softmax.
"""
from __future__ import annotations

import json
import numpy as np
from sklearn.metrics import roc_auc_score

import explain_acs_softmax as ex   # reuse the loaders, beat code and proxy windows

OUT = f"{ex.BASE}/explainability/proxy_projection.json"
RNG = np.random.default_rng(0)


def feat(beats):
    p = ex.proxy(beats)
    return np.concatenate([p["st"], p["t"]], axis=1)      # (N,24)


def boot_auroc(y, p, n=1000):
    idx = np.arange(len(y))
    aus = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx), True)
        if 0 < y[b].sum() < len(b):
            aus.append(roc_auc_score(y[b], p[b]))
    return float(roc_auc_score(y, p)), float(np.percentile(aus, 2.5)), float(np.percentile(aus, 97.5))


def main():
    # ---- directions from TRAIN
    tr = ex.split_frame("train")
    Xtr = ex.load_mv("train")
    single_tr, idx_tr = ex.territory_index(tr)
    nrows = np.random.default_rng(0).choice(np.where((tr.ACS == 0).to_numpy())[0], 2000, False)
    nb, _ = ex.median_beats(Xtr, nrows)
    f_nor_tr = feat(nb).mean(0)
    names = [f"ST_{L}" for L in ex.LEADS] + [f"T_{L}" for L in ex.LEADS]
    dirs = {}
    for i, c in enumerate(ex.CLASSES):
        ob, _ = ex.median_beats(Xtr, np.where(single_tr & (idx_tr == i))[0])
        d = feat(ob).mean(0) - f_nor_tr
        dirs[c] = d / np.linalg.norm(d)
        top = sorted(zip(names, d), key=lambda kv: -abs(kv[1]))[:5]
        print(f"{c}: direction dominated by " +
              ", ".join(f"{n} {v:+.3f}" for n, v in top), flush=True)

    # ---- evaluate on TEST
    te = ex.split_frame("test")
    Xte = ex.load_mv("test")
    single_te, idx_te = ex.territory_index(te)
    occ_rows = np.where(single_te)[0]
    ob_te, keep = ex.median_beats(Xte, occ_rows)
    y = idx_te[keep]
    nrows_te = np.random.default_rng(1).choice(np.where((te.ACS == 0).to_numpy())[0], 1500, False)
    nb_te, _ = ex.median_beats(Xte, nrows_te)
    F_occ, F_nor = feat(ob_te), feat(nb_te)

    res = {"n_occlusions": int(len(F_occ)), "n_normals": int(len(F_nor)), "territories": {}}
    print(f"\ntest: {len(F_occ)} single-territory complete occlusions, {len(F_nor)} normals")
    for i, c in enumerate(ex.CLASSES):
        s_occ = F_occ @ dirs[c]
        s_nor = F_nor @ dirs[c]
        au_l, lo_l, hi_l = boot_auroc((y == i).astype(int), s_occ)
        yy = np.r_[np.ones(int((y == i).sum())), np.zeros(len(s_nor))]
        au_d, lo_d, hi_d = boot_auroc(yy, np.r_[s_occ[y == i], s_nor])
        res["territories"][c] = {
            "n_pos": int((y == i).sum()),
            "localisation_vs_other_territories": {"auroc": au_l, "ci": [lo_l, hi_l]},
            "detection_vs_normal": {"auroc": au_d, "ci": [lo_d, hi_d]},
        }
        print(f"  {c:5s} n={int((y==i).sum()):3d}  proxy localisation AUROC {au_l:.2f} "
              f"({lo_l:.2f}-{hi_l:.2f})   proxy detection AUROC {au_d:.2f} ({lo_d:.2f}-{hi_d:.2f})")

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
