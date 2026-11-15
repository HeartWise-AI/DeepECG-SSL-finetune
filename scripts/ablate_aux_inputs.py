#!/usr/bin/env python3
"""How much do the age/sex inputs contribute? Inference-time ablation for the WCR v2 aux-input models.

Scores each test split twice with the same checkpoint: with the recorded age and sex, and with both
set to unknown (NaN -> zero + missingness flag, as at inference without demographics). Reports
per-label AUROC for both and the paired delta with a patient-level bootstrap 95 % CI. Optionally
compares against a reference model's saved outputs on the same rows (e.g. WCR v1 wcr_afib_5y).

This measures what the trained model loses when age/sex are withheld; it is not the same as a model
trained without them (see the waveform-only control run for that).

    PYTHONPATH=$PWD /volume/venvs/fss39/bin/python scripts/ablate_aux_inputs.py \
        --model-dir /media/data1/models/DeepECG-SSL/DeepECG_SSL_WCRv2_EchoNext12_v1 \
        --data data/wcrv2_echonext12 --subsets echonext_test mhi_test --out eval_aux_ablation
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_wcrv2_aux import load, predict  # noqa: E402


def load_memmap(path):
    h = pickle.load(open(str(path).replace(".npy", "_header.pkl"), "rb"))
    return np.asarray(np.memmap(path, dtype=h["dtype"], mode="r", shape=tuple(h["shape"])), dtype=np.float64)


def paired_boot(y, a, b, groups, B=1000, seed=0):
    """AUROC(a) - AUROC(b) on the same rows; patients resampled as whole groups."""
    rng = np.random.default_rng(seed)
    _, gi = np.unique(groups, return_inverse=True)
    ng = gi.max() + 1
    d = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, ng, ng), minlength=ng)[gi]
        k = w > 0
        if y[k].min() == y[k].max():
            continue
        d.append(roc_auc_score(y[k], a[k], sample_weight=w[k]) - roc_auc_score(y[k], b[k], sample_weight=w[k]))
    d = np.asarray(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(2 * min((d <= 0).mean(), (d >= 0).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--subsets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ref", default=None, help="dir with outputs_<subset>.npy of a reference model")
    ap.add_argument("--ref-map", default=None, help='JSON {"ref_col": "label"}')
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    m = load(args.model_dir, device=args.device)
    mu, sd = m["aux_norm"]["age_mean"], m["aux_norm"]["age_std"]
    labels = m["labels"]
    rows = []
    for s in args.subsets:
        x = np.load(data / "data" / f"{s}_x.npy", mmap_mode="r")
        y = np.load(data / "data" / f"{s}_y.npy")
        aux = np.load(data / "data" / f"{s}_aux.npy")
        pid = pd.read_parquet(data / "data" / f"{s}_meta.parquet", columns=["patient_id"])["patient_id"].astype(str).to_numpy()
        age, sex = aux[:, 0] * sd + mu, aux[:, 1]
        cache = out / f"{s}_probs.npz"
        if cache.exists():
            z = np.load(cache)
            p_with, p_without = z["with_aux"], z["without_aux"]
        else:
            pw, pn = [], []
            for i0 in range(0, len(x), 8192):
                xb = np.asarray(x[i0:i0 + 8192])
                pw.append(predict(m, xb, age[i0:i0 + 8192], sex[i0:i0 + 8192]))
                pn.append(predict(m, xb, None, None))
                print(f"[{s}] {min(i0 + 8192, len(x)):,}/{len(x):,}", flush=True)
            p_with, p_without = np.concatenate(pw), np.concatenate(pn)
            np.savez(cache, with_aux=p_with, without_aux=p_without)
        ref = None
        if args.ref and (Path(args.ref) / f"outputs_{s}.npy").exists():
            ref = load_memmap(Path(args.ref) / f"outputs_{s}.npy")
            ref_map = {int(k): v for k, v in json.loads(args.ref_map).items()}
        known = ~np.isnan(aux[:, 0]) & ~np.isnan(aux[:, 1])
        for j, lab in enumerate(labels):
            k = ~np.isnan(y[:, j])
            if k.sum() < 50 or y[k, j].min() == y[k, j].max():
                continue
            yt = y[k, j]
            a_w, a_n = roc_auc_score(yt, p_with[k, j]), roc_auc_score(yt, p_without[k, j])
            lo, hi, p = paired_boot(yt, p_with[k, j], p_without[k, j], pid[k], B=args.B)
            r = {"subset": s, "label": lab, "n": int(k.sum()), "n_pos": int(yt.sum()),
                 "pct_age_sex_known": round(100 * float(known[k].mean()), 1),
                 "auroc_with_age_sex": round(a_w, 4), "auroc_age_sex_withheld": round(a_n, 4),
                 "delta": round(a_w - a_n, 4), "delta_ci": f"{lo:+.4f} to {hi:+.4f}", "p_boot": round(p, 4),
                 "mean_abs_dprob": round(float(np.abs(p_with[k, j] - p_without[k, j]).mean()), 4)}
            if ref is not None:
                for c, rl in ref_map.items():
                    if rl == lab:
                        rr = ref[k, c]
                        a_r = roc_auc_score(yt, rr)
                        r["auroc_ref"] = round(a_r, 4)
                        for nm, pp in (("with", p_with[k, j]), ("withheld", p_without[k, j])):
                            lo2, hi2, p2 = paired_boot(yt, pp, rr, pid[k], B=args.B)
                            r[f"delta_{nm}_vs_ref"] = round(roc_auc_score(yt, pp) - a_r, 4)
                            r[f"delta_{nm}_vs_ref_ci"] = f"{lo2:+.4f} to {hi2:+.4f}"
                            r[f"p_{nm}_vs_ref"] = round(p2, 4)
            rows.append(r)
            print(r, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "aux_ablation.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
