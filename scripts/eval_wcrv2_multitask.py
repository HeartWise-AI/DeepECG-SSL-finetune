#!/usr/bin/env python3
"""Evaluate a WCR v2 multi-task checkpoint on the combined and per-source test sets.

Runs ``fairseq-hydra-inference`` for each requested manifest, then reports
per-label AUROC / AUPRC (patient-level bootstrap 95 % CI), Youden-J and sensitivity-0.90
operating points, restricted to rows where the label is defined (NaN-masked).
Optionally (--baselines) evaluates the two reference models on the same rows:

  * EchoNext v6 7-label model      (/media/data1/models/DeepECG-SSL/DeepECG_SSL_WCRv2_EchoNext_v6)
  * WCR v1 incident-AF 5y model    (/media/data1/models/DeepECG-SSL/wcr_afib_5y/wcr_afib_5y.pt)

Usage:
    PYTHONPATH=/volume/DeepECG-SSL-finetune /volume/venvs/fss39/bin/python scripts/eval_wcrv2_multitask.py \
        --ckpt data/runs/<RUN>/checkpoints/checkpoint_best.pt --data data/wcrv2_multitask \
        --subsets test echonext_test mhi_test --device 0
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[1]
CFG_DIR = REPO / "examples/w2v_cmsc/config/finetuning/ecg_transformer"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def load_memmap(path):
    # the header is a pickle written by fairseq_signals.utils.store for our own inference runs;
    # only point this at result directories you produced
    with open(str(path).replace(".npy", "_header.pkl"), "rb") as f:
        h = pickle.load(f)
    return np.asarray(np.memmap(path, dtype=h["dtype"], mode="r", shape=tuple(h["shape"])), dtype=np.float32)


def _cache_key(ckpt, manifest_dir, subset, num_labels):
    ck = Path(ckpt)
    man = Path(manifest_dir) / f"{subset}.tsv"
    return {"ckpt": str(ck.resolve()), "ckpt_mtime": ck.stat().st_mtime, "ckpt_size": ck.stat().st_size,
            "manifest": str(man.resolve()), "manifest_mtime": man.stat().st_mtime, "num_labels": int(num_labels)}


def run_inference(py, ckpt, manifest_dir, subset, results, num_labels, device, batch_size=256,
                  expected_rows=None, force=False):
    """Run fairseq-hydra-inference unless a cached output for the same checkpoint, manifest and size exists."""
    results.mkdir(parents=True, exist_ok=True)
    out = results / f"outputs_{subset}.npy"
    meta = results / f"outputs_{subset}.cache.json"
    key = _cache_key(ckpt, manifest_dir, subset, num_labels)
    if out.exists() and not force:
        try:
            cached = json.load(open(meta))
            rows_ok = expected_rows is None or load_memmap(out).shape[0] == expected_rows
            if cached == key and rows_ok:
                return out
            print(f"[cache] {out} is stale or incomplete; recomputing", flush=True)
        except (OSError, ValueError, KeyError):
            print(f"[cache] {out} has no valid cache record; recomputing", flush=True)
        for f in (out, Path(str(out).replace(".npy", "_header.pkl")), meta):
            if f.exists():
                f.unlink()
    env = dict(os.environ, PYTHONPATH=str(REPO), CUDA_VISIBLE_DEVICES=str(device), WANDB_MODE="disabled",
               HYDRA_FULL_ERROR="1")
    # argument list, no shell: values are passed verbatim to the interpreter
    cmd = [py, "-m", "fairseq_cli.hydra_inference", "--config-dir", str(CFG_DIR), "--config-name", "eval",
           f"task.data={manifest_dir}", f"common_eval.path={ckpt}", f"common_eval.results_path={results}",
           "task.npy_dataset=true", f"model.num_labels={num_labels}", f"dataset.valid_subset={subset}",
           f"dataset.batch_size={batch_size}", "dataset.num_workers=6", "common.wandb_project=null"]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)
    json.dump(key, open(meta, "w"), indent=2)
    return out


def boot_ci(y, s, fn, B=500, seed=0, groups=None):
    """Percentile bootstrap CI. With ``groups`` (e.g. patient ids) whole groups are resampled, so the
    several ECGs of one patient stay together; implemented as integer sample weights (= replication)."""
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    if groups is not None:
        g_codes, g_idx = np.unique(groups, return_inverse=True)
        n_g = len(g_codes)
    for _ in range(B):
        if groups is None:
            i = rng.integers(0, n, n)
            if y[i].min() == y[i].max():
                continue
            vals.append(fn(y[i], s[i]))
        else:
            w = np.bincount(rng.integers(0, n_g, n_g), minlength=n_g)[g_idx]
            keep = w > 0
            if y[keep].min() == y[keep].max():
                continue
            vals.append(fn(y[keep], s[keep], sample_weight=w[keep]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def per_label_metrics(y, p, labels, ci=True, groups=None):
    rows = []
    for j, lab in enumerate(labels):
        m = ~np.isnan(y[:, j])
        yt, ps = y[m, j], p[m, j]
        if m.sum() == 0 or yt.min() == yt.max():
            rows.append({"label": lab, "n": int(m.sum()), "n_pos": int(np.nansum(yt)) if m.sum() else 0})
            continue
        auroc = roc_auc_score(yt, ps)
        auprc = average_precision_score(yt, ps)
        fpr, tpr, thr = roc_curve(yt, ps)
        j_idx = int(np.argmax(tpr - fpr))
        se90 = np.where(tpr >= 0.90)[0]
        r = {"label": lab, "n": int(m.sum()), "n_pos": int(yt.sum()), "prev": float(yt.mean()),
             "auroc": float(auroc), "auprc": float(auprc),
             "youden_thr": float(thr[j_idx]), "youden_sens": float(tpr[j_idx]), "youden_spec": float(1 - fpr[j_idx]),
             "sens90_thr": float(thr[se90[0]]) if len(se90) else np.nan,
             "sens90_spec": float(1 - fpr[se90[0]]) if len(se90) else np.nan}
        if ci:
            lo, hi = boot_ci(yt, ps, roc_auc_score, groups=None if groups is None else groups[m])
            r["auroc_ci"] = f"{lo:.3f}-{hi:.3f}"
            r["ci_unit"] = "row" if groups is None else "patient"
        rows.append(r)
    return pd.DataFrame(rows)


def patient_groups(data, subset, n):
    """Patient ids aligned with the first n rows of a split, or None when the meta parquet is absent."""
    mp = data / "data" / f"{subset}_meta.parquet"
    if not mp.exists():
        return None
    pid = pd.read_parquet(mp, columns=["patient_id"])["patient_id"].astype(str).to_numpy()
    return pid[:n] if len(pid) >= n else None


# Reference models evaluated on the same rows (section 6 of the plan). Both consume mV waveforms
# (v6: WCR v2; wcr_afib_5y: WCR v1 fine-tuned with scale 0.00488, i.e. also mV), so the same x arrays apply.
BASELINES = {
    "echonext_v6": {
        "ckpt": "/media/data1/models/DeepECG-SSL/DeepECG_SSL_WCRv2_EchoNext_v6/weights/checkpoint_best.pt",
        "num_labels": 7,
        # v6 output index -> our head name
        "map": {0: "mr_mod_plus", 1: "as_mod_plus", 2: "ar_mod_plus", 3: "tr_mod_plus", 4: "lvef_lte_45", 5: "rv_dysf_mod_plus"},
        # v6 has no composite head; its documented SHD composite is the max over all 7 head probabilities
        # (incl. index 6, LVWT >= 15 mm, which has no counterpart in our label set)
        "composite_from_max": "shd_composite",
    },
    "wcr_v1_afib_5y": {
        "ckpt": "/media/data1/models/DeepECG-SSL/wcr_afib_5y/wcr_afib_5y.pt",
        "num_labels": 1,
        "map": {0: "incident_af_5y"},
    },
}
IAF5_PARQUET = "/media/data1/datasets/DeepECG/DeepECG-PNG/downstream/incident_afib5y.parquet"


def paired_report(name, y, p_ours, ref_cols, labels, rows_mask=None):
    """AUROC of ours vs a reference on exactly the rows where both exist and the label is defined."""
    out = []
    for j, lab in enumerate(labels):
        if lab not in ref_cols:
            continue
        m = ~np.isnan(y[:, j]) & ~np.isnan(ref_cols[lab])
        if rows_mask is not None:
            m &= rows_mask
        if m.sum() < 50 or y[m, j].min() == y[m, j].max():
            continue
        a_ours = roc_auc_score(y[m, j], p_ours[m, j])
        a_ref = roc_auc_score(y[m, j], ref_cols[lab][m])
        out.append({"label": lab, "n": int(m.sum()), "n_pos": int(y[m, j].sum()),
                    "auroc_ours": round(float(a_ours), 4), f"auroc_{name}": round(float(a_ref), 4),
                    "delta": round(float(a_ours - a_ref), 4)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="our checkpoint; omit with --baselines to score only the reference models")
    ap.add_argument("--data", default=str(REPO / "data/wcrv2_multitask"))
    ap.add_argument("--subsets", nargs="+", default=["test", "echonext_test", "mhi_test"])
    ap.add_argument("--device", default="0")
    ap.add_argument("--results", default=None, help="where inference outputs go (default: next to the checkpoint)")
    ap.add_argument("--no-ci", action="store_true")
    ap.add_argument("--baselines", action="store_true", help="also run EchoNext v6 and WCR v1 AF-5y on the same rows")
    ap.add_argument("--force", action="store_true", help="ignore cached inference outputs")
    ap.add_argument("--py", default=sys.executable)
    args = ap.parse_args()

    # hydra changes the working directory, so every path handed to the subprocess must be absolute
    data = Path(args.data).resolve()
    if args.ckpt:
        args.ckpt = str(Path(args.ckpt).resolve())
    if args.results:
        args.results = str(Path(args.results).resolve())
    info = json.load(open(data / "labels.json"))
    labels = info["labels"]
    if args.ckpt is None and not args.baselines:
        ap.error("--ckpt is required unless --baselines is given")
    results = Path(args.results) if args.results else (Path(args.ckpt).parent.parent / "eval" if args.ckpt else data / "baselines")
    results.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    summary = {}
    for subset in args.subsets:
        y = np.load(data / "data" / f"{subset}_y.npy")
        summary[subset] = {}
        if args.ckpt is None:
            # baselines only: standalone per-label metrics of each reference model on our labels
            n = len(y)
            for name, b in BASELINES.items():
                if not os.path.exists(b["ckpt"]):
                    print(f"[skip] {name}: {b['ckpt']} not found")
                    continue
                bout = run_inference(args.py, b["ckpt"], data / "manifests", subset, results / name, b["num_labels"],
                                     args.device, expected_rows=n, force=args.force)
                bp = sigmoid(load_memmap(bout)[:n])
                cols = {lab: bp[:, k] for k, lab in b["map"].items()}
                if b.get("composite_from_max"):
                    cols[b["composite_from_max"]] = bp.max(axis=1)
                sub_labels = [l for l in labels if l in cols]
                pm = np.stack([cols[l] for l in sub_labels], axis=1)
                ym = y[:n][:, [labels.index(l) for l in sub_labels]]
                df = per_label_metrics(ym, pm, sub_labels, ci=not args.no_ci, groups=patient_groups(data, subset, n))
                df.to_csv(results / f"metrics_{name}_{subset}.csv", index=False)
                print(f"\n===== {name} on {subset} (n={n:,}) =====")
                print(df.round(4).to_string(index=False))
                if "auroc" in df.columns:  # e.g. the AF model has no evaluable label on EchoNext rows
                    summary[subset][name] = df.set_index("label")["auroc"].round(4).to_dict()
                else:
                    summary[subset][name] = {}
            continue
        out = run_inference(args.py, args.ckpt, data / "manifests", subset, results, len(labels), args.device,
                            expected_rows=len(y), force=args.force)
        logits = load_memmap(out)
        n = min(len(logits), len(y))
        p = sigmoid(logits[:n])
        df = per_label_metrics(y[:n], p, labels, ci=not args.no_ci, groups=patient_groups(data, subset, n))
        df.to_csv(results / f"metrics_{subset}.csv", index=False)
        ok = df["auroc"].dropna() if "auroc" in df.columns else pd.Series(dtype=float)
        print(f"\n===== {subset}  (n={n:,})  macro AUROC {ok.mean():.4f} over {len(ok)} labels =====")
        print(df.round(4).to_string(index=False))
        summary[subset].update({"n": int(n), "macro_auroc": float(ok.mean()) if len(ok) else None,
                                "per_label": df.set_index("label")["auroc"].round(4).to_dict() if "auroc" in df.columns else {}})

        # incident-AF heads on the curated iAF5 test rows (comparability with DeepECG-PNG numbers)
        meta_path = data / "data" / f"{subset}_meta.parquet"
        if subset in ("test", "mhi_test") and meta_path.exists() and os.path.exists(IAF5_PARQUET):
            meta = pd.read_parquet(meta_path)
            iaf5 = pd.read_parquet(IAF5_PARQUET, columns=["npy_path", "Split"])
            in_iaf5 = meta["ecg_id"].isin(set(iaf5.loc[iaf5["Split"] == "test", "npy_path"])).to_numpy()[:n]
            for lab in ("incident_af_2y", "incident_af_5y"):
                if lab not in labels:
                    continue
                j = labels.index(lab)
                m = in_iaf5 & ~np.isnan(y[:n, j])
                if m.sum() > 50 and y[m, j].min() != y[m, j].max():
                    a = roc_auc_score(y[m, j], p[m, j])
                    print(f"[iAF5 test rows] {lab}: n={m.sum():,} pos={int(y[m, j].sum()):,} AUROC {a:.4f}")
                    summary[subset][f"{lab}_on_iaf5_test"] = {"n": int(m.sum()), "auroc": round(float(a), 4)}

        if args.baselines:
            for name, b in BASELINES.items():
                if not os.path.exists(b["ckpt"]):
                    print(f"[skip] {name}: {b['ckpt']} not found")
                    continue
                bout = run_inference(args.py, b["ckpt"], data / "manifests", subset, results / name, b["num_labels"],
                                     args.device, expected_rows=len(y), force=args.force)
                bl = load_memmap(bout)[:n]
                bp = sigmoid(bl)
                ref_cols = {lab: bp[:, k] for k, lab in b["map"].items()}
                if b.get("composite_from_max"):
                    ref_cols[b["composite_from_max"]] = bp.max(axis=1)
                rep = paired_report(name, y[:n], p, ref_cols, labels)
                if len(rep):
                    print(f"\n--- paired vs {name} on {subset} ---")
                    print(rep.to_string(index=False))
                    rep.to_csv(results / f"paired_{name}_{subset}.csv", index=False)
                    summary[subset][f"paired_{name}"] = rep.set_index("label")["delta"].to_dict()
    with open(results / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {results / 'summary.json'}")


if __name__ == "__main__":
    main()
