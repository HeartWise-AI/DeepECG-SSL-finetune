#!/usr/bin/env python3
"""Carve a task-specific dataset out of the merged WCR v2 multi-task dataset (local arrays, no NAS reads).

Rows are selected either by label availability (keep rows where at least one of the chosen heads is
defined) or by an explicit list of ECG ids (e.g. the Notion-documented iAF5 cohort). Labels are
restricted to the chosen heads; x / aux / meta are copied row-wise; manifests and labels.json are
written so the result is a drop-in `task.data` for fairseq-signals.

Examples
  # EchoNext model: the 12 official EchoNext labels, rows with any echo label
  python3 scripts/preprocess/ecg/subset_wcrv2_dataset.py --src data/wcrv2_multitask \
      --out data/wcrv2_echonext12 --labels lvef_lte_45 lvwt_gte_13 as_mod_plus ar_mod_plus mr_mod_plus \
      tr_mod_plus pr_mod_plus rv_dysf_mod_plus peric_eff_mod_large pasp_gte_45 tr_max_gte_32 shd_composite

  # Incident-AF model restricted to the iAF5 cohort rows (Notion: DeepECG-PNG iAF5)
  python3 scripts/preprocess/ecg/subset_wcrv2_dataset.py --src data/wcrv2_multitask \
      --out data/wcrv2_iaf5 --labels incident_af_2y incident_af_5y \
      --ids-parquet /media/data1/datasets/DeepECG/DeepECG-PNG/downstream/incident_afib5y.parquet --ids-col npy_path
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

T, L = 2500, 12
SPLITS = ["train", "val", "test", "echonext_test", "mhi_test"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_manifest(out_data, out_man, s, n_labels):
    with open(out_man / f"{s}.tsv", "w") as f:
        f.write(f"x_path:{out_data / f'{s}_x.npy'}\n")
        f.write(f"x_shape:({T}, {L}, 1)\n")
        f.write(f"y_path:{out_data / f'{s}_y.npy'}\n")
        f.write(f"aux_path:{out_data / f'{s}_aux.npy'}\n")
        f.write(f"label_indexes:{list(range(n_labels))}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", nargs="+", required=True, help="head names to keep, in output order")
    ap.add_argument("--ids-parquet", default=None, help="restrict rows to ids found in this parquet")
    ap.add_argument("--ids-col", default="npy_path")
    ap.add_argument("--ids-split-col", default=None,
                    help="if set, take the split from this column of --ids-parquet instead of the source split")
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--require-all", action="store_true", help="keep only rows where every chosen head is defined")
    args = ap.parse_args()

    src, out = Path(args.src).resolve(), Path(args.out).resolve()
    out_data, out_man = out / "data", out / "manifests"
    out_data.mkdir(parents=True, exist_ok=True)
    out_man.mkdir(parents=True, exist_ok=True)
    info = json.load(open(src / "labels.json"))
    all_labels = info["labels"]
    idx = [all_labels.index(l) for l in args.labels]

    ids_set, ids_split = None, None
    if args.ids_parquet:
        cols = [args.ids_col] + ([args.ids_split_col] if args.ids_split_col else [])
        idp = pd.read_parquet(args.ids_parquet, columns=cols).drop_duplicates(args.ids_col)
        ids_set = set(idp[args.ids_col].astype(str))
        if args.ids_split_col:
            ids_split = dict(zip(idp[args.ids_col].astype(str), idp[args.ids_split_col].astype(str)))
        log(f"id list: {len(ids_set):,} ids from {args.ids_parquet}")

    # when the split comes from the id parquet, rows may move between splits: gather everything first
    pooled = []
    cohort = {}
    for s in args.splits:
        if not (src / "data" / f"{s}_y.npy").exists():
            continue
        y = np.load(src / "data" / f"{s}_y.npy")[:, idx]
        meta = pd.read_parquet(src / "data" / f"{s}_meta.parquet")
        keep = (~np.isnan(y)).all(axis=1) if args.require_all else (~np.isnan(y)).any(axis=1)
        if ids_set is not None:
            keep &= meta["ecg_id"].astype(str).isin(ids_set).to_numpy()
        rows = np.where(keep)[0]
        log(f"[{s}] {len(rows):,} / {len(meta):,} rows kept")
        if ids_split is not None and s in ("train", "val", "test"):
            tgt = meta["ecg_id"].astype(str).map(ids_split).to_numpy()
            for sp in ("train", "val", "test"):
                r = rows[tgt[rows] == sp]
                if len(r):
                    pooled.append((s, sp, r))
        else:
            pooled.append((s, s, rows))

    # write each output split by concatenating its (source split, rows) chunks
    for out_split in dict.fromkeys(p[1] for p in pooled):
        chunks = [(s, r) for s, sp, r in pooled if sp == out_split]
        N = sum(len(r) for _, r in chunks)
        X = np.lib.format.open_memmap(out_data / f"{out_split}_x.npy", mode="w+", dtype=np.float32, shape=(N, T, L, 1))
        ys, auxs, metas = [], [], []
        j = 0
        for s, r in chunks:
            sx = np.load(src / "data" / f"{s}_x.npy", mmap_mode="r")
            sy = np.load(src / "data" / f"{s}_y.npy")[:, idx]
            sa = np.load(src / "data" / f"{s}_aux.npy")
            sm = pd.read_parquet(src / "data" / f"{s}_meta.parquet")
            for i0 in range(0, len(r), 1024):
                rr = r[i0:i0 + 1024]
                X[j:j + len(rr)] = sx[rr]
                j += len(rr)
            ys.append(sy[r]); auxs.append(sa[r]); metas.append(sm.iloc[r].assign(src_split=s))
            del sx
        X.flush(); del X
        y = np.concatenate(ys).astype(np.float32)
        np.save(out_data / f"{out_split}_y.npy", y)
        np.save(out_data / f"{out_split}_aux.npy", np.concatenate(auxs).astype(np.float32))
        m = pd.concat(metas, ignore_index=True)
        m.to_parquet(out_data / f"{out_split}_meta.parquet", index=False)
        write_manifest(out_data, out_man, out_split, len(idx))
        cohort[out_split] = {"n": int(N), "n_patients": int(m["patient_id"].nunique()),
                             "by_source": m["source"].value_counts().to_dict(),
                             "label_n": {l: int((~np.isnan(y[:, k])).sum()) for k, l in enumerate(args.labels)},
                             "label_prev": {l: (float(np.nanmean(y[:, k])) if (~np.isnan(y[:, k])).any() else None)
                                            for k, l in enumerate(args.labels)}}
        log(f"[{out_split}] written N={N:,} patients={cohort[out_split]['n_patients']:,} "
            f"prev={ {l: round(v, 4) if v is not None else None for l, v in cohort[out_split]['label_prev'].items()} }")

    out_info = {"labels": args.labels, "aux": info["aux"], "aux_norm": info["aux_norm"],
                "source_dataset": str(src), "ids_parquet": args.ids_parquet, "ids_split_col": args.ids_split_col,
                "require_all": args.require_all, "cohort": cohort, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mhi_waveform_source": info.get("mhi_waveform_source")}
    json.dump(out_info, open(out / "labels.json", "w"), indent=2)
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
