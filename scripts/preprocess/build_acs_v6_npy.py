#!/usr/bin/env python3
"""Stack per-ECG npy signals into per-split arrays + write fairseq manifests.

Builds two tasks from acs_v6_split_labels.csv:
  - acs    : binary Acute_Obstruction (ACCO|AICO), full cohort, num_labels=1
  - vessel : 4-label culprit territory [LAD,RCA,LCX,Left_Main], ACS+ subset only
Signals are (2500,12,1) -> squeezed to (2500,12); MHI scale 0.00488 (set in manifest).
"""
import ast, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

SP  = "/media/data1/datasets/DeepECG/acs_v6_final_labels.csv"  # post-exclusion final labels+split
OUT = "/volume/DeepECG-SSL-finetune/data/acs_final"
SCALE = 0.00488
VESSELS = ["LAD", "RCA", "LCX", "Left_Main"]
SPLITMAP = {"train": "train", "val": "valid", "test": "test"}  # fairseq uses 'valid'

os.makedirs(f"{OUT}/arrays", exist_ok=True)
for t in ("acs", "vessel"):
    os.makedirs(f"{OUT}/manifest/{t}", exist_ok=True)

df = pd.read_csv(SP, low_memory=False)
df = df[df.in_database_A == 1].reset_index(drop=True)  # Database A (post-exclusion cohort)

def _load_one(args):
    i, p, X = args
    try:
        a = np.asarray(np.load(p)).squeeze()
        if a.shape == (12, 2500):
            a = a.T
        X[i] = np.nan_to_num(a.astype(np.float32))
        return 0
    except Exception:
        return 1

def stack(paths):
    X = np.zeros((len(paths), 2500, 12), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=24) as ex:
        bad = sum(ex.map(_load_one, ((i, p, X) for i, p in enumerate(paths)), chunksize=64))
    return X, bad

def write_manifest(path, x_path, y_path, label_indexes):
    with open(path, "w") as f:
        f.write(f"x_path:{x_path}\n")
        f.write("x_shape:(2500, 12)\n")
        f.write(f"y_path:{y_path}\n")
        f.write(f"label_indexes:{label_indexes}\n")
        f.write(f"scale:{SCALE}\n")

for split, fsub in SPLITMAP.items():
    sub = df[df.Split == split].reset_index(drop=True)
    X, bad = stack(sub.npy_path.tolist())
    xp = f"{OUT}/arrays/X_{split}.npy"
    np.save(xp, X)
    # ACS binary head (full cohort)
    yacs = sub[["ACS"]].to_numpy(np.float32)
    yap = f"{OUT}/arrays/Y_{split}_acs.npy"
    np.save(yap, yacs)
    write_manifest(f"{OUT}/manifest/acs/{fsub}.tsv", xp, yap, "[0]")
    print(f"[acs]    {split:5s} X{X.shape} ACS+={int(yacs.sum())} bad={bad} -> {fsub}.tsv")

    # vessel head = Database B (ACS+ with a native culprit territory; graft-only already excluded)
    vmask = (sub.ACS == 1) & (sub.anyvessel == 1)
    Xv = X[vmask.to_numpy()]
    xvp = f"{OUT}/arrays/X_{split}_vessel.npy"
    np.save(xvp, Xv)
    yv = sub.loc[vmask, VESSELS].to_numpy(np.float32)
    yvp = f"{OUT}/arrays/Y_{split}_vessel.npy"
    np.save(yvp, yv)
    write_manifest(f"{OUT}/manifest/vessel/{fsub}.tsv", xvp, yvp, "[0,1,2,3]")
    print(f"[vessel] {split:5s} X{Xv.shape} pos={yv.sum(0).astype(int).tolist()} ({VESSELS})")

print("DONE. manifests under", f"{OUT}/manifest/")
