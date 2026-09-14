#!/usr/bin/env python3
"""Beat dataset via NeuroKit2 — the "standard methods" the Nature 2026 beat model cites.

s41586-026-10674-6 segments beats "using standard methods", citing Makowski et al., NeuroKit2
(Behav. Res. Methods 53, 1689-1696, 2021). Our previous detector was a hand-rolled
Pan-Tompkins-style envelope peak-picker; on the Porr & Macfarlane benchmark (PLOS ONE 2024,
doi:10.1371/journal.pone.0309739) that family scores 75% against ~99% for the NeuroKit /
EngZee detectors once timing jitter is penalised, and jitter is exactly what blurs an averaged
or autoencoded beat.

Detection runs on the CLEANED lead II; the window is then sliced from the RAW 12 leads so the
VAE sees true amplitudes. Window is fixed at -0.28 s .. +0.72 s (70/180 samples at 250 Hz), not
NeuroKit's RR-adaptive `ecg_segment` window, so every beat has the same length and the leads
stay time-aligned.

Quality control: records need >= 4 detected beats, and individual beats need lead-II correlation
>= 0.85 with the record's median beat. The Zhao 2018 verdict is stored as a `quality` column
rather than used to reject, because it discards acutely occluded ECGs far more often than
normals (39% of single-territory occlusion beats were lost when it was a filter).

-> data/acs_final/beat_cache_nk/{split}_beats.npy      (N,12,250) float16 mV
   data/acs_final/beat_cache_nk/{split}_median.npy     (n_rec,12,250) float16 mV
   data/acs_final/beat_cache_nk/{split}_beats.parquet / _records.parquet
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FS, PRE, POST = 250, 70, 180
BEAT_LEN = PRE + POST
SCALE = 0.00488
POWERLINE = 60                      # Canadian mains
MIN_BEATS = 4
MIN_CORR = 0.85
MAX_PER_REC = int(os.environ.get("MAX_PER_REC", 4))
WORKERS = int(os.environ.get("WORKERS", 24))

BASE = "/volume/DeepECG-SSL-finetune/data/acs_final"
SP = "/media/data1/datasets/DeepECG/acs_v6_final_labels.csv"
OUT = f"{BASE}/beat_cache_nk"
CLASSES = ["LEFT", "RCA", "LCX"]
os.makedirs(OUT, exist_ok=True)

_X = None
_nk = None


def _init(path):
    global _X, _nk
    import neurokit2
    _nk = neurokit2
    _X = np.load(path, mmap_mode="r")


def _one(i):
    """-> (i, beats (K,12,250) float16 mV, median (12,250) float16, reason)"""
    x = np.nan_to_num(np.asarray(_X[i], dtype=np.float32)) * SCALE      # (2500,12) mV
    lead2 = x[:, 1]
    if not np.isfinite(lead2).all() or lead2.std() < 1e-6:
        return i, None, None, "flat"
    try:
        clean = _nk.ecg_clean(lead2, sampling_rate=FS, method="neurokit", powerline=POWERLINE)
        _, info = _nk.ecg_peaks(clean, sampling_rate=FS, method="neurokit",
                                correct_artifacts=True)
        r = np.asarray(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return i, None, None, "detector_error"
    r = r[(r >= PRE) & (r + POST <= len(lead2))]
    if len(r) < MIN_BEATS:
        return i, None, None, "too_few_beats"
    # Quality is RECORDED, not used to reject: the Zhao 2018 verdict rejects acutely occluded
    # ECGs far more often than normals (ST-T distortion and ectopy read as poor quality), so
    # filtering on it would bias the positive class. Downstream can filter on this column.
    try:
        q = _nk.ecg_quality(clean, rpeaks=r, sampling_rate=FS, method="zhao2018",
                            approach="simple")
        q = str(q).split()[0].lower() if q is not None else "unknown"
    except Exception:
        q = "unknown"
    beats = np.stack([x[p - PRE:p + POST] for p in r])                  # (K,250,12)
    beats = beats - beats[:, :20].mean(1, keepdims=True)                # PR-segment baseline
    beats = np.transpose(beats, (0, 2, 1))                              # (K,12,250)
    med = np.median(beats, 0)
    c = np.array([np.corrcoef(b[1], med[1])[0, 1] for b in beats])
    keep = np.nan_to_num(c) >= MIN_CORR
    if keep.sum() < 1:
        return i, None, None, "no_consistent_beat"
    beats = beats[keep]
    med = np.median(beats, 0)
    return (i, beats[:MAX_PER_REC].astype(np.float16), med.astype(np.float16), q)


def main():
    df = pd.read_csv(SP, low_memory=False)
    df = df[df.in_database_A == 1].reset_index(drop=True)
    df["LEFT"] = ((df.LAD == 1) | (df.Left_Main == 1)).astype(int)

    for split in ("train", "val", "test"):
        sub = df[df.Split == split].reset_index(drop=True)
        onehot = sub[CLASSES].to_numpy(np.float32)
        n_terr = onehot.sum(1)
        single = (sub.ACCO == 1).to_numpy() & (n_terr == 1)
        terr = np.where(n_terr == 1, onehot.argmax(1), -1).astype(np.int64)

        xp = f"{BASE}/arrays/X_{split}.npy"
        n = len(sub)
        t0 = time.time()
        B, idx, meds, med_rec, reasons, qual, rqual = [], [], [], [], [], [], []
        with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(xp,)) as ex:
            for k, (i, b, med, why) in enumerate(ex.map(_one, range(n), chunksize=32)):
                reasons.append(why)
                if b is not None:
                    B.append(b)
                    idx.extend([i] * len(b))
                    qual.extend([why] * len(b))
                    meds.append(med)
                    med_rec.append(i)
                    rqual.append(why)
                if (k + 1) % 10000 == 0:
                    print(f"  {split} {k+1}/{n} -> {sum(len(z) for z in B)} beats "
                          f"({time.time()-t0:.0f}s)", flush=True)
        B = np.concatenate(B) if B else np.zeros((0, 12, BEAT_LEN), np.float16)
        idx = np.array(idx)
        meds = np.stack(meds) if meds else np.zeros((0, 12, BEAT_LEN), np.float16)
        med_rec = np.array(med_rec)

        meta = pd.DataFrame({
            "rec": idx, "ACS": sub.ACS.to_numpy()[idx], "ACCO": sub.ACCO.to_numpy()[idx],
            "territory": terr[idx], "single_terr_acco": single[idx].astype(int),
            "new_PatientID": sub.new_PatientID.to_numpy()[idx],
            "quality": qual})
        rmeta = pd.DataFrame({
            "rec": med_rec, "ACS": sub.ACS.to_numpy()[med_rec],
            "ACCO": sub.ACCO.to_numpy()[med_rec], "territory": terr[med_rec],
            "single_terr_acco": single[med_rec].astype(int),
            "new_PatientID": sub.new_PatientID.to_numpy()[med_rec],
            "quality": rqual})
        np.save(f"{OUT}/{split}_beats.npy", B)
        np.save(f"{OUT}/{split}_median.npy", meds)
        meta.to_parquet(f"{OUT}/{split}_beats.parquet")
        rmeta.to_parquet(f"{OUT}/{split}_records.parquet")

        rc = pd.Series(reasons).value_counts().to_dict()
        print(f"{split}: {len(B)} beats from {len(med_rec)}/{n} records "
              f"({len(med_rec)/n*100:.1f}%) | ACCO {meta.ACCO.mean():.3f} | "
              f"single-territory ACCO beats {int(meta.single_terr_acco.sum())} "
              + str({c: int(((meta.territory == i) & (meta.single_terr_acco == 1)).sum())
                     for i, c in enumerate(CLASSES)})
              + f" | rejects {rc} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
