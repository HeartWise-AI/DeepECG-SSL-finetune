#!/usr/bin/env python3
"""Build the WCR v2 multi-task fine-tuning dataset.

Merges three label programmes on one ECG table, with NaN for "label not
available" so that ``criterion=masked_bce`` can train all heads jointly:

  A. EchoNext structural heart disease, the full 12-label set used by the
     published EchoNext model (11 component labels + composite), on
     * EchoNext (Columbia / NYP, PhysioNet 100K) train/val/test, and
     * MHI v1.6 ECGs with an echo in the year *before* the ECG
       (``echonext_*`` columns of the v1.6 GT parquet; same linkage as v6).
  B. Low LVEF (<= 40 %, < 50 %; <= 45 % already in A) from the DeepECHO
     visually estimated EF (MHI) or ``lvef_value`` (EchoNext), same 1-year
     window.
  C. Incident atrial fibrillation at 2 and 5 years (MHI only): no prevalent AF
     at or before the ECG, positives have first AF within the horizon.
     Negatives: ``--af-negatives horizon`` (default) requires >= horizon of
     follow-up; ``--af-negatives any`` keeps every AF-free ECG, which is the
     EHJ 2024 paper definition used for the delivered incident-AF model.

Auxiliary inputs: age (z-scored on train, NaN if unknown) and sex
(1 = male, 0 = female, NaN if unknown).

Outputs (default root: /volume/DeepECG-SSL-finetune/data/wcrv2_multitask):
  data/{train,val,test,echonext_test,mhi_test}_x.npy   (N, 2500, 12, 1) float32 mV
  data/{split}_y.npy                                    (N, 16) float32, NaN = missing
  data/{split}_aux.npy                                  (N, 2)  float32, NaN = missing
  data/{split}_meta.parquet                             row-aligned ids (source, ecg id, patient, dates)
  manifests/{split}.tsv                                 fairseq-signals NpECGDataset manifests
  labels.json                                           label order, aux normalisation, cohort counts

Splits: MHI uses the v1.6 ``Split`` column (patient-level, verified 0 patients
in >1 split); EchoNext uses its own ``split`` column. ``--dry-run`` prints the
cohort table without touching waveforms.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

T, L = 2500, 12
DAY_NS = 8.64e13
NAT_SENTINEL = -9.2e18

ECHONEXT_DIR = Path("/media/data1/datasets/EchoNext")
MHI_GT = Path("/media/data1/datasets/MHI/ECG_ad20241231_gt_labels_v1.6.parquet")
MHI_META = Path("/media/data1/datasets/MHI/ECG_ad20241231_metadata.v1.6.ROXs42Bb.cleaned.parquet")
EN_SCALE = 0.162889  # EchoNext pre-normalised waveform -> mV (calibrated, see DeepECG-EchoNext v5/v6)

# Output label order. Keep in sync with diagnosis_wcrv2_multitask.yaml (criterion.label_names).
LABELS = [
    # A. EchoNext SHD block (official EchoNext label set)
    "lvef_lte_45",
    "lvwt_gte_13",
    "as_mod_plus",
    "ar_mod_plus",
    "mr_mod_plus",
    "tr_mod_plus",
    "pr_mod_plus",
    "rv_dysf_mod_plus",
    "peric_eff_mod_large",
    "pasp_gte_45",
    "tr_max_gte_32",
    "shd_composite",
    # B. Low LVEF block
    "lvef_lte_40",
    "lvef_lt_50",
    # C. Incident AF block
    "incident_af_2y",
    "incident_af_5y",
]
AUX = ["age_z", "sex_male"]

EN_FLAGS = {
    "lvef_lte_45": "lvef_lte_45_flag",
    "lvwt_gte_13": "lvwt_gte_13_flag",
    "as_mod_plus": "aortic_stenosis_moderate_or_greater_flag",
    "ar_mod_plus": "aortic_regurgitation_moderate_or_greater_flag",
    "mr_mod_plus": "mitral_regurgitation_moderate_or_greater_flag",
    "tr_mod_plus": "tricuspid_regurgitation_moderate_or_greater_flag",
    "pr_mod_plus": "pulmonary_regurgitation_moderate_or_greater_flag",
    "rv_dysf_mod_plus": "rv_systolic_dysfunction_moderate_or_greater_flag",
    "peric_eff_mod_large": "pericardial_effusion_moderate_large_flag",
    "pasp_gte_45": "pasp_gte_45_flag",
    "tr_max_gte_32": "tr_max_gte_32_flag",
    "shd_composite": "shd_moderate_or_greater_flag",
}
MHI_FLAGS = {
    "lvef_lte_45": "echonext_lvef_lte_45",
    "lvwt_gte_13": "echonext_lvwt_gte_13",
    "as_mod_plus": "echonext_aortic_stenosis_moderate_severe",
    "ar_mod_plus": "echonext_aortic_regurgitation_moderate_severe",
    "mr_mod_plus": "echonext_mitral_regurgitation_moderate_severe",
    "tr_mod_plus": "echonext_tricuspid_regurgitation_moderate_severe",
    "pr_mod_plus": "echonext_pulmonary_regurgitation_moderate_severe",
    "rv_dysf_mod_plus": "echonext_rv_systolic_dysfunction_moderate_severe",
    "peric_eff_mod_large": "echonext_pericardial_effusion_moderate_large",
    "pasp_gte_45": "echonext_pasp_gte_45",
    "tr_max_gte_32": "echonext_tr_max_gte_32",
    "shd_composite": "echonext_shd",
}
# MHI never populated this one (prevalence 1e-4 on 454K echo-linked rows): treat as missing.
MHI_UNRELIABLE = {"tr_max_gte_32"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ns_to_days(s):
    v = pd.to_numeric(s, errors="coerce").astype("float64")
    v[v <= NAT_SENTINEL] = np.nan
    return v / DAY_NS


# --------------------------------------------------------------------------- EchoNext
def echonext_table():
    meta = pd.read_csv(ECHONEXT_DIR / "echonext_metadata_100k.csv")
    meta = meta[meta["split"].isin(["train", "val", "test"])].copy()
    # row position inside the per-split waveform file == order of rows with that split in metadata
    meta["row_in_split"] = meta.groupby("split").cumcount()

    y = pd.DataFrame(index=meta.index, columns=LABELS, dtype="float32")
    for lab, col in EN_FLAGS.items():
        y[lab] = pd.to_numeric(meta[col], errors="coerce").astype("float32")
    ef = pd.to_numeric(meta["lvef_value"], errors="coerce")
    y["lvef_lte_40"] = np.where(ef.notna(), (ef <= 40).astype("float32"), np.nan)
    y["lvef_lt_50"] = np.where(ef.notna(), (ef < 50).astype("float32"), np.nan)
    # EchoNext has no AF follow-up -> missing
    y["incident_af_2y"] = np.nan
    y["incident_af_5y"] = np.nan

    aux = pd.DataFrame(index=meta.index)
    aux["age"] = pd.to_numeric(meta["age_at_ecg"], errors="coerce")
    aux["sex_male"] = meta["sex"].map({"male": 1.0, "female": 0.0, 1: 1.0, 0: 0.0}).astype("float32")

    ids = pd.DataFrame({
        "source": "echonext",
        "split": meta["split"].values,
        "ecg_id": meta["ecg_key"].astype(str).values,
        "patient_id": meta["patient_key"].astype(str).values,
        "row_in_split": meta["row_in_split"].values,
        "ecg_date": pd.NaT,
        "echo_date": pd.NaT,
    }, index=meta.index)
    return ids, y, aux


# --------------------------------------------------------------------------- MHI age
# The v1.6 metadata carries PatientAge / DateofBirth on only ~18 % of ECGs. The DeepECG-PNG downstream
# cohorts carry them on ~100 % of their rows (same patients), so a date of birth can be pooled per
# patient from every file and propagated to all of that patient's ECGs. Last resort: afib_Age.
MHI_DOB_SOURCES = [
    MHI_META,
    Path("/media/data1/datasets/DeepECG/DeepECG-PNG/downstream/shd_lvef_cohort.parquet"),
    Path("/media/data1/datasets/DeepECG/DeepECG-PNG/downstream/lvef_cohort_1ecg_per_echo.parquet"),
    Path("/media/data1/datasets/DeepECG/DeepECG-PNG/downstream/incident_afib5y.parquet"),
]


def patient_dob_table():
    """PatientID -> date of birth (median over sources), from DOB columns and from age-at-acquisition."""
    parts = []
    for p in MHI_DOB_SOURCES:
        if not p.exists():
            log(f"[age] missing source {p}")
            continue
        d = pd.read_parquet(p, columns=["PatientID", "RestingECG_PatientDemographics_DateofBirth",
                                        "RestingECG_PatientDemographics_PatientAge",
                                        "RestingECG_TestDemographics_AcquisitionDate"])
        dob = pd.to_datetime(d["RestingECG_PatientDemographics_DateofBirth"], errors="coerce")
        parts.append(pd.DataFrame({"PatientID": d["PatientID"], "dob": dob}))
        acq = pd.to_datetime(d["RestingECG_TestDemographics_AcquisitionDate"], errors="coerce")
        age = pd.to_numeric(d["RestingECG_PatientDemographics_PatientAge"], errors="coerce")
        parts.append(pd.DataFrame({"PatientID": d["PatientID"],
                                   "dob": acq - pd.to_timedelta(age * 365.25, unit="D")}))
    allp = pd.concat(parts).dropna()
    tab = allp.groupby("PatientID")["dob"].median()
    log(f"[age] date of birth known for {len(tab):,} MHI patients")
    return tab


def mhi_age(patient_id, ecg_dt, patient_age_col, afib_age):
    age = pd.to_numeric(patient_age_col, errors="coerce")
    dob = patient_id.map(patient_dob_table())
    age_dob = (ecg_dt - dob).dt.days / 365.25
    age = age.fillna(age_dob).fillna(afib_age)
    age = age.where((age >= 0) & (age <= 110))
    log(f"[age] MHI age known on {age.notna().mean()*100:.1f} % of rows "
        f"(direct {pd.to_numeric(patient_age_col, errors='coerce').notna().mean()*100:.1f} %)")
    return age


# --------------------------------------------------------------------------- MHI v1.6
def mhi_table(gap_days, horizons, af_negatives="horizon"):
    cols = ["npy_path", "Split", "waveform_path_psa", "echonext_TTE_StudyDate",
            "deepecho_Visually_Estimated_EF", "deepecho_TTE_StudyDate",
            "afib_label_2y", "afib_label_5y", "afib_dt_minaf", "afib_dt_maxfu", "afib_Age"]
    cols += list(MHI_FLAGS.values())
    gt = pd.read_parquet(MHI_GT, columns=cols)
    gt = gt[gt["Split"].isin(["train", "val", "test"]) & gt["waveform_path_psa"].notna()].copy()
    gt = gt.drop_duplicates(subset=["npy_path"])
    meta = pd.read_parquet(MHI_META, columns=[
        "npy_path", "PatientID", "RestingECG_TestDemographics_AcquisitionDate",
        "RestingECG_PatientDemographics_PatientAge", "RestingECG_PatientDemographics_DateofBirth",
        "RestingECG_PatientDemographics_Gender",
    ]).drop_duplicates(subset=["npy_path"])
    g = gt.merge(meta, on="npy_path", how="left")
    log(f"MHI v1.6 rows with split + waveform: {len(g):,}")

    ecg_dt = pd.to_datetime(g["RestingECG_TestDemographics_AcquisitionDate"], errors="coerce")

    y = pd.DataFrame(index=g.index, columns=LABELS, dtype="float32")
    y[:] = np.nan

    # A. EchoNext labels, ECG within `gap_days` before the echo
    echo_dt = pd.to_datetime(g["echonext_TTE_StudyDate"], errors="coerce")
    d_echo = (ecg_dt - echo_dt).dt.days
    en_ok = echo_dt.notna() & (d_echo >= -gap_days) & (d_echo <= 0)
    for lab, col in MHI_FLAGS.items():
        if lab in MHI_UNRELIABLE:
            continue
        v = pd.to_numeric(g[col], errors="coerce").astype("float32")
        y.loc[en_ok & v.notna(), lab] = v[en_ok & v.notna()]

    # B. LVEF thresholds from DeepECHO EF, same window
    ef = pd.to_numeric(g["deepecho_Visually_Estimated_EF"], errors="coerce")
    ef_dt = pd.to_datetime(g["deepecho_TTE_StudyDate"], errors="coerce")
    d_ef = (ecg_dt - ef_dt).dt.days
    ef_ok = ef.notna() & ef_dt.notna() & (d_ef >= -gap_days) & (d_ef <= 0)
    y.loc[ef_ok, "lvef_lte_40"] = (ef[ef_ok] <= 40).astype("float32")
    y.loc[ef_ok, "lvef_lt_50"] = (ef[ef_ok] < 50).astype("float32")
    fill45 = ef_ok & y["lvef_lte_45"].isna()
    y.loc[fill45, "lvef_lte_45"] = (ef[fill45] <= 45).astype("float32")

    # C. Incident AF
    minaf = _ns_to_days(g["afib_dt_minaf"])
    maxfu = _ns_to_days(g["afib_dt_maxfu"])
    no_prevalent = minaf.isna() | (minaf > 0)
    for lab, col, h in [("incident_af_2y", "afib_label_2y", horizons[0]), ("incident_af_5y", "afib_label_5y", horizons[1])]:
        raw = pd.to_numeric(g[col], errors="coerce")
        pos = raw.notna() & no_prevalent & (raw == 1) & (minaf <= h)
        if af_negatives == "any":
            # EHJ-paper definition (Jabbour et al. 2024): every sinus-rhythm ECG without prevalent AF is
            # kept; a negative is "no AF recorded during follow-up", however short the follow-up.
            neg = raw.notna() & no_prevalent & (raw == 0)
        else:
            neg = raw.notna() & no_prevalent & (raw == 0) & (maxfu >= h)
        y.loc[pos, lab] = 1.0
        y.loc[neg, lab] = 0.0

    keep = y.notna().any(axis=1)
    log(f"MHI rows with >=1 label: {keep.sum():,}  (EchoNext-linked {en_ok.sum():,}, EF-linked {ef_ok.sum():,}, "
        f"incident-AF eligible {y['incident_af_5y'].notna().sum():,})")

    age = mhi_age(g["PatientID"], ecg_dt, g["RestingECG_PatientDemographics_PatientAge"],
                  pd.to_numeric(g["afib_Age"], errors="coerce"))
    sex = g["RestingECG_PatientDemographics_Gender"].map({"MALE": 1.0, "FEMALE": 0.0}).astype("float32")
    aux = pd.DataFrame({"age": age.astype("float32"), "sex_male": sex}, index=g.index)

    ids = pd.DataFrame({
        "source": "mhi",
        "split": g["Split"].values,
        "ecg_id": g["npy_path"].astype(str).values,
        "patient_id": g["PatientID"].astype(str).values,
        "waveform_path": g["waveform_path_psa"].astype(str).values,
        "ecg_date": ecg_dt.values,
        "echo_date": echo_dt.where(en_ok, ef_dt.where(ef_ok, pd.NaT)).values,
    }, index=g.index)
    return ids[keep], y[keep], aux[keep]


# --------------------------------------------------------------------------- MHI consolidated arrays
# The 77-label WCR fine-tunes used one big float16 raw-ADC array per split. Their row order is the
# 2022-07 export parquet order filtered by the v1.6 Split (verified for val and test: corr 1.0 against
# the per-ECG files; the train array is shuffled and its order is unknown). Reading 60 KB rows
# sequentially from one file is ~50x faster over NFS than 465 K small files.
MHI_ARRAYS = {
    "val": Path("/media/data1/anolin/X_val_v1.1.npy"),
    "test": Path("/media/data1/muse_ge/X_test_v1.2.npy"),
}
MHI_ORDER_PARQUET = Path("/media/data1/muse_ge/ECG_ad202207_1453937_cat_labels_v1.1_with_additional_columns.parquet")
MHI_ADC_SCALE = 0.00488  # MUSE GE ADC -> mV (same factor as the deployed WCR v2 pipeline)


def mhi_array_rows(split):
    """basename(npy_path) -> row index in MHI_ARRAYS[split]."""
    old = pd.read_parquet(MHI_ORDER_PARQUET, columns=["npy_path"])
    old["base"] = old["npy_path"].astype(str).map(os.path.basename)
    gt = pd.read_parquet(MHI_GT, columns=["npy_path", "Split"])
    split_of = pd.Series(gt["Split"].values, index=gt["npy_path"].astype(str).map(os.path.basename).values)
    split_of = split_of[~split_of.index.duplicated()]
    old["Split"] = old["base"].map(split_of)
    sub = old[old["Split"] == split].reset_index(drop=True)
    n_arr = np.load(MHI_ARRAYS[split], mmap_mode="r").shape[0]
    assert len(sub) == n_arr, f"{split}: order table has {len(sub)} rows, array has {n_arr}"
    return pd.Series(np.arange(len(sub)), index=sub["base"].values)


# --------------------------------------------------------------------------- local reuse cache
REUSE_CACHE = None


def local_cache_index(reuse_dir):
    """ecg_id -> (local x array path, row) for every row of an already-built dataset (train/val/test only)."""
    idx = {}
    for s in ["train", "val", "test"]:
        mp = reuse_dir / "data" / f"{s}_meta.parquet"
        xp = reuse_dir / "data" / f"{s}_x.npy"
        if not (mp.exists() and xp.exists()):
            continue
        m = pd.read_parquet(mp, columns=["ecg_id", "source"])
        for r, (e, src) in enumerate(zip(m["ecg_id"].astype(str), m["source"])):
            if src == "mhi":
                idx[e] = (str(xp), r)
    log(f"[reuse] {len(idx):,} MHI ECGs available locally in {reuse_dir}")
    return idx


# --------------------------------------------------------------------------- waveform writers
def _load_mhi(i_p):
    i, p = i_p
    try:
        a = np.load(p)
        if a.ndim == 3:
            a = a[:, :, 0]
        if a.shape != (T, L):
            return i, None
        return i, a.astype(np.float32)
    except Exception:
        return i, None


def write_split(name, ids, y, aux, out_data, out_man, workers, array_splits=()):
    """Write x/y/aux/meta + manifest for one split. Returns row-aligned ids (bad rows dropped)."""
    N = len(ids)
    x_path = out_data / f"{name}_x.npy"
    log(f"[{name}] writing {N:,} ECGs -> {x_path}")
    X = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(N, T, L, 1))
    bad = np.zeros(N, dtype=bool)
    from_array = np.zeros(N, dtype=bool)

    # MHI rows served from the consolidated per-split arrays (sequential reads)
    for sp in array_splits:
        sel = np.where((ids["source"].values == "mhi") & (ids["split"].values == sp))[0]
        if len(sel) == 0:
            continue
        rows_of = mhi_array_rows(sp)
        base = ids["ecg_id"].values[sel]
        base = np.array([os.path.basename(str(b)) for b in base])
        r = rows_of.reindex(base).to_numpy()
        ok = ~np.isnan(r)
        sel, r = sel[ok], r[ok].astype(int)
        order = np.argsort(r)
        sel, r = sel[order], r[order]
        arr = np.load(MHI_ARRAYS[sp], mmap_mode="r")
        log(f"[{name}] MHI {sp}: {len(sel):,} rows from {MHI_ARRAYS[sp].name} ({(~ok).sum()} not in array -> per-file)")
        CH = 2048
        t0 = time.time()
        for i0 in range(0, len(sel), CH):
            rr = r[i0:i0 + CH]
            a = np.asarray(arr[rr], dtype=np.float32) * MHI_ADC_SCALE  # (B, 2500, 12)
            X[sel[i0:i0 + CH], :, :, 0] = a
            if (i0 // CH) % 10 == 0:
                log(f"[{name}] MHI {sp} array {i0 + len(rr):,}/{len(sel):,} ({time.time()-t0:.0f}s)")
        from_array[sel] = True
        del arr

    # EchoNext rows: chunked reads from the split waveform file
    en = ids["source"].values == "echonext"
    if en.any():
        for sp in ids.loc[en, "split"].unique():
            sel = np.where(en & (ids["split"].values == sp))[0]
            wav = np.load(ECHONEXT_DIR / f"EchoNext_{sp}_waveforms.npy", mmap_mode="r")
            rows = ids["row_in_split"].values[sel].astype(int)
            order = np.argsort(rows)
            sel, rows = sel[order], rows[order]
            CH = 512
            for i0 in range(0, len(sel), CH):
                r = rows[i0:i0 + CH]
                a = np.asarray(wav[r]).astype(np.float32)[:, 0] * EN_SCALE  # (B, 2500, 12)
                X[sel[i0:i0 + CH], :, :, 0] = a
            del wav
        log(f"[{name}] EchoNext rows done ({en.sum():,})")

    # MHI rows already present in a local dataset: copy from its arrays (sequential per source array)
    if REUSE_CACHE:
        cand = np.where(~en & ~from_array)[0]
        hits = [(i, *REUSE_CACHE[e]) for i, e in zip(cand, ids["ecg_id"].values[cand]) if str(e) in REUSE_CACHE]
        if hits:
            log(f"[{name}] {len(hits):,} MHI rows copied from local cache ({len(cand) - len(hits):,} still from NAS)")
            by_src = {}
            for i, xp, r in hits:
                by_src.setdefault(xp, []).append((r, i))
            for xp, pairs in by_src.items():
                pairs.sort()
                src = np.load(xp, mmap_mode="r")
                rs = np.array([p[0] for p in pairs]); ii = np.array([p[1] for p in pairs])
                for i0 in range(0, len(rs), 1024):
                    X[ii[i0:i0 + 1024]] = src[rs[i0:i0 + 1024]]
                del src
            from_array[[h[0] for h in hits]] = True

    # MHI rows: threaded per-file reads from the NAS
    mh = np.where(~en & ~from_array)[0]
    if len(mh):
        paths = list(zip(mh, ids["waveform_path"].values[mh]))
        done = 0
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(_load_mhi, ip) for ip in paths]):
                i, a = fut.result()
                if a is None:
                    bad[i] = True
                else:
                    X[i, :, :, 0] = a
                done += 1
                if done % 20000 == 0:
                    log(f"[{name}] MHI {done:,}/{len(mh):,} ({bad.sum()} bad, {time.time()-t0:.0f}s)")
    X.flush()
    del X

    if bad.any():
        log(f"[{name}] dropping {bad.sum()} unreadable ECGs")
        good = ~bad
        src = np.load(x_path, mmap_mode="r")
        tmp = out_data / f"{name}_x.tmp.npy"
        dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(int(good.sum()), T, L, 1))
        j = 0
        for i0 in range(0, N, 1024):
            blk = np.where(good[i0:i0 + 1024])[0] + i0
            if len(blk):
                dst[j:j + len(blk)] = src[blk]
                j += len(blk)
        dst.flush()
        del dst, src
        os.replace(tmp, x_path)
        ids, y, aux = ids[good], y[good], aux[good]
        N = len(ids)

    np.save(out_data / f"{name}_y.npy", y.to_numpy(dtype=np.float32))
    np.save(out_data / f"{name}_aux.npy", aux.to_numpy(dtype=np.float32))
    ids.reset_index(drop=True).to_parquet(out_data / f"{name}_meta.parquet", index=False)
    with open(out_man / f"{name}.tsv", "w") as f:
        f.write(f"x_path:{x_path}\n")
        f.write(f"x_shape:({T}, {L}, 1)\n")
        f.write(f"y_path:{out_data / f'{name}_y.npy'}\n")
        f.write(f"aux_path:{out_data / f'{name}_aux.npy'}\n")
        f.write(f"label_indexes:{list(range(len(LABELS)))}\n")
    log(f"[{name}] done: N={N:,}")
    return ids, y, aux


def summarize(ids, y, aux):
    rows = []
    for src in ["echonext", "mhi", "all"]:
        m = np.ones(len(ids), bool) if src == "all" else (ids["source"].values == src)
        for sp in ["train", "val", "test"]:
            mm = m & (ids["split"].values == sp)
            if mm.sum() == 0:
                continue
            r = {"source": src, "split": sp, "n_ecg": int(mm.sum()), "n_patients": int(ids.loc[mm, "patient_id"].nunique()),
                 "age_known": float(aux.loc[mm, "age"].notna().mean()), "sex_known": float(aux.loc[mm, "sex_male"].notna().mean())}
            for lab in LABELS:
                v = y.loc[mm, lab]
                r[f"{lab}.n"] = int(v.notna().sum())
                r[f"{lab}.prev"] = float(v.mean()) if v.notna().any() else None
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/volume/DeepECG-SSL-finetune/data/wcrv2_multitask")
    ap.add_argument("--gap-days", type=int, default=365, help="max days between ECG and the echo that labels it")
    ap.add_argument("--af-horizons", type=int, nargs=2, default=[730, 1826], metavar=("D2Y", "D5Y"))
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true", help="only print the cohort table and write labels.json")
    ap.add_argument("--refresh-aux", action="store_true",
                    help="recompute {split}_aux.npy (age, sex) from the existing {split}_meta.parquet files "
                         "without touching waveforms or labels; updates labels.json")
    ap.add_argument("--no-echonext", action="store_true")
    ap.add_argument("--no-mhi", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--mhi-array-splits", nargs="*", default=[], choices=list(MHI_ARRAYS),
                    help="MHI splits to read from the consolidated raw-ADC arrays (x0.00488) instead of per-ECG files")
    ap.add_argument("--af-negatives", choices=["horizon", "any"], default="horizon",
                    help="'horizon': negatives need >= horizon follow-up (strict); 'any': EHJ-paper definition")
    ap.add_argument("--only-labels", nargs="*", default=None,
                    help="restrict the label set (and keep only rows where at least one of these is defined)")
    ap.add_argument("--reuse-from", default=None,
                    help="existing dataset dir: rows already present there are copied from its local arrays "
                         "instead of being re-read from the NAS")
    args = ap.parse_args()
    global LABELS
    if args.only_labels:
        assert all(l in LABELS for l in args.only_labels), args.only_labels
    global REUSE_CACHE
    REUSE_CACHE = local_cache_index(Path(args.reuse_from)) if args.reuse_from else None

    out = Path(args.out)
    out_data, out_man = out / "data", out / "manifests"
    out_data.mkdir(parents=True, exist_ok=True)
    out_man.mkdir(parents=True, exist_ok=True)

    if args.refresh_aux:
        return refresh_aux(out, out_data)

    parts = []
    if not args.no_echonext:
        parts.append(echonext_table())
    if not args.no_mhi:
        parts.append(mhi_table(args.gap_days, args.af_horizons, args.af_negatives))
    ids = pd.concat([p[0] for p in parts], ignore_index=True)
    y = pd.concat([p[1] for p in parts], ignore_index=True)
    aux = pd.concat([p[2] for p in parts], ignore_index=True)
    for c in ["row_in_split", "waveform_path"]:
        if c not in ids.columns:
            ids[c] = np.nan
    if args.only_labels:
        LABELS = list(args.only_labels)
        keep = y[LABELS].notna().any(axis=1).to_numpy()
        ids, y, aux = ids[keep].reset_index(drop=True), y.loc[keep, LABELS].reset_index(drop=True), aux[keep].reset_index(drop=True)
        log(f"only-labels {LABELS}: {len(ids):,} rows kept")

    # cross-source patient-level split sanity (MHI only; EchoNext ids are opaque)
    mh = ids[ids["source"] == "mhi"]
    leak = (mh.groupby("patient_id")["split"].nunique() > 1).sum()
    log(f"MHI patients present in more than one split: {leak}")

    # age normalisation on train rows
    tr = ids["split"].values == "train"
    age_mean = float(aux.loc[tr, "age"].mean())
    age_std = float(aux.loc[tr, "age"].std())
    aux_out = pd.DataFrame({"age_z": (aux["age"] - age_mean) / age_std, "sex_male": aux["sex_male"]}, index=aux.index)

    summary = summarize(ids, y, aux)
    info = {
        "labels": LABELS,
        "aux": AUX,
        "aux_norm": {"age_mean": age_mean, "age_std": age_std},
        "echonext_scale": EN_SCALE,
        "mhi_scale": "waveform_path_psa already in mV",
        "gap_days": args.gap_days,
        "af_horizons_days": args.af_horizons,
        "mhi_unreliable_labels_masked": sorted(MHI_UNRELIABLE),
        "cohort": summary,
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": args.dry_run,
    }
    with open(out / "labels.json", "w") as f:
        json.dump(info, f, indent=2)

    df = pd.DataFrame(summary)
    pd.set_option("display.width", 250)
    print(df[["source", "split", "n_ecg", "n_patients", "age_known", "sex_known"]].to_string(index=False))
    prev = df[df["source"] == "all"].set_index("split")[[f"{l}.prev" for l in LABELS]].T
    prev.index = [i.replace(".prev", "") for i in prev.index]
    cnt = df[df["source"] == "all"].set_index("split")[[f"{l}.n" for l in LABELS]].T
    cnt.index = [i.replace(".n", "") for i in cnt.index]
    print("\nlabel availability (n) per split:\n", cnt.to_string())
    print("\nprevalence per split:\n", prev.round(4).to_string())
    if args.dry_run:
        log(f"dry run: wrote {out / 'labels.json'} only")
        return

    for sp in args.splits:
        m = ids["split"].values == sp
        res = write_split(sp, ids[m].reset_index(drop=True), y[m].reset_index(drop=True), aux_out[m].reset_index(drop=True),
                          out_data, out_man, args.workers, array_splits=tuple(args.mhi_array_splits))
        if sp == "test":
            # per-source test sets, sliced from the local combined test array (no second pass over the NAS)
            t_ids, t_y, t_aux = res
            subset_split("test", t_ids, t_y, t_aux, out_data, out_man)
    log("all done")


def refresh_aux(out, out_data):
    """Recompute age/sex for already-built splits from their row-aligned meta parquets."""
    info = json.load(open(out / "labels.json"))
    en_meta = pd.read_csv(ECHONEXT_DIR / "echonext_metadata_100k.csv", usecols=["ecg_key", "age_at_ecg", "sex"])
    en_meta["ecg_key"] = en_meta["ecg_key"].astype(str)
    en_age = en_meta.set_index("ecg_key")["age_at_ecg"].astype(float)
    en_sex = en_meta.set_index("ecg_key")["sex"].map({"male": 1.0, "female": 0.0})
    mh_meta = pd.read_parquet(MHI_META, columns=["npy_path", "PatientID", "RestingECG_PatientDemographics_PatientAge",
                                                 "RestingECG_PatientDemographics_Gender",
                                                 "RestingECG_TestDemographics_AcquisitionDate"]).drop_duplicates("npy_path")
    mh_af = pd.read_parquet(MHI_GT, columns=["npy_path", "afib_Age"]).drop_duplicates("npy_path")
    mh = mh_meta.merge(mh_af, on="npy_path", how="left").set_index("npy_path")
    dob_tab = patient_dob_table()

    splits = [s for s in ["train", "val", "test", "echonext_test", "mhi_test"] if (out_data / f"{s}_meta.parquet").exists()]
    raw = {}
    for s in splits:
        ids = pd.read_parquet(out_data / f"{s}_meta.parquet")
        age = pd.Series(np.nan, index=ids.index, dtype="float64")
        sex = pd.Series(np.nan, index=ids.index, dtype="float64")
        en = ids["source"].values == "echonext"
        age[en] = ids.loc[en, "ecg_id"].map(en_age).values
        sex[en] = ids.loc[en, "ecg_id"].map(en_sex).values
        m = ~en
        sub = mh.reindex(ids.loc[m, "ecg_id"].values)
        acq = pd.to_datetime(sub["RestingECG_TestDemographics_AcquisitionDate"], errors="coerce")
        a = pd.to_numeric(sub["RestingECG_PatientDemographics_PatientAge"], errors="coerce")
        a_dob = (acq - sub["PatientID"].map(dob_tab)).dt.days / 365.25
        a = a.fillna(a_dob).fillna(pd.to_numeric(sub["afib_Age"], errors="coerce"))
        a = a.where((a >= 0) & (a <= 110))
        age[m] = a.values
        sex[m] = sub["RestingECG_PatientDemographics_Gender"].map({"MALE": 1.0, "FEMALE": 0.0}).values
        raw[s] = (ids, age, sex)
        log(f"[refresh-aux] {s}: age known {age.notna().mean()*100:.1f} %, sex known {sex.notna().mean()*100:.1f} %")

    tr_age = raw["train"][1] if "train" in raw else pd.concat([r[1] for r in raw.values()])
    age_mean, age_std = float(tr_age.mean()), float(tr_age.std())
    for s, (ids, age, sex) in raw.items():
        aux = np.stack([((age - age_mean) / age_std).to_numpy(dtype=np.float32), sex.to_numpy(dtype=np.float32)], axis=1)
        np.save(out_data / f"{s}_aux.npy", aux)
    info["aux_norm"] = {"age_mean": age_mean, "age_std": age_std}
    info["aux_coverage"] = {s: {"age_known": float(a.notna().mean()), "sex_known": float(x.notna().mean())}
                            for s, (_, a, x) in raw.items()}
    info["aux_refreshed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(out / "labels.json", "w") as f:
        json.dump(info, f, indent=2)
    log("aux refreshed")


def subset_split(name, ids, y, aux, out_data, out_man):
    src_x = np.load(out_data / f"{name}_x.npy", mmap_mode="r")
    for src in ["echonext", "mhi"]:
        rows = np.where(ids["source"].values == src)[0]
        if len(rows) == 0:
            continue
        tag = f"{src}_{name}"
        x_path = out_data / f"{tag}_x.npy"
        log(f"[{tag}] slicing {len(rows):,} rows from {name}_x.npy")
        dst = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32, shape=(len(rows), T, L, 1))
        for i0 in range(0, len(rows), 1024):
            r = rows[i0:i0 + 1024]
            dst[i0:i0 + len(r)] = src_x[r]
        dst.flush()
        del dst
        np.save(out_data / f"{tag}_y.npy", y.iloc[rows].to_numpy(dtype=np.float32))
        np.save(out_data / f"{tag}_aux.npy", aux.iloc[rows].to_numpy(dtype=np.float32))
        ids.iloc[rows].reset_index(drop=True).to_parquet(out_data / f"{tag}_meta.parquet", index=False)
        with open(out_man / f"{tag}.tsv", "w") as f:
            f.write(f"x_path:{x_path}\n")
            f.write(f"x_shape:({T}, {L}, 1)\n")
            f.write(f"y_path:{out_data / f'{tag}_y.npy'}\n")
            f.write(f"aux_path:{out_data / f'{tag}_aux.npy'}\n")
            f.write(f"label_indexes:{list(range(len(LABELS)))}\n")
        log(f"[{tag}] done: N={len(rows):,}")


if __name__ == "__main__":
    sys.exit(main())
