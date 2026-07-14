#!/usr/bin/env python3
"""Build the DeepECG-ACS v6 cohort: derive labels + a patient-consistent split.

Split policy (decided 2026-06-25):
  - Inherit the canonical MHI `Split` (from ECG_ad20241231_gt_labels_v1.6.parquet,
    joined on npy_path) for every ECG whose patient appears in any canonical fold.
    Verified: 0 patients span >1 canonical fold, so this is unambiguous.
  - The genuinely-new patients (only in the unmatched/NaN group) get a fresh
    patient-level stratified split (70/10/20, seeds 42 outer / 123 inner),
    stratified on ACS / ACCO / sex / coarse age-bin (age is 61% missing -> 'unk').

Output: a clean per-ECG table (npy_path, new_PatientID, Split, derived labels)
that can be added as a Split column to the master file and saved to NAS.
"""
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

V6 = "/media/data1/ravram/DeepECG_Datasets/DEEPECG_ACS_FINAL_2017_2024_cath_ecg_merged_v6.csv"
LAB = "/media/data1/muse_ge/ECG_ad20241231_gt_labels_v1.6.parquet"
OUT = "/tmp/claude-0/-volume-DeepECG-SSL-finetune/370fb1ed-6b36-4663-96aa-09465bce7a76/scratchpad/acs_v6_split_labels.csv"

ACCO_S = "Acute Complete Coronary Occlusion"
AICO_S = "Acute Incomplete Coronary Occlusion"
SEED_OUTER, SEED_INNER = 42, 123

# ---- load ----
usecols = ["npy_path", "new_PatientID", "condition_severity", "pci_regions",
           "RestingECG_PatientDemographics_Gender",
           "RestingECG_PatientDemographics_PatientAge"]
df = pd.read_csv(V6, usecols=usecols, low_memory=False)
df = df.rename(columns={"RestingECG_PatientDemographics_Gender": "sex",
                        "RestingECG_PatientDemographics_PatientAge": "age"})
lab = pd.read_parquet(LAB, columns=["npy_path", "Split"])
df = df.merge(lab, on="npy_path", how="left")

# ---- labels ----
df["ACS"]  = df.condition_severity.isin([ACCO_S, AICO_S]).astype(int)
df["ACCO"] = (df.condition_severity == ACCO_S).astype(int)
df["AICO"] = (df.condition_severity == AICO_S).astype(int)

def pci_to_territories(x):
    reg = ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else []
    t = set()
    for r in reg:
        if   r.startswith(("IVA", "Diagonale")):               t.add("LAD")
        elif r.startswith(("CD", "IVP")):                      t.add("RCA")
        elif r.startswith(("Cx", "Marginale", "Bissectrice")): t.add("LCX")
        elif r.startswith("Tronc commun"):                     t.add("Left_Main")
    return t
terr = df.pci_regions.apply(pci_to_territories)
for v in ["LAD", "RCA", "LCX", "Left_Main"]:
    df[v] = terr.apply(lambda s, v=v: int(v in s))

# ---- recovered, patient-consistent split ----
df["Split"] = df["Split"].where(df["Split"].notna(), np.nan)
canon = df[df.Split.notna()]
pt_fold = canon.groupby("new_PatientID").Split.agg(lambda s: s.iloc[0])  # 0 conflicts verified
assert canon.groupby("new_PatientID").Split.nunique().max() == 1, "patient spans >1 canonical fold!"

# patients with a canonical fold -> inherit it for ALL their ECGs
df["Split_recovered"] = df.new_PatientID.map(pt_fold)

# genuinely-new patients -> fresh stratified patient-level split
new_mask = df.Split_recovered.isna()
new_pts = df[new_mask].copy()

# patient-level aggregation for stratification
def agebin(a):
    if pd.isna(a):      return "unk"
    if a < 35:          return "<35"
    if a <= 60:         return "35-60"
    return ">60"
g = new_pts.groupby("new_PatientID").agg(
    ACS=("ACS", "max"), ACCO=("ACCO", "max"),
    sex=("sex", lambda s: s.dropna().iloc[0] if s.notna().any() else "UNK"),
    age=("age", "median")).reset_index()
g["agebin"] = g.age.apply(agebin)
g["stratum"] = g.ACS.astype(str) + g.ACCO.astype(str) + g.sex.astype(str) + g.agebin

# collapse singleton strata so train_test_split(stratify=...) is valid
vc = g.stratum.value_counts()
g["stratum"] = g.stratum.where(g.stratum.map(vc) >= 3, "rare")

# outer 70/30 then inner holdout -> val(0.34)/test(0.66)  => ~10/20
tr, hold = train_test_split(g, test_size=0.30, random_state=SEED_OUTER, stratify=g.stratum)
vc2 = hold.stratum.value_counts()
hold_strat = hold.stratum.where(hold.stratum.map(vc2) >= 2, "rare")
val, test = train_test_split(hold, test_size=0.66, random_state=SEED_INNER, stratify=hold_strat)

fold_new = {}
for sub, name in [(tr, "train"), (val, "val"), (test, "test")]:
    for pid in sub.new_PatientID:
        fold_new[pid] = name
df.loc[new_mask, "Split_recovered"] = df.loc[new_mask, "new_PatientID"].map(fold_new)

# ---- report ----
src = np.where(df.Split.notna(), "inherited", "new")
rep = df.assign(src=src).groupby(["Split_recovered"]).agg(
    n_ecg=("npy_path", "size"),
    n_pt=("new_PatientID", "nunique"),
    ACS=("ACS", "sum"), ACCO=("ACCO", "sum"), AICO=("AICO", "sum"),
    LAD=("LAD", "sum"), RCA=("RCA", "sum"), LCX=("LCX", "sum"), Left_Main=("Left_Main", "sum"),
).reindex(["train", "val", "test"])
print("=== FINAL recovered split (v6, n=%d ECGs) ===" % len(df))
print(rep.to_string())
print("\n=== source breakdown (inherited canonical vs newly-split) ===")
print(df.assign(src=src).groupby(["Split_recovered", "src"]).size().unstack(fill_value=0)
      .reindex(["train", "val", "test"]).to_string())
print("\nTOTAL ACS+: %d  ACCO: %d  AICO: %d" % (df.ACS.sum(), df.ACCO.sum(), df.AICO.sum()))
print("any-unassigned:", int(df.Split_recovered.isna().sum()))

out = df[["npy_path", "new_PatientID", "Split_recovered",
          "condition_severity", "ACS", "ACCO", "AICO",
          "LAD", "RCA", "LCX", "Left_Main"]].rename(columns={"Split_recovered": "Split"})
out.to_csv(OUT, index=False)
print("\nwrote:", OUT, out.shape)
