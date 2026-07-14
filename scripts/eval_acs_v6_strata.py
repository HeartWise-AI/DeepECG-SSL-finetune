#!/usr/bin/env python3
"""Vessel head AUROC stratified by occlusion completeness (all ACS+ / ACCO-only / AICO-only)
to compare apples-to-apples with the prior deployment numbers (which were ACCO-only)."""
import numpy as np, torch, pandas as pd
from sklearn.metrics import roc_auc_score
from fairseq_signals.utils import checkpoint_utils

SCALE = 0.00488
A = "/volume/DeepECG-SSL-finetune/data/acs"
SP = "/tmp/claude-0/-volume-DeepECG-SSL-finetune/370fb1ed-6b36-4663-96aa-09465bce7a76/scratchpad/acs_v6_split_labels.csv"
VES = ["LAD", "RCA", "LCX", "Left_Main"]
RNG = np.random.default_rng(42)

def ci(y, p, n=1000):
    if y.sum() == 0 or y.sum() == len(y): return (float("nan"),)*3
    a = roc_auc_score(y, p); idx = np.arange(len(y)); bs = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx), True)
        if 0 < y[b].sum() < len(b): bs.append(roc_auc_score(y[b], p[b]))
    lo, hi = np.percentile(bs, [2.5, 97.5]); return a, lo, hi

@torch.no_grad()
def score(ckpt, X):
    m = checkpoint_utils.load_model_and_task(ckpt)[0]
    m = (m[0] if isinstance(m, list) else m).float().cuda().eval()
    X = np.transpose(np.load(X).astype(np.float32) * SCALE, (0, 2, 1))
    o = [m(source=torch.from_numpy(X[i:i+256]).cuda())["out"].float().cpu().numpy() for i in range(0, len(X), 256)]
    return 1/(1+np.exp(-np.concatenate(o)))

# reconstruct condition_severity aligned to X_test_vessel (built as test rows, ACS+ in order)
df = pd.read_csv(SP)
sub = df[df.Split == "test"].reset_index(drop=True)
acs = sub[sub.ACS == 1].reset_index(drop=True)
acco = (acs.condition_severity == "Acute Complete Coronary Occlusion").to_numpy()

pv = score(f"{A}/checkpoints/vessel/checkpoint_best.pt", f"{A}/arrays/X_test_vessel.npy")
yv = np.load(f"{A}/arrays/Y_test_vessel.npy")
assert len(acco) == len(yv) == len(pv), (len(acco), len(yv), len(pv))

for name, mask in [("ALL ACS+", np.ones(len(acco), bool)), ("ACCO-only", acco), ("AICO-only", ~acco)]:
    print(f"\n=== vessel — {name} (n={int(mask.sum())}) ===")
    aus = []
    for i, v in enumerate(VES):
        a, lo, hi = ci(yv[mask, i], pv[mask, i])
        aus.append(a)
        print(f"  {v:10s} n_pos={int(yv[mask,i].sum()):3d}  AUROC = {a:.2f} (95% CI {lo:.2f}-{hi:.2f})")
    print(f"  macro AUROC = {np.nanmean(aus):.2f}")
