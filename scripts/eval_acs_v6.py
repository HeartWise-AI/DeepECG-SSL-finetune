#!/usr/bin/env python3
"""Score the trained DeepECG-ACS v6 heads on the held-out test set.
AUROC/AUPRC with 1000-iter bootstrap 95% CIs. Raw arrays -> x0.00488 mV."""
import numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score
from fairseq_signals.utils import checkpoint_utils

SCALE = 0.00488
A = "/volume/DeepECG-SSL-finetune/data/acs"
RNG = np.random.default_rng(42)

def boot_ci(y, p, n=1000):
    auroc = roc_auc_score(y, p); auprc = average_precision_score(y, p)
    idx = np.arange(len(y)); aus = []; aps = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx), replace=True)
        if y[b].sum() == 0 or y[b].sum() == len(b):
            continue
        aus.append(roc_auc_score(y[b], p[b])); aps.append(average_precision_score(y[b], p[b]))
    lo, hi = np.percentile(aus, [2.5, 97.5]); plo, phi = np.percentile(aps, [2.5, 97.5])
    return auroc, lo, hi, auprc, plo, phi

@torch.no_grad()
def score(ckpt, Xnpy, batch=256):
    models, cfg, task = checkpoint_utils.load_model_and_task(ckpt)
    m = models[0] if isinstance(models, list) else models
    m = m.float().cuda().eval()
    X = np.load(Xnpy).astype(np.float32) * SCALE          # (N,2500,12) -> mV
    X = np.transpose(X, (0, 2, 1))                          # (N,12,2500)
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i+batch]).cuda()
        logits = m(source=xb)["out"].float().cpu().numpy()
        out.append(logits)
    return 1 / (1 + np.exp(-np.concatenate(out)))           # sigmoid probs

print("=== ACS detection head (binary Acute_Obstruction) ===")
p = score(f"{A}/checkpoints/acs/checkpoint_best.pt", f"{A}/arrays/X_test.npy")[:, 0]
y = np.load(f"{A}/arrays/Y_test_acs.npy")[:, 0]
au, lo, hi, ap, plo, phi = boot_ci(y, p)
print(f"  n={len(y)}  pos={int(y.sum())}")
print(f"  AUROC = {au:.2f} (95% CI {lo:.2f}–{hi:.2f})")
print(f"  AUPRC = {ap:.2f} (95% CI {plo:.2f}–{phi:.2f})")

print("\n=== Culprit-vessel head (4-label, ACS+ test subset) ===")
pv = score(f"{A}/checkpoints/vessel/checkpoint_best.pt", f"{A}/arrays/X_test_vessel.npy")
yv = np.load(f"{A}/arrays/Y_test_vessel.npy")
VES = ["LAD", "RCA", "LCX", "Left_Main"]
aurocs = []
for i, v in enumerate(VES):
    au, lo, hi, ap, plo, phi = boot_ci(yv[:, i], pv[:, i])
    aurocs.append(au)
    print(f"  {v:10s} n_pos={int(yv[:,i].sum()):3d}  AUROC = {au:.2f} (95% CI {lo:.2f}–{hi:.2f})  AUPRC = {ap:.2f}")
print(f"  macro AUROC = {np.mean(aurocs):.2f}")
