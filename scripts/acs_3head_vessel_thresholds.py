#!/usr/bin/env python3
"""Per-vessel operating thresholds for the 3-head deployment model.
Population = ACCO subset (where the ACCO gate fires and a culprit is reported).
Threshold per territory fit on VAL (Youden's J); SENS/SPEC/PPV/NPV reported on TEST
at that threshold with 1000-iter bootstrap 95% CIs."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_curve
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel

A = "/volume/DeepECG-SSL-finetune/data/acs"
SP = "/tmp/claude-0/-volume-DeepECG-SSL-finetune/370fb1ed-6b36-4663-96aa-09465bce7a76/scratchpad/acs_v6_split_labels.csv"
VES = ["LAD", "RCA", "LCX", "Left_Main"]; SCALE = 0.00488; RNG = np.random.default_rng(42)

class ThreeHead(nn.Module):
    def __init__(self, base, d=768):
        super().__init__(); self.base = base
        self.h_acs = nn.Linear(d,1); self.h_acco = nn.Linear(d,1); self.h_ves = nn.Linear(d,4)
    def forward(self, source):
        res = ECGTransformerFinetuningModel.forward(self.base, source=source)
        x = res["x"]; pad = res["padding_mask"]; x = self.base.final_dropout(x)
        if pad is not None and pad.any(): x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        f = x.sum(1) / (x != 0).sum(1).clamp(min=1)
        return self.h_acs(f), self.h_acco(f), self.h_ves(f)

base = checkpoint_utils.load_model_and_task(f"{A}/checkpoints/acs/checkpoint_best.pt")[0]
base = base[0] if isinstance(base, list) else base; base.num_updates = 10**9
model = ThreeHead(base); ck = torch.load(f"{A}/checkpoints/acs_3head/best_model.pt", map_location="cpu")
model.load_state_dict(ck["model_state_dict"]); model.eval().cuda()

df = pd.read_csv(SP)
@torch.no_grad()
def vessel_probs(split):
    sub = df[df.Split == split].reset_index(drop=True)
    X = np.transpose(np.load(f"{A}/arrays/X_{split}.npy").astype(np.float32) * SCALE, (0,2,1))
    out = []
    for i in range(0, len(X), 256):
        with torch.cuda.amp.autocast():
            _, _, lv = model(torch.from_numpy(X[i:i+256]).cuda())
        out.append(torch.sigmoid(lv).float().cpu().numpy())
    pv = np.concatenate(out)
    acco = (sub.ACCO == 1).to_numpy()
    return pv[acco], sub.loc[acco, VES].to_numpy(np.float32)

pv_va, y_va = vessel_probs("val")
pv_te, y_te = vessel_probs("test")

def metrics(y, p, t):
    pred = p >= t
    tp = int((pred & (y==1)).sum()); fp = int((pred & (y==0)).sum())
    tn = int((~pred & (y==0)).sum()); fn = int((~pred & (y==1)).sum())
    sens = tp/(tp+fn) if tp+fn else np.nan; spec = tn/(tn+fp) if tn+fp else np.nan
    ppv = tp/(tp+fp) if tp+fp else np.nan; npv = tn/(tn+fn) if tn+fn else np.nan
    return sens, spec, ppv, npv

def boot(y, p, t, n=1000):
    idx = np.arange(len(y)); S=[]; SP_=[]; P=[]; N=[]
    for _ in range(n):
        b = RNG.choice(idx, len(idx), True)
        s,sp,pp,nv = metrics(y[b], p[b], t)
        S.append(s); SP_.append(sp); P.append(pp); N.append(nv)
    def ci(a): a=np.array(a,float); a=a[~np.isnan(a)]; return np.percentile(a,[2.5,97.5])
    return ci(S), ci(SP_), ci(P), ci(N)

print(f"ACCO subset: val n={len(y_va)}, test n={len(y_te)}")
print(f"\n{'Vessel':10s} {'thr':>6s} {'prev':>5s} {'Sens':>16s} {'Spec':>16s} {'PPV':>16s} {'NPV':>16s}")
rows = []
for i, v in enumerate(VES):
    fpr, tpr, thr = roc_curve(y_va[:, i], pv_va[:, i])
    t = float(thr[np.argmax(tpr - fpr)])               # Youden's J on val
    t = min(max(t, 0.0), 1.0)
    sens, spec, ppv, npv = metrics(y_te[:, i], pv_te[:, i], t)
    cS, cSp, cP, cN = boot(y_te[:, i], pv_te[:, i], t)
    prev = y_te[:, i].mean()
    fmt = lambda x, c: f"{x:.2f} ({c[0]:.2f}-{c[1]:.2f})"
    print(f"{v:10s} {t:6.3f} {prev:5.2f} {fmt(sens,cS):>16s} {fmt(spec,cSp):>16s} {fmt(ppv,cP):>16s} {fmt(npv,cN):>16s}")
    rows.append((v, t, int(y_te[:,i].sum()), sens, spec, ppv, npv, cP, cN))
