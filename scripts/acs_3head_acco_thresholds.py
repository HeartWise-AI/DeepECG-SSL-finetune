#!/usr/bin/env python3
"""ACCO gate operating points (ACCO-vs-rest, full cohort). Thresholds fit on VAL,
metrics on TEST with 1000-iter bootstrap 95% CIs. Shows the PPV<->NPV trade-off."""
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_curve, f1_score
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel

A = "/volume/DeepECG-SSL-finetune/data/acs"
SP = "/tmp/claude-0/-volume-DeepECG-SSL-finetune/370fb1ed-6b36-4663-96aa-09465bce7a76/scratchpad/acs_v6_split_labels.csv"
SCALE = 0.00488; RNG = np.random.default_rng(42)

class ThreeHead(nn.Module):
    def __init__(self, base, d=768):
        super().__init__(); self.base = base
        self.h_acs = nn.Linear(d,1); self.h_acco = nn.Linear(d,1); self.h_ves = nn.Linear(d,4)
    def forward(self, source):
        res = ECGTransformerFinetuningModel.forward(self.base, source=source)
        x = res["x"]; pad = res["padding_mask"]; x = self.base.final_dropout(x)
        if pad is not None and pad.any(): x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        f = x.sum(1)/(x!=0).sum(1).clamp(min=1)
        return self.h_acs(f), self.h_acco(f), self.h_ves(f)

base = checkpoint_utils.load_model_and_task(f"{A}/checkpoints/acs/checkpoint_best.pt")[0]
base = base[0] if isinstance(base, list) else base; base.num_updates = 10**9
model = ThreeHead(base); ck = torch.load(f"{A}/checkpoints/acs_3head/best_model.pt", map_location="cpu")
model.load_state_dict(ck["model_state_dict"]); model.eval().cuda()
df = pd.read_csv(SP)

@torch.no_grad()
def acco_prob(split):
    sub = df[df.Split == split].reset_index(drop=True)
    X = np.transpose(np.load(f"{A}/arrays/X_{split}.npy").astype(np.float32)*SCALE, (0,2,1))
    out=[]
    for i in range(0,len(X),256):
        with torch.cuda.amp.autocast():
            _, lc, _ = model(torch.from_numpy(X[i:i+256]).cuda())
        out.append(torch.sigmoid(lc).float().cpu().numpy())
    return np.concatenate(out)[:,0], sub.ACCO.to_numpy(np.float32)

pv, yv = acco_prob("val"); pt, yt = acco_prob("test")

def m(y,p,t):
    pred=p>=t; tp=int((pred&(y==1)).sum()); fp=int((pred&(y==0)).sum())
    tn=int((~pred&(y==0)).sum()); fn=int((~pred&(y==1)).sum())
    return (tp/(tp+fn) if tp+fn else np.nan, tn/(tn+fp) if tn+fp else np.nan,
            tp/(tp+fp) if tp+fp else np.nan, tn/(tn+fn) if tn+fn else np.nan)
def boot(y,p,t,n=1000):
    idx=np.arange(len(y)); acc=[[],[],[],[]]
    for _ in range(n):
        b=RNG.choice(idx,len(idx),True)
        for k,val in enumerate(m(y[b],p[b],t)): acc[k].append(val)
    def ci(a): a=np.array(a,float); a=a[~np.isnan(a)]; return np.percentile(a,[2.5,97.5])
    return [ci(a) for a in acc]

# thresholds fit on VAL
fpr,tpr,thr=roc_curve(yv,pv)
youden=float(thr[np.argmax(tpr-fpr)])
neg=pv[yv==0]; pos=pv[yv==1]
spec90=float(np.percentile(neg,90)); spec95=float(np.percentile(neg,95)); spec99=float(np.percentile(neg,99))
sens95=float(np.percentile(pos,5))   # threshold catching 95% of positives -> high NPV/rule-out
# F1-optimal on val
cand=np.unique(np.quantile(pv,np.linspace(0,1,200)))
f1s=[f1_score(yv,(pv>=c).astype(int),zero_division=0) for c in cand]
f1opt=float(cand[int(np.argmax(f1s))])

pts=[("current gate 0.002677",0.002677),("Youden J (balanced)",youden),("F1-optimal",f1opt),
     ("spec=90% (rule-in)",spec90),("spec=95% (rule-in)",spec95),("spec=99% (max PPV)",spec99),
     ("sens=95% (rule-out / max NPV)",sens95)]
print(f"TEST ACCO-vs-rest: n={len(yt)}, ACCO+={int(yt.sum())} (prev {yt.mean()*100:.1f}%)\n")
print(f"{'operating point':30s} {'thr':>9s} {'Sens':>15s} {'Spec':>15s} {'PPV':>15s} {'NPV':>15s}")
for name,t in pts:
    s,sp,pp,nv=m(yt,pt,t); cs,csp,cp,cn=boot(yt,pt,t)
    f=lambda x,c: f"{x:.2f}({c[0]:.2f}-{c[1]:.2f})"
    print(f"{name:30s} {t:9.5f} {f(s,cs):>15s} {f(sp,csp):>15s} {f(pp,cp):>15s} {f(nv,cn):>15s}")
