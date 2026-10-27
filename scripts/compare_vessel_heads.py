#!/usr/bin/env python3
"""Head-to-head: 4-sigmoid vessel head (v6-final) vs 3-way softmax territory head.

Both models share the WCRv2 backbone, the Database A cohort, the same split and the same
recipe. They are compared on ONE population -- the held-out single-territory acute complete
occlusions -- which is the population the deployment gate actually selects and the only one
where "which territory" is a well-posed question.

The old head's 4 sigmoids are mapped to the same 3 classes as the new head:
    LEFT = max(P(LAD), P(Left_Main)),  RCA = P(RCA),  LCX = P(LCX)
so top-1 is decided over identical label space. Paired comparison (same ECGs, same order)
with bootstrap CIs and an exact McNemar test on top-1 correctness.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel

SCALE = 0.00488
A = "/volume/DeepECG-SSL-finetune/data/acs"
BASE = "/volume/DeepECG-SSL-finetune/data/acs_final"
OLD = f"{BASE}/checkpoints/acs_3head/best_model.pt"
NEW = f"{BASE}/checkpoints/acs_3head_softmax/best_model.pt"
SP = "/media/data1/datasets/DeepECG/acs_v6_final_labels.csv"
OUT = f"{BASE}/explainability/vessel_head_comparison.json"
VES4 = ["LAD", "RCA", "LCX", "Left_Main"]
CLASSES = ["LEFT", "RCA", "LCX"]
DEV = "cuda"
RNG = np.random.default_rng(42)


class ThreeHead(nn.Module):
    def __init__(self, base, d=768, n_out=4):
        super().__init__()
        self.base = base
        self.h_acs = nn.Linear(d, 1)
        self.h_acco = nn.Linear(d, 1)
        self.h_ves = nn.Linear(d, n_out)

    def forward(self, source):
        res = ECGTransformerFinetuningModel.forward(self.base, source=source)
        x = res["x"]
        pad = res["padding_mask"]
        x = self.base.final_dropout(x)
        if pad is not None and pad.any():
            x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        f = x.sum(1) / (x != 0).sum(1).clamp(min=1)
        return self.h_acs(f), self.h_acco(f), self.h_ves(f)


def load(ckpt, n_out):
    base = checkpoint_utils.load_model_and_task(f"{A}/checkpoints/acs/checkpoint_best.pt")[0]
    base = base[0] if isinstance(base, list) else base
    base.num_updates = 10 ** 9
    m = ThreeHead(base, n_out=n_out)
    ck = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(ck["model_state_dict"])
    return m.eval().to(DEV), ck


@torch.no_grad()
def run(model, X, softmax, bs=256):
    pa, pc, pv = [], [], []
    for i in range(0, len(X), bs):
        xb = np.transpose(np.asarray(X[i:i + bs], np.float32) * SCALE, (0, 2, 1))
        with torch.cuda.amp.autocast():
            la, lc, lv = model(torch.from_numpy(np.ascontiguousarray(xb)).to(DEV))
        pa.append(torch.sigmoid(la.float())[:, 0].cpu().numpy())
        pc.append(torch.sigmoid(lc.float())[:, 0].cpu().numpy())
        pv.append((torch.softmax(lv.float(), 1) if softmax
                   else torch.sigmoid(lv.float())).cpu().numpy())
    return np.concatenate(pa), np.concatenate(pc), np.concatenate(pv)


def boot_auroc(y, p, n=1000):
    idx = np.arange(len(y))
    aus = []
    for _ in range(n):
        b = RNG.choice(idx, len(idx), True)
        if 0 < y[b].sum() < len(b):
            aus.append(roc_auc_score(y[b], p[b]))
    return float(roc_auc_score(y, p)), float(np.percentile(aus, 2.5)), float(np.percentile(aus, 97.5))


def boot_acc(c, n=1000):
    idx = np.arange(len(c))
    a = [c[RNG.choice(idx, len(idx), True)].mean() for _ in range(n)]
    return float(c.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    df = pd.read_csv(SP, low_memory=False)
    df = df[df.in_database_A == 1].reset_index(drop=True)
    df["LEFT"] = ((df.LAD == 1) | (df.Left_Main == 1)).astype(int)
    te = df[df.Split == "test"].reset_index(drop=True)
    onehot = te[CLASSES].to_numpy(np.float32)
    n_terr = onehot.sum(1)
    m = (te.ACCO == 1).to_numpy() & (n_terr == 1)
    y = np.where(n_terr == 1, onehot.argmax(1), -1)[m]
    X = np.load(f"{BASE}/arrays/X_test.npy", mmap_mode="r")
    Xm = np.asarray(X[np.where(m)[0]])
    print(f"paired population: single-territory ACCO test ECGs, n={len(Xm)}")
    print("  " + ", ".join(f"{c}={int((y==i).sum())}" for i, c in enumerate(CLASSES)))

    res = {"n": int(len(Xm)),
           "class_counts": {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}}

    # full held-out test set: are the shared ACS / ACCO heads comparable between the two runs?
    Xfull = np.asarray(X)
    y_acs = te.ACS.to_numpy(np.float32)
    y_acco = te.ACCO.to_numpy(np.float32)
    res["full_test"] = {"n": int(len(Xfull))}

    old, ck_old = load(OLD, 4)
    pa_f, pc_f, _ = run(old, Xfull, softmax=False)
    for nm, yy, pp in (("acs", y_acs, pa_f), ("acco", y_acco, pc_f)):
        au, lo, hi = boot_auroc(yy, pp)
        res["full_test"].setdefault("sigmoid_4head", {})[nm] = {"auroc": au, "ci": [lo, hi]}
        print(f"  [full test] sigmoid_4head {nm.upper():4s} AUROC {au:.2f} ({lo:.2f}-{hi:.2f})")
    _, pc_old, pv4 = run(old, Xm, softmax=False)
    del old
    torch.cuda.empty_cache()
    p_old = np.stack([np.maximum(pv4[:, VES4.index("LAD")], pv4[:, VES4.index("Left_Main")]),
                      pv4[:, VES4.index("RCA")], pv4[:, VES4.index("LCX")]], 1)

    new, ck_new = load(NEW, 3)
    pa_f, pc_f, _ = run(new, Xfull, softmax=True)
    for nm, yy, pp in (("acs", y_acs, pa_f), ("acco", y_acco, pc_f)):
        au, lo, hi = boot_auroc(yy, pp)
        res["full_test"].setdefault("softmax_3class", {})[nm] = {"auroc": au, "ci": [lo, hi]}
        print(f"  [full test] softmax_3class {nm.upper():4s} AUROC {au:.2f} ({lo:.2f}-{hi:.2f})")
    _, pc_new, p_new = run(new, Xm, softmax=True)
    del new
    torch.cuda.empty_cache()

    for name, p, gate, pc in (("sigmoid_4head", p_old, float(ck_old["gate"]), pc_old),
                              ("softmax_3class", p_new, float(ck_new["gate"]), pc_new)):
        r = {"gate": gate}
        aus = []
        for i, c in enumerate(CLASSES):
            au, lo, hi = boot_auroc((y == i).astype(int), p[:, i])
            aus.append(au)
            r[c] = {"auroc": au, "ci": [lo, hi]}
        r["macro_ovr_auroc"] = float(np.mean(aus))
        correct = (p.argmax(1) == y).astype(float)
        acc, lo, hi = boot_acc(correct)
        r["top1"] = {"acc": acc, "ci": [lo, hi]}
        fired = pc > gate
        if fired.sum():
            a2, l2, h2 = boot_acc(correct[fired])
            r["top1_gate_fired"] = {"acc": a2, "ci": [l2, h2], "n": int(fired.sum()),
                                    "gate_sensitivity_on_this_population": float(fired.mean())}
        cm = np.zeros((3, 3), int)
        for t, q in zip(y, p.argmax(1)):
            cm[t, q] += 1
        r["confusion"] = cm.tolist()
        res[name] = r
        print(f"\n=== {name} (gate {gate:.6g}) ===")
        for c in CLASSES:
            print(f"  {c:5s} AUROC {r[c]['auroc']:.2f} ({r[c]['ci'][0]:.2f}-{r[c]['ci'][1]:.2f})")
        print(f"  macro OVR AUROC {r['macro_ovr_auroc']:.2f}")
        print(f"  top-1 {acc*100:.1f}% ({lo*100:.1f}-{hi*100:.1f})")
        if fired.sum():
            print(f"  top-1 | gate fired (n={int(fired.sum())}) {r['top1_gate_fired']['acc']*100:.1f}%"
                  f" ({r['top1_gate_fired']['ci'][0]*100:.1f}-{r['top1_gate_fired']['ci'][1]*100:.1f})")
        print("  confusion (rows true, cols pred): " + " ".join(f"{c:>5s}" for c in CLASSES))
        for i, c in enumerate(CLASSES):
            print(f"    {c:5s} " + " ".join(f"{v:5d}" for v in cm[i]))

    c_old = (p_old.argmax(1) == y)
    c_new = (p_new.argmax(1) == y)
    b = int((c_new & ~c_old).sum())
    cc = int((~c_new & c_old).sum())
    pval = binomtest(b, b + cc, 0.5).pvalue if (b + cc) else 1.0
    diff, lo, hi = boot_acc((c_new.astype(float) - c_old.astype(float)))
    res["paired"] = {"softmax_right_sigmoid_wrong": b, "sigmoid_right_softmax_wrong": cc,
                     "mcnemar_p": float(pval), "top1_diff": diff, "top1_diff_ci": [lo, hi]}
    print(f"\npaired top-1: softmax correct / sigmoid wrong = {b}; reverse = {cc}; "
          f"McNemar p = {pval:.3g}")
    print(f"top-1 difference (softmax - sigmoid) = {diff*100:+.1f} pp "
          f"(95% CI {lo*100:+.1f} to {hi*100:+.1f})")

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
