#!/usr/bin/env python3
"""Shared-backbone 3-head DeepECG-ACS with a SOFTMAX culprit-territory head.

Changes vs scripts/train_acs_3head.py (the 4-sigmoid v6-final model):
  1. The vessel head is a 3-way SOFTMAX (cross-entropy), not 4 independent sigmoids.
  2. It is trained ONLY on single-territory ACUTE COMPLETE OCCLUSIONS (ACCO==1 with
     exactly one territory flag) -- the same population it is evaluated on and the
     same population the deployment gate selects. The old head was trained on all
     ACS+ rows (ACCO + AICO, 11% multi-hot) and evaluated on ACCO only.
  3. Left_Main is merged into a LEFT ("left system") class with LAD until more cases
     accrue: DB-A has 2 single-vessel Left Main ECGs in train and 1 in test.

Heads:
  acs    : P(ACS = acute obstruction)      BCE, full Database A cohort
  acco   : P(acute complete occlusion)     BCE, full Database A cohort   (the gate)
  vessel : softmax over [LEFT, RCA, LCX]   CE, masked to single-territory ACCO rows

Deployment rule is unchanged: report a territory only when P(ACCO) > gate; the
territory is argmax of the softmax. Exports a .pt bundle + ONNX.
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import roc_auc_score

from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel

SCALE = 0.00488
A = "/volume/DeepECG-SSL-finetune/data/acs"           # backbone donor (architecture + init)
BASE = "/volume/DeepECG-SSL-finetune/data/acs_final"  # post-exclusion arrays (Database A)
SP = "/media/data1/datasets/DeepECG/acs_v6_final_labels.csv"
OUT = f"{BASE}/checkpoints/acs_3head_softmax"
CLASSES = ["LEFT", "RCA", "LCX"]      # LEFT = LAD or Left_Main
DEV = "cuda"
BATCH = int(os.environ.get("ACS_BATCH", 96))
SEED = int(os.environ.get("ACS_SEED", 42))
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- data
df = pd.read_csv(SP, low_memory=False)
df = df[df.in_database_A == 1].reset_index(drop=True)   # same order as build_acs_v6_npy.py
df["LEFT"] = ((df.LAD == 1) | (df.Left_Main == 1)).astype(int)


def load_split(split):
    sub = df[df.Split == split].reset_index(drop=True)
    X = np.load(f"{BASE}/arrays/X_{split}.npy")                       # (N,2500,12) raw ADC
    y_acs = sub.ACS.to_numpy(np.float32)
    y_acco = sub.ACCO.to_numpy(np.float32)
    onehot = sub[CLASSES].to_numpy(np.float32)
    n_terr = onehot.sum(1)
    m_ves = ((sub.ACCO == 1).to_numpy() & (n_terr == 1)).astype(np.float32)
    y_ves = np.where(n_terr == 1, onehot.argmax(1), -1).astype(np.int64)   # -1 where masked
    return X, y_acs, y_acco, y_ves, m_ves


tr = load_split("train")
va = load_split("val")
te = load_split("test")
for nm, d in (("train", tr), ("val", va), ("test", te)):
    counts = {c: int((d[3][d[4] == 1] == i).sum()) for i, c in enumerate(CLASSES)}
    print(f"{nm:5s} n={d[0].shape[0]:6d}  ACS+={int(d[1].sum()):5d}  ACCO={int(d[2].sum()):4d}  "
          f"single-territory ACCO={int(d[4].sum()):4d} {counts}", flush=True)

wandb.init(project="deepecg-acs", name=f"v6final-3head-softmax-s{SEED}",
           config={"cohort": "Database A (post-exclusion: CABG/LBBB/paced/graft/non-coronary)",
                   "vessel_head": "3-way softmax [LEFT(LAD|Left_Main), RCA, LCX]",
                   "vessel_cohort": "single-territory acute COMPLETE occlusions (ACCO) only",
                   "dataset": "acs_v6_final_labels.csv", "backbone": "WCRv2 amp-preserved",
                   "lr": 5e-5, "weight_decay": 5e-3, "batch": BATCH, "max_epoch": 40,
                   "patience": 8, "seed": SEED,
                   "n_train": int(tr[0].shape[0]), "n_val": int(va[0].shape[0]),
                   "n_test": int(te[0].shape[0]),
                   "n_vessel_train": int(tr[4].sum()), "n_vessel_val": int(va[4].sum()),
                   "n_vessel_test": int(te[4].sum()),
                   "select_metric": "0.5*ACCO_auroc + 0.5*vessel_macro_ovr"})


# ---------------------------------------------------------------- model
class ThreeHeadSoftmax(nn.Module):
    def __init__(self, base, d=768, n_cls=3):
        super().__init__()
        self.base = base
        self.h_acs = nn.Linear(d, 1)
        self.h_acco = nn.Linear(d, 1)
        self.h_ves = nn.Linear(d, n_cls)
        for h in (self.h_acs, self.h_acco, self.h_ves):
            nn.init.xavier_uniform_(h.weight)
            nn.init.constant_(h.bias, 0.0)

    def pooled(self, source):
        res = ECGTransformerFinetuningModel.forward(self.base, source=source)
        x = res["x"]
        pad = res["padding_mask"]
        x = self.base.final_dropout(x)
        if pad is not None and pad.any():
            x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        return x.sum(1) / (x != 0).sum(1).clamp(min=1)

    def forward(self, source):
        f = self.pooled(source)
        return self.h_acs(f), self.h_acco(f), self.h_ves(f)


base = checkpoint_utils.load_model_and_task(f"{A}/checkpoints/acs/checkpoint_best.pt")[0]
base = base[0] if isinstance(base, list) else base
base.num_updates = 10 ** 9          # ft=True -> backbone trains
model = ThreeHeadSoftmax(base).to(DEV)


# ---------------------------------------------------------------- helpers
def batches(n, bs, shuffle):
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def to_in(Xb):
    x = np.transpose(Xb.astype(np.float32) * SCALE, (0, 2, 1))   # (B,12,2500) mV
    return torch.from_numpy(x).to(DEV)


@torch.no_grad()
def predict(split_data, bs=256):
    model.eval()
    X = split_data[0]
    pa, pc, pv = [], [], []
    for b in batches(len(X), bs, False):
        with torch.cuda.amp.autocast():
            la, lc, lv = model(to_in(X[b]))
        pa.append(torch.sigmoid(la).float().cpu().numpy())
        pc.append(torch.sigmoid(lc).float().cpu().numpy())
        pv.append(torch.softmax(lv.float(), dim=1).cpu().numpy())
    return np.concatenate(pa)[:, 0], np.concatenate(pc)[:, 0], np.concatenate(pv)


def vessel_macro_ovr(y_cls, p_ves, mask):
    """Macro one-vs-rest AUROC over single-territory ACCO rows."""
    m = mask.astype(bool)
    y = y_cls[m]
    p = p_ves[m]
    aus = []
    for i in range(len(CLASSES)):
        yt = (y == i).astype(int)
        if 0 < yt.sum() < len(yt):
            aus.append(roc_auc_score(yt, p[:, i]))
    return float(np.mean(aus)) if aus else float("nan")


def boot_auroc(y, p, n=1000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    idx = np.arange(len(y))
    aus = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if 0 < y[b].sum() < len(b):
            aus.append(roc_auc_score(y[b], p[b]))
    lo, hi = np.percentile(aus, [2.5, 97.5]) if aus else (np.nan, np.nan)
    return roc_auc_score(y, p), lo, hi


def boot_acc(correct, n=1000, rng=None):
    rng = rng or np.random.default_rng(SEED)
    idx = np.arange(len(correct))
    accs = [correct[rng.choice(idx, len(idx), True)].mean() for _ in range(n)]
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(correct.mean()), float(lo), float(hi)


# ---------------------------------------------------------------- train
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-3)
scaler = torch.cuda.amp.GradScaler()
bce = nn.BCEWithLogitsLoss()

Xtr, ya, yc, yv, mv = tr
best, best_state, bad = -1, None, 0
for epoch in range(1, 41):
    model.train()
    for b in batches(len(Xtr), BATCH, True):
        xb = to_in(Xtr[b])
        tb_a = torch.from_numpy(ya[b]).to(DEV)
        tb_c = torch.from_numpy(yc[b]).to(DEV)
        tb_v = torch.from_numpy(np.maximum(yv[b], 0)).to(DEV)
        mb = torch.from_numpy(mv[b]).to(DEV)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            la, lc, lv = model(xb)
            loss = bce(la[:, 0], tb_a) + bce(lc[:, 0], tb_c)
            if mb.sum() > 0:
                ce = nn.functional.cross_entropy(lv.float(), tb_v, reduction="none")
                loss = loss + (ce * mb).sum() / mb.sum()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    pa, pc, pv = predict(va)
    au_acs = roc_auc_score(va[1], pa)
    au_acco = roc_auc_score(va[2], pc)
    vmac = vessel_macro_ovr(va[3], pv, va[4])
    m = va[4].astype(bool)
    top1 = float((pv[m].argmax(1) == va[3][m]).mean())
    combined = 0.5 * au_acco + 0.5 * vmac
    print(f"epoch {epoch:2d}  val ACS {au_acs:.3f}  ACCO {au_acco:.3f}  "
          f"vessel_macro {vmac:.3f}  top1 {top1:.3f}  combined {combined:.4f}", flush=True)
    wandb.log({"epoch": epoch, "val/acs_auroc": au_acs, "val/acco_auroc": au_acco,
               "val/vessel_macro_ovr": vmac, "val/vessel_top1": top1,
               "val/combined": combined, "val/best_combined": max(best, combined)})
    if combined > best:
        best, bad = combined, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 8:
            print("early stop", flush=True)
            break

model.load_state_dict(best_state)

# ---------------------------------------------------------------- gate (val, ~90% spec)
pa, pc, pv = predict(va)
GATE = float(np.percentile(pc[va[2] == 0], 90))
print(f"\nGATE (P(ACCO) @ ~90% val spec) = {GATE:.6g}", flush=True)

# ---------------------------------------------------------------- test
pa, pc, pv = predict(te)
ya_t, yc_t, yv_t, mv_t = te[1], te[2], te[3], te[4]
m = mv_t.astype(bool)

au_acs, lo_a, hi_a = boot_auroc(ya_t, pa)
au_acco, lo_c, hi_c = boot_auroc(yc_t, pc)
print(f"\n=== TEST (n={len(ya_t)}) ===")
print(f"  ACS detection AUROC = {au_acs:.2f} (95% CI {lo_a:.2f}-{hi_a:.2f})")
print(f"  ACCO-vs-rest AUROC  = {au_acco:.2f} (95% CI {lo_c:.2f}-{hi_c:.2f})")
summary = {"test/acs_auroc": au_acs, "test/acs_auroc_lo": lo_a, "test/acs_auroc_hi": hi_a,
           "test/acco_auroc": au_acco, "test/acco_auroc_lo": lo_c, "test/acco_auroc_hi": hi_c,
           "gate": GATE, "n_test": int(len(ya_t)), "n_vessel_test": int(m.sum())}

print(f"\n  culprit territory, single-territory ACCO test rows (n={int(m.sum())}):")
aus = []
for i, c in enumerate(CLASSES):
    yt = (yv_t[m] == i).astype(int)
    if 0 < yt.sum() < len(yt):
        au, lo, hi = boot_auroc(yt, pv[m][:, i])
        aus.append(au)
        print(f"    {c:5s} n_pos={int(yt.sum()):3d}  AUROC = {au:.2f} (95% CI {lo:.2f}-{hi:.2f})")
        summary[f"test/vessel_{c}_auroc"] = au
        summary[f"test/vessel_{c}_lo"] = lo
        summary[f"test/vessel_{c}_hi"] = hi
summary["test/vessel_macro_ovr"] = float(np.mean(aus))
print(f"    macro OVR AUROC = {np.mean(aus):.2f}")

acc, lo, hi = boot_acc((pv[m].argmax(1) == yv_t[m]).astype(float))
print(f"    top-1 accuracy (all single-territory ACCO) = {acc*100:.1f}% "
      f"(95% CI {lo*100:.1f}-{hi*100:.1f})")
summary["test/top1_all"] = acc

fired = m & (pc > GATE)
if fired.sum() > 0:
    acc_g, lo_g, hi_g = boot_acc((pv[fired].argmax(1) == yv_t[fired]).astype(float))
    print(f"    top-1 accuracy when gate fires (n={int(fired.sum())}) = {acc_g*100:.1f}% "
          f"(95% CI {lo_g*100:.1f}-{hi_g*100:.1f})")
    summary["test/top1_gate_fired"] = acc_g
    summary["test/n_gate_fired"] = int(fired.sum())

cm = np.zeros((len(CLASSES), len(CLASSES)), int)
for t, p_ in zip(yv_t[m], pv[m].argmax(1)):
    cm[t, p_] += 1
print("\n  confusion (rows = true, cols = predicted): " + " ".join(f"{c:>5s}" for c in CLASSES))
for i, c in enumerate(CLASSES):
    print(f"    {c:5s} " + " ".join(f"{v:5d}" for v in cm[i]))

summary["test/gate_fire_rate_nonACCO"] = float((pc[yc_t == 0] > GATE).mean())
print(f"\n  gate fire rate on non-ACCO (1-spec) = {summary['test/gate_fire_rate_nonACCO']*100:.1f}%")
wandb.log(summary)
wandb.summary.update(summary)

# ---------------------------------------------------------------- save
torch.save({"model_state_dict": best_state, "gate": GATE, "classes": CLASSES,
            "scale": SCALE, "head": "softmax", "seed": SEED,
            "vessel_train_population": "single-territory ACCO"},
           f"{OUT}/best_model.pt")
np.savez(f"{OUT}/test_predictions.npz", p_acs=pa, p_acco=pc, p_vessel=pv,
         y_acs=ya_t, y_acco=yc_t, y_vessel=yv_t, vessel_mask=mv_t, gate=GATE,
         classes=np.array(CLASSES))
print(f"\nsaved {OUT}/best_model.pt", flush=True)


class Export(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, ecg_12lead):
        la, lc, lv = self.m(ecg_12lead)
        return torch.sigmoid(la), torch.sigmoid(lc), torch.softmax(lv, dim=1)


model.eval()
try:
    exp = Export(model).to(DEV).eval()
    dummy = torch.randn(1, 12, 2500, device=DEV)
    torch.onnx.export(exp, dummy, f"{OUT}/acs_acco_territory_softmax.onnx",
                      input_names=["ecg_12lead"],
                      output_names=["acs_probability", "acco_probability", "territory_probability"],
                      dynamic_axes={"ecg_12lead": {0: "B"}, "acs_probability": {0: "B"},
                                    "acco_probability": {0: "B"}, "territory_probability": {0: "B"}},
                      opset_version=14)
    print(f"saved {OUT}/acs_acco_territory_softmax.onnx", flush=True)
except Exception as e:
    print(f"ONNX export failed ({e}); TorchScript fallback", flush=True)
    try:
        ts = torch.jit.trace(Export(model).eval(), torch.randn(1, 12, 2500, device=DEV))
        ts.save(f"{OUT}/acs_acco_territory_softmax.ts.pt")
        print("saved TorchScript", flush=True)
    except Exception as e2:
        print(f"TorchScript also failed ({e2})", flush=True)

try:
    art = wandb.Artifact("acs_3head_softmax_v6final", type="model")
    art.add_file(f"{OUT}/best_model.pt")
    if os.path.exists(f"{OUT}/acs_acco_territory_softmax.onnx"):
        art.add_file(f"{OUT}/acs_acco_territory_softmax.onnx")
    wandb.log_artifact(art)
except Exception as e:
    print(f"wandb artifact log skipped ({e})", flush=True)
wandb.finish()
print("DONE", flush=True)
