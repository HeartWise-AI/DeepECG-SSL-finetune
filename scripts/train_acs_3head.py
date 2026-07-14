#!/usr/bin/env python3
"""Shared-backbone 3-head DeepECG-ACS (faithful to prior deployment).

One WCRv2 backbone -> 3 heads:
  - acs    : P(Acute_Obstruction)   trained on full cohort
  - acco   : P(ACCO complete-occl.) trained on full cohort   (the gate)
  - vessel : [LAD,RCA,LCX,Left_Main] trained on ACS+ subset only (masked loss)
Joint masked BCE. Gate threshold fit on val (~90% specificity for ACCO-vs-rest).
Deployment rule: show vessel only when P(ACCO) > gate. Exports a single .pt bundle + ONNX.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, wandb
from sklearn.metrics import roc_auc_score
from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel

SCALE = 0.00488
A = "/volume/DeepECG-SSL-finetune/data/acs"              # existing ACS head for backbone init
BASE = "/volume/DeepECG-SSL-finetune/data/acs_final"     # new post-exclusion arrays + checkpoints
SP = "/media/data1/datasets/DeepECG/acs_v6_final_labels.csv"
OUT = f"{BASE}/checkpoints/acs_3head"
VES = ["LAD", "RCA", "LCX", "Left_Main"]
DEV = "cuda"
torch.manual_seed(42)
import os; os.makedirs(OUT, exist_ok=True)

# ---------------- data (Database A; vessel head = Database B via anyvessel mask) ----------------
df = pd.read_csv(SP, low_memory=False)
df = df[df.in_database_A == 1].reset_index(drop=True)
def load_split(split):
    sub = df[df.Split == split].reset_index(drop=True)
    X = np.load(f"{BASE}/arrays/X_{split}.npy")                   # (N,2500,12) raw, Database A
    y_acs  = sub.ACS.to_numpy(np.float32)
    y_acco = sub.ACCO.to_numpy(np.float32)
    y_ves  = sub[VES].to_numpy(np.float32)
    m_ves  = ((sub.ACS == 1) & (sub.anyvessel == 1)).to_numpy(np.float32)  # Database B: ACS+ w/ culprit
    return X, y_acs, y_acco, y_ves, m_ves
tr = load_split("train"); va = load_split("val"); te = load_split("test")
print(f"train {tr[0].shape[0]}  val {va[0].shape[0]}  test {te[0].shape[0]}", flush=True)

wandb.init(project="deepecg-acs", name="v6final-3head-databaseA",
           config={"cohort": "Database A (post-exclusion: CABG/LBBB/paced/graft/non-coronary)",
                   "vessel_cohort": "Database B (ACS+ w/ native culprit)",
                   "dataset": "acs_v6_final_labels.csv", "backbone": "WCRv2 amp-preserved",
                   "lr": 5e-5, "weight_decay": 5e-3, "batch": 128, "max_epoch": 40, "patience": 8,
                   "n_train": int(tr[0].shape[0]), "n_val": int(va[0].shape[0]), "n_test": int(te[0].shape[0]),
                   "select_metric": "0.5*ACCO_auroc + 0.5*vessel_macro"})

# ---------------- model ----------------
class ThreeHead(nn.Module):
    def __init__(self, base, d=768):
        super().__init__()
        self.base = base
        self.h_acs = nn.Linear(d, 1); self.h_acco = nn.Linear(d, 1); self.h_ves = nn.Linear(d, 4)
        for h in (self.h_acs, self.h_acco, self.h_ves):
            nn.init.xavier_uniform_(h.weight); nn.init.constant_(h.bias, 0.0)
    def pooled(self, source):
        res = ECGTransformerFinetuningModel.forward(self.base, source=source)
        x = res["x"]; pad = res["padding_mask"]
        x = self.base.final_dropout(x)
        if pad is not None and pad.any(): x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        return x.sum(1) / (x != 0).sum(1).clamp(min=1)
    def forward(self, source):
        f = self.pooled(source)
        return self.h_acs(f), self.h_acco(f), self.h_ves(f)

base = checkpoint_utils.load_model_and_task(f"{A}/checkpoints/acs/checkpoint_best.pt")[0]
base = base[0] if isinstance(base, list) else base
base.num_updates = 10**9  # ensure ft=True (backbone trains)
model = ThreeHead(base).to(DEV)

# ---------------- train ----------------
def batches(X, n, bs, shuffle):
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs): yield idx[i:i+bs]
def to_in(Xb):
    x = np.transpose(Xb.astype(np.float32) * SCALE, (0, 2, 1))   # (B,12,2500) mV
    return torch.from_numpy(x).to(DEV)

opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-3)
scaler = torch.cuda.amp.GradScaler()
bce = nn.BCEWithLogitsLoss()

@torch.no_grad()
def predict(split_data, bs=256):
    model.eval(); X = split_data[0]; n = len(X)
    pa, pc, pv = [], [], []
    for b in batches(X, n, bs, False):
        with torch.cuda.amp.autocast():
            la, lc, lv = model(to_in(X[b]))
        pa.append(torch.sigmoid(la).float().cpu().numpy())
        pc.append(torch.sigmoid(lc).float().cpu().numpy())
        pv.append(torch.sigmoid(lv).float().cpu().numpy())
    return np.concatenate(pa)[:, 0], np.concatenate(pc)[:, 0], np.concatenate(pv)

def vessel_macro(y_ves, p_ves, mask):
    m = mask.astype(bool); aus = []
    for i in range(4):
        yt = y_ves[m, i]
        if 0 < yt.sum() < len(yt): aus.append(roc_auc_score(yt, p_ves[m, i]))
    return float(np.mean(aus)) if aus else float("nan")

best, best_state, bad = -1, None, 0
Xtr, ya, yc, yv, mv = tr
for epoch in range(1, 41):
    model.train()
    for b in batches(Xtr, len(Xtr), 128, True):
        xb = to_in(Xtr[b])
        tb_a = torch.from_numpy(ya[b]).to(DEV); tb_c = torch.from_numpy(yc[b]).to(DEV)
        tb_v = torch.from_numpy(yv[b]).to(DEV);  mb = torch.from_numpy(mv[b]).to(DEV)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            la, lc, lv = model(xb)
            loss = bce(la[:, 0], tb_a) + bce(lc[:, 0], tb_c)
            if mb.sum() > 0:                                       # vessel loss on ACS+ only
                vl = nn.functional.binary_cross_entropy_with_logits(lv, tb_v, reduction="none").mean(1)
                loss = loss + (vl * mb).sum() / mb.sum()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    pa, pc, pv = predict(va)
    au_acs = roc_auc_score(va[1], pa); au_acco = roc_auc_score(va[2], pc)
    vmac = vessel_macro(va[3], pv, va[4])
    combined = 0.5 * au_acco + 0.5 * vmac
    print(f"epoch {epoch:2d}  val ACS {au_acs:.3f}  ACCO {au_acco:.3f}  vessel_macro {vmac:.3f}  combined {combined:.4f}", flush=True)
    wandb.log({"epoch": epoch, "val/acs_auroc": au_acs, "val/acco_auroc": au_acco,
               "val/vessel_macro": vmac, "val/combined": combined, "val/best_combined": max(best, combined)})
    if combined > best:
        best, bad = combined, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 8: print("early stop", flush=True); break

model.load_state_dict(best_state)

# ---------------- gate threshold (val, ~90% specificity for ACCO-vs-rest) ----------------
pa, pc, pv = predict(va)
neg = pc[va[2] == 0]
GATE = float(np.percentile(neg, 90))
print(f"\nGATE (P(ACCO) @ ~90% val spec) = {GATE:.6g}", flush=True)

# ---------------- test eval ----------------
pa, pc, pv = predict(te)
ya_t, yc_t, yv_t, mv_t = te[1], te[2], te[3], te[4]
acco_mask = (yc_t == 1)
t_acs = roc_auc_score(ya_t, pa); t_acco = roc_auc_score(yc_t, pc)
t_vmac = vessel_macro(yv_t, pv, acco_mask.astype(float))
print(f"\n=== TEST (n={len(ya_t)}) ===")
print(f"  ACS detection AUROC  = {t_acs:.2f}")
print(f"  ACCO-vs-rest AUROC   = {t_acco:.2f}")
print(f"  vessel macro (ACCO)  = {t_vmac:.2f}")
summary = {"test/acs_auroc": t_acs, "test/acco_auroc": t_acco, "test/vessel_macro_ACCO": t_vmac,
           "gate": GATE, "n_test": int(len(ya_t))}
for i, v in enumerate(VES):
    yt = yv_t[acco_mask, i]
    if 0 < yt.sum() < len(yt):
        au = roc_auc_score(yt, pv[acco_mask, i])
        print(f"    {v:10s} AUROC = {au:.2f}  (n_pos={int(yt.sum())})")
        summary[f"test/vessel_{v}_ACCO"] = au
fired = (pc > GATE) & acco_mask
if fired.sum() > 0:
    top1 = pv[fired].argmax(1)
    correct = yv_t[fired][np.arange(fired.sum()), top1]
    summary["test/top1_culprit_acc"] = float(correct.mean())
    print(f"  top-1 culprit acc when gate fires = {correct.mean()*100:.1f}%  (n={int(fired.sum())})")
summary["test/gate_fire_rate_nonACCO"] = float((pc[~acco_mask] > GATE).mean())
print(f"  gate fire rate on non-ACCO (1-spec) = {summary['test/gate_fire_rate_nonACCO']*100:.1f}%")
wandb.log(summary); wandb.summary.update(summary)

# ---------------- save bundle + ONNX ----------------
torch.save({"model_state_dict": best_state, "gate": GATE, "vessels": VES, "scale": SCALE},
           f"{OUT}/best_model.pt")
print(f"\nsaved {OUT}/best_model.pt", flush=True)

class Export(nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, ecg_12lead):
        la, lc, lv = self.m(ecg_12lead)
        return torch.sigmoid(la), torch.sigmoid(lc), torch.sigmoid(lv)
try:
    model.eval(); exp = Export(model).to(DEV).eval()
    dummy = torch.randn(1, 12, 2500, device=DEV)
    torch.onnx.export(exp, dummy, f"{OUT}/acs_acco_vessel_3head.onnx",
                      input_names=["ecg_12lead"],
                      output_names=["acs_probability", "acco_probability", "vessel_probability"],
                      dynamic_axes={"ecg_12lead": {0: "B"}, "acs_probability": {0: "B"},
                                    "acco_probability": {0: "B"}, "vessel_probability": {0: "B"}},
                      opset_version=14)
    print(f"saved {OUT}/acs_acco_vessel_3head.onnx", flush=True)
except Exception as e:
    print(f"ONNX export failed ({e}); saving TorchScript fallback", flush=True)
    try:
        ts = torch.jit.trace(Export(model).eval(), torch.randn(1,12,2500,device=DEV))
        ts.save(f"{OUT}/acs_acco_vessel_3head.ts.pt"); print("saved TorchScript", flush=True)
    except Exception as e2:
        print(f"TorchScript also failed ({e2}); .pt bundle is available", flush=True)
try:
    art = wandb.Artifact("acs_3head_v6final", type="model")
    art.add_file(f"{OUT}/best_model.pt")
    if os.path.exists(f"{OUT}/acs_acco_vessel_3head.onnx"): art.add_file(f"{OUT}/acs_acco_vessel_3head.onnx")
    wandb.log_artifact(art)
except Exception as e:
    print(f"wandb artifact log skipped ({e})", flush=True)
wandb.finish()
print("DONE", flush=True)
