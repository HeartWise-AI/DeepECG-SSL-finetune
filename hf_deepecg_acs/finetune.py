#!/usr/bin/env python3
"""DeepECG-ACS 3-head — fine-tuning / refining template.

Continue training the 3-head model on your own ECG cohort. Provide arrays:
  X_{train,val}.npy  (N, 2500, 12) raw ADC   +   labels for ACS / ACCO / vessel.
Joint masked BCE (ACS + ACCO on all rows; vessel loss on ACS+/culprit rows only).
"""
import numpy as np, torch, torch.nn as nn
from model import load_model

# --- edit these paths ---
BEST = "best_model.pt"                       # this repo
WCRV2 = "/path/to/wcrv2_backbone.pt"         # WCRv2 ecg_transformer_classifier ckpt (architecture donor)
SCALE = 0.00488                              # ADC->mV (1.0 if your arrays are already mV)
DEV = "cuda"

model, meta = load_model(BEST, WCRV2, DEV)   # starts from the released weights
opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-3)
scaler = torch.cuda.amp.GradScaler()
bce = nn.BCEWithLogitsLoss()

def to_in(Xb):                               # (B,2500,12) ADC -> (B,12,2500) mV
    return torch.from_numpy(np.transpose(Xb.astype(np.float32) * SCALE, (0, 2, 1))).to(DEV)

# X: (N,2500,12); y_acs,y_acco: (N,); y_ves: (N,4); m_ves: (N,) 1 where vessel is supervised
X = np.load("X_train.npy"); y_acs = np.load("y_acs.npy"); y_acco = np.load("y_acco.npy")
y_ves = np.load("y_vessel.npy"); m_ves = np.load("m_vessel.npy")

for epoch in range(20):
    model.train()
    idx = np.random.permutation(len(X))
    for i in range(0, len(X), 128):
        b = idx[i:i+128]
        xb = to_in(X[b])
        ta = torch.from_numpy(y_acs[b].astype("f4")).to(DEV)
        tc = torch.from_numpy(y_acco[b].astype("f4")).to(DEV)
        tv = torch.from_numpy(y_ves[b].astype("f4")).to(DEV)
        mb = torch.from_numpy(m_ves[b].astype("f4")).to(DEV)
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            la, lc, lv = model(xb)
            loss = bce(la[:, 0], ta) + bce(lc[:, 0], tc)
            if mb.sum() > 0:
                vl = nn.functional.binary_cross_entropy_with_logits(lv, tv, reduction="none").mean(1)
                loss = loss + (vl * mb).sum() / mb.sum()
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    print(f"epoch {epoch} done")

torch.save({"model_state_dict": model.state_dict(), "gate": meta["gate"],
            "vessels": meta["vessels"], "scale": SCALE}, "best_model_refined.pt")
