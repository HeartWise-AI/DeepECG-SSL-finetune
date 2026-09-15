#!/usr/bin/env python3
"""Beat-level DeepECG-ACS classifier — the predictive model that guides the morph.

Nature 2026 (s41586-026-10674-6): "To retrain the predictive model, we use the same network
architecture and training procedure as those used for the model trained on 10-s ECGs."

So this is the same WCRv2 ECG-transformer 3-head architecture and the same recipe as
scripts/train_acs_3head_softmax.py, refit on individual beats (12, 250) instead of 10-s strips.
This is the piece the earlier beat-morph work was missing: it guided a beat-level morph with a
model that had only ever seen 10-s strips, and then tiled one beat ten times to feed it.

Heads: acs (BCE), acco (BCE), territory softmax over [LEFT, RCA, LCX] (CE, masked to beats from
single-territory complete occlusions).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from fairseq_signals.utils import checkpoint_utils
from fairseq_signals.models.ecg_transformer import ECGTransformerFinetuningModel

A = "/volume/DeepECG-SSL-finetune/data/acs"
BASE = "/volume/DeepECG-SSL-finetune/data/acs_final"
CACHE = os.environ.get("BEAT_CACHE", f"{BASE}/beat_cache_nk")
OUT = os.environ.get("BEAT_OUT", f"{BASE}/checkpoints/beat_3head_nk")
CLASSES = ["LEFT", "RCA", "LCX"]
DEV = "cuda"
BATCH = int(os.environ.get("BEAT_BATCH", 128))
EPOCHS = int(os.environ.get("BEAT_EPOCHS", 25))
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUT, exist_ok=True)


def load(split):
    X = np.load(f"{CACHE}/{split}_beats.npy")               # (N,12,250) float16 mV
    m = pd.read_parquet(f"{CACHE}/{split}_beats.parquet")
    return (X, m.ACS.to_numpy(np.float32), m.ACCO.to_numpy(np.float32),
            m.territory.to_numpy(np.int64), m.single_terr_acco.to_numpy(np.float32),
            m.rec.to_numpy())


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

    def forward(self, source):
        res = ECGTransformerFinetuningModel.forward(self.base, source=source)
        x = res["x"]
        pad = res["padding_mask"]
        x = self.base.final_dropout(x)
        if pad is not None and pad.any():
            x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        f = x.sum(1) / (x != 0).sum(1).clamp(min=1)
        return self.h_acs(f), self.h_acco(f), self.h_ves(f)


def batches(n, bs, shuffle):
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def main():
    tr, va, te = load("train"), load("val"), load("test")
    for nm, d in (("train", tr), ("val", va), ("test", te)):
        print(f"{nm:5s} {d[0].shape[0]:7d} beats  ACCO {d[2].mean():.3f}  "
              f"territory-labelled {int(d[4].sum())}", flush=True)

    base = checkpoint_utils.load_model_and_task(f"{A}/checkpoints/acs/checkpoint_best.pt")[0]
    base = base[0] if isinstance(base, list) else base
    base.num_updates = 10 ** 9
    model = ThreeHeadSoftmax(base).to(DEV)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-3)
    scaler = torch.cuda.amp.GradScaler()
    bce = nn.BCEWithLogitsLoss()

    def to_in(Xb):
        return torch.from_numpy(np.ascontiguousarray(Xb, dtype=np.float32)).to(DEV)

    @torch.no_grad()
    def predict(d, bs=512):
        model.eval()
        pa, pc, pv = [], [], []
        for b in batches(len(d[0]), bs, False):
            with torch.cuda.amp.autocast():
                la, lc, lv = model(to_in(d[0][b]))
            pa.append(torch.sigmoid(la.float())[:, 0].cpu().numpy())
            pc.append(torch.sigmoid(lc.float())[:, 0].cpu().numpy())
            pv.append(torch.softmax(lv.float(), 1).cpu().numpy())
        return np.concatenate(pa), np.concatenate(pc), np.concatenate(pv)

    def macro_ovr(y, p, mask):
        m = mask.astype(bool)
        aus = []
        for i in range(3):
            yt = (y[m] == i).astype(int)
            if 0 < yt.sum() < len(yt):
                aus.append(roc_auc_score(yt, p[m][:, i]))
        return float(np.mean(aus)) if aus else float("nan")

    Xtr, ya, yc, yv, mv, _ = tr
    best, best_state, bad = -1, None, 0
    for ep in range(1, EPOCHS + 1):
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
        vmac = macro_ovr(va[3], pv, va[4])
        m = va[4].astype(bool)
        top1 = float((pv[m].argmax(1) == va[3][m]).mean())
        comb = 0.5 * au_acco + 0.5 * vmac
        print(f"epoch {ep:2d}  val ACS {au_acs:.3f}  ACCO {au_acco:.3f}  "
              f"territory {vmac:.3f}  top1 {top1:.3f}  combined {comb:.4f}", flush=True)
        if comb > best:
            best, bad = comb, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                print("early stop", flush=True)
                break

    model.load_state_dict(best_state)

    # ---- test, at beat level and pooled to the record (mean over a record's beats)
    pa, pc, pv = predict(te)
    Xte, ya_t, yc_t, yv_t, mv_t, rec_t = te
    m = mv_t.astype(bool)
    print(f"\n=== TEST, beat level (n={len(pa)}) ===")
    print(f"  ACS  AUROC = {roc_auc_score(ya_t, pa):.2f}")
    print(f"  ACCO AUROC = {roc_auc_score(yc_t, pc):.2f}")
    aus = []
    for i, c in enumerate(CLASSES):
        yt = (yv_t[m] == i).astype(int)
        aus.append(roc_auc_score(yt, pv[m][:, i]))
        print(f"    {c:5s} AUROC = {aus[-1]:.2f}  (n_pos={int(yt.sum())})")
    print(f"    macro = {np.mean(aus):.2f}   top1 = {(pv[m].argmax(1) == yv_t[m]).mean()*100:.1f}%")

    df = pd.DataFrame({"rec": rec_t, "acs": pa, "acco": pc, "terr": list(pv),
                       "y_acs": ya_t, "y_acco": yc_t, "y_terr": yv_t, "mask": mv_t})
    g = df.groupby("rec")
    rp = pd.DataFrame({"acs": g.acs.mean(), "acco": g.acco.mean(),
                       "y_acs": g.y_acs.first(), "y_acco": g.y_acco.first(),
                       "y_terr": g.y_terr.first(), "mask": g["mask"].first()})
    pv_rec = np.stack(g.terr.apply(lambda s: np.mean(np.stack(s.values), 0)).values)
    print(f"\n=== TEST, beats pooled to the record (n={len(rp)}) ===")
    print(f"  ACS  AUROC = {roc_auc_score(rp.y_acs, rp.acs):.2f}")
    print(f"  ACCO AUROC = {roc_auc_score(rp.y_acco, rp.acco):.2f}")
    mr = rp["mask"].to_numpy().astype(bool)
    aus = []
    for i, c in enumerate(CLASSES):
        yt = (rp.y_terr.to_numpy()[mr] == i).astype(int)
        aus.append(roc_auc_score(yt, pv_rec[mr][:, i]))
        print(f"    {c:5s} AUROC = {aus[-1]:.2f}  (n_pos={int(yt.sum())})")
    print(f"    macro = {np.mean(aus):.2f}   "
          f"top1 = {(pv_rec[mr].argmax(1) == rp.y_terr.to_numpy()[mr]).mean()*100:.1f}%")

    # risk percentiles of REAL beats -- the Nature stopping rule needs these
    stop = {"acco_p90_real_acco_beats": float(np.percentile(pc[yc_t == 1], 90)),
            "acco_p90_all_beats": float(np.percentile(pc, 90))}
    for i, c in enumerate(CLASSES):
        sel = m & (yv_t == i)
        stop[f"terr_{c}_p90_real"] = float(np.percentile(pv[sel][:, i], 90))
    print("\nstopping thresholds (90th percentile of real beats):",
          {k: round(v, 4) for k, v in stop.items()}, flush=True)

    torch.save({"model_state_dict": best_state, "classes": CLASSES, "beat_len": 250,
                "stop_percentiles": stop, "seed": SEED}, f"{OUT}/beat_classifier.pt")
    np.savez(f"{OUT}/test_beat_preds.npz", acs=pa, acco=pc, terr=pv, y_acs=ya_t, y_acco=yc_t,
             y_terr=yv_t, mask=mv_t, rec=rec_t)
    print(f"saved {OUT}/beat_classifier.pt", flush=True)


if __name__ == "__main__":
    main()
