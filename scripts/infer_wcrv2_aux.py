#!/usr/bin/env python3
"""Programmatic inference for the WCR v2 aux-input classifiers (EchoNext-12, incident AF).

The models take the 12-lead waveform in millivolts AND two auxiliary scalars per ECG:
age (years) and sex (1 = male, 0 = female). Age is z-scored with the mean/std stored in
labels.json next to the checkpoint; unknown age or sex is passed as NaN (the model has a
missingness flag). Waveforms: 250 Hz, 10 s, shape (N, 2500, 12) or (N, 12, 2500), in mV.
MHI MUSE raw ADC -> mV: x 0.00488. EchoNext PhysioNet pre-normalised arrays -> mV: x 0.162889.

Usage
  PYTHONPATH=/volume/DeepECG-SSL-finetune /volume/venvs/fss39/bin/python scripts/infer_wcrv2_aux.py \
      --model-dir /media/data1/models/DeepECG-SSL/DeepECG_SSL_WCRv2_EchoNext12_v1 \
      --ecg ecgs_mV.npy --age ages.npy --sex sex.npy --out probs.csv --device 0

Python
  from scripts.infer_wcrv2_aux import load, predict
  m = load("/media/data1/models/DeepECG-SSL/DeepECG_SSL_WCRv2_EchoNext12_v1", device="cuda:0")
  probs = predict(m, ecg_mV, age_years, sex_male)   # (N, K) numpy, columns = m["labels"]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load(model_dir, device="cuda:0", ckpt_name="checkpoint_best.pt"):
    """Load checkpoint + labels.json. Returns a dict with model, labels, aux normalisation, device."""
    from fairseq_signals import tasks
    from fairseq_signals.utils import checkpoint_utils

    model_dir = Path(model_dir)
    info = json.load(open(model_dir / "labels.json"))
    state = checkpoint_utils.load_checkpoint_to_cpu(str(model_dir / ckpt_name))
    cfg = state["cfg"]
    task = tasks.setup_task(cfg["task"])
    model = task.build_model(cfg["model"])
    model.load_state_dict(state["model"], strict=True)
    model = model.to(device).eval()
    if cfg["common"].get("fp16", False) and str(device).startswith("cuda"):
        model = model.half()
    return {"model": model, "labels": info["labels"], "aux_norm": info["aux_norm"], "device": device,
            "aux_dim": int(cfg["model"].get("aux_dim", 2)), "half": next(model.parameters()).dtype == torch.float16}


def _prep_ecg(ecg):
    x = np.asarray(ecg, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    if x.ndim == 4:  # (N, 2500, 12, 1)
        x = x[..., 0]
    if x.shape[1] == 2500 and x.shape[2] == 12:
        x = np.transpose(x, (0, 2, 1))  # -> (N, 12, 2500)
    assert x.shape[1:] == (12, 2500), f"expected (N, 12, 2500) or (N, 2500, 12), got {x.shape}"
    return np.nan_to_num(x, nan=0.0)


@torch.no_grad()
def predict(m, ecg_mV, age_years=None, sex_male=None, batch_size=256):
    """Return sigmoid probabilities (N, K). age_years / sex_male: arrays of length N, NaN = unknown."""
    x = _prep_ecg(ecg_mV)
    n = len(x)
    age = np.full(n, np.nan, dtype=np.float32) if age_years is None else np.asarray(age_years, dtype=np.float32)
    sex = np.full(n, np.nan, dtype=np.float32) if sex_male is None else np.asarray(sex_male, dtype=np.float32)
    age_z = (age - m["aux_norm"]["age_mean"]) / m["aux_norm"]["age_std"]
    aux = np.stack([age_z, sex], axis=1)
    out = []
    for i0 in range(0, n, batch_size):
        xb = torch.from_numpy(x[i0:i0 + batch_size]).to(m["device"])
        ab = torch.from_numpy(aux[i0:i0 + batch_size]).to(m["device"])
        if m["half"]:
            xb, ab = xb.half(), ab.half()
        logits = m["model"](source=xb, padding_mask=None, aux=ab)["out"].float()
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--ecg", required=True, help=".npy with waveforms in mV, (N, 2500, 12) or (N, 12, 2500)")
    ap.add_argument("--age", default=None, help=".npy of ages in years (NaN = unknown)")
    ap.add_argument("--sex", default=None, help=".npy of sex (1 = male, 0 = female, NaN = unknown)")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply waveform by this to reach mV (MHI raw ADC: 0.00488)")
    ap.add_argument("--out", default="probs.csv")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    m = load(args.model_dir, device=args.device)
    ecg = np.load(args.ecg, mmap_mode="r")
    ecg = np.asarray(ecg, dtype=np.float32) * args.scale
    age = np.load(args.age) if args.age else None
    sex = np.load(args.sex) if args.sex else None
    if age is None or sex is None:
        print("warning: age and/or sex not provided; the model was trained with them and expects them", file=sys.stderr)
    probs = predict(m, ecg, age, sex)
    import pandas as pd
    pd.DataFrame(probs, columns=m["labels"]).to_csv(args.out, index=False)
    print(f"wrote {args.out}  shape={probs.shape}  labels={m['labels']}")


if __name__ == "__main__":
    main()
