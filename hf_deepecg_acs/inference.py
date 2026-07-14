#!/usr/bin/env python3
"""DeepECG-ACS 3-head — ONNX inference (self-contained: onnxruntime + numpy only).

Outputs, from a single 12-lead ECG:
  acs_probability   - P(acute coronary syndrome; ACCO or AICO)
  acco_probability  - P(complete occlusion) -> the gate for showing a vessel
  vessel_probability- [LAD, RCA, LCX, Left_Main]
Rule: report a culprit vessel only when acco_probability > GATE.

Usage:
  python inference.py --ecg path/to/ecg.npy [--scale 0.00488]
"""
import argparse, numpy as np, onnxruntime as ort

GATE = 0.0070648            # ACCO gate fit on validation (~90% specificity)
VESSELS = ["LAD", "RCA", "LCX", "Left_Main"]

def predict(onnx_path, ecg, scale=0.00488):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    x = np.nan_to_num(np.asarray(ecg).squeeze()).astype(np.float32) * scale   # ADC -> mV (MHI MUSE = 0.00488)
    x = x.T if x.shape == (2500, 12) else x                                   # want (12, 2500)
    x = x[None].astype(np.float32)                                            # (1, 12, 2500)
    acs, acco, vessel = sess.run(None, {"ecg_12lead": x})
    culprit = VESSELS[int(vessel[0].argmax())] if acco[0, 0] > GATE else None
    return {"acs_probability": float(acs[0, 0]),
            "acco_probability": float(acco[0, 0]),
            "vessel_probability": dict(zip(VESSELS, vessel[0].round(4).tolist())),
            "culprit_vessel": culprit}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="acs_acco_vessel_3head.onnx")
    ap.add_argument("--ecg", required=True, help="ECG .npy, shape (2500,12) or (2500,12,1), raw ADC units")
    ap.add_argument("--scale", type=float, default=0.00488, help="ADC->mV factor (MHI MUSE 0.00488; use 1.0 if already mV)")
    a = ap.parse_args()
    import json; print(json.dumps(predict(a.onnx, np.load(a.ecg), a.scale), indent=2))
