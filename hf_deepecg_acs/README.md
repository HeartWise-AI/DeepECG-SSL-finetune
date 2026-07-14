---
license: other
tags:
- ecg
- cardiology
- acute-coronary-syndrome
- occlusion-mi
library_name: onnxruntime
---

# DeepECG-ACS — Acute Coronary Syndrome + Culprit Vessel (3-head, WCRv2)

Detects **acute coronary syndrome (ACS)** and localizes the **culprit coronary territory**
(LAD, RCA, LCX, Left Main) from a single 12-lead ECG. Shared WCRv2 amplitude-preserved SSL
backbone with **3 heads**: ACS detection, ACCO-vs-rest gate, and culprit-vessel localization.
Trained at the Montreal Heart Institute (MHI) on the v6 cath–ECG cohort (post-exclusion of
CABG / LBBB / paced / graft-only).

## Outputs
| Output | Meaning |
|---|---|
| `acs_probability` | P(acute coronary syndrome) = P(complete ∪ incomplete occlusion) |
| `acco_probability` | P(complete occlusion) — **the gate** for showing a vessel |
| `vessel_probability` | [LAD, RCA, LCX, Left_Main] |

**Deployment rule:** report a culprit vessel only when `acco_probability > 0.0070648`
(ACCO gate, val-fit ~90% specificity); otherwise report no vessel. Culprit = argmax of
`vessel_probability`.

## Input
Raw 12-lead ECG, 250 Hz, 10 s → shape `(2500, 12)` (or `(2500, 12, 1)`), leads
I, II, III, aVR, aVL, aVF, V1–V6. **Scale ADC → millivolts with ×0.00488** for MHI MUSE-GE
(use `1.0` if already in mV). The model is amplitude-preserved — **do not z-score**.

## Held-out test performance (v6-final, n=9,021)
ACS detection AUROC **0.89** · ACCO-gate AUROC **0.92** · vessel macro (ACCO) **0.89**
(LAD 0.93 / RCA 0.95 / LCX 0.81 / LM 0.88) · top-1 culprit accuracy 85.8%.

## Files
- `acs_acco_vessel_3head.onnx` — deployable model (inference; onnxruntime only)
- `best_model.pt` — PyTorch weights (backbone + 3 heads) for fine-tuning
- `inference.py` — ONNX inference (self-contained)
- `model.py` — PyTorch model definition + loader (fine-tuning)
- `finetune.py` — fine-tuning / refining template
- `requirements.txt`

## Quick start — inference (ONNX, no deep-learning framework needed)
```bash
pip install onnxruntime numpy huggingface_hub
huggingface-cli download heartwise/deepecg-acs-3head acs_acco_vessel_3head.onnx inference.py --local-dir .
python inference.py --ecg your_ecg.npy            # raw ADC; --scale 1.0 if already mV
```
```python
from huggingface_hub import hf_hub_download
import numpy as np, onnxruntime as ort
onnx = hf_hub_download("heartwise/deepecg-acs-3head", "acs_acco_vessel_3head.onnx")
sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
x = np.nan_to_num(np.load("your_ecg.npy").squeeze()) * 0.00488   # ADC -> mV
x = (x.T if x.shape == (2500, 12) else x)[None].astype("float32")  # (1,12,2500)
acs, acco, vessel = sess.run(None, {"ecg_12lead": x})
VESSELS = ["LAD", "RCA", "LCX", "Left_Main"]
culprit = VESSELS[int(vessel[0].argmax())] if acco[0,0] > 0.0070648 else None
print(dict(acs=float(acs[0,0]), acco=float(acco[0,0]), culprit=culprit))
```

## Fine-tuning / refining (PyTorch)
Needs the fairseq-signals framework and a WCRv2 backbone checkpoint (architecture donor).
```bash
pip install torch
pip install --editable git+https://github.com/HeartWise-AI/DeepECG-SSL-finetune.git#egg=fairseq_signals
huggingface-cli download heartwise/deepecg-acs-3head best_model.pt model.py finetune.py --local-dir .
# obtain a WCRv2 backbone checkpoint (see DeepECG-SSL-finetune), then:
python finetune.py    # edit WCRV2 path + your arrays inside
```
```python
from model import load_model
model, meta = load_model("best_model.pt", "/path/to/wcrv2_backbone.pt", device="cuda")
# model(source) -> (acs_logit, acco_logit, vessel_logits); continue training as usual
```

## Intended use & limitations
Research use. The **ACS head** is a triage aid for suspected ACS / chest pain; the
**culprit-vessel head** should only be interpreted once ACS is clinically confirmed (and the
ACCO gate fires). Single-center (MHI) retrospective; no external/prospective validation.
Not a substitute for clinical assessment or angiography.

**License:** research/internal use, Montreal Heart Institute. Do not redistribute patient data.
