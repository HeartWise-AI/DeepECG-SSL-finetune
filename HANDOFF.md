# Handoff: WCR v2 EchoNext-12 and incident-AF models

This handoff is for whoever picks up the DeepECG-SSL WCR v2 fine-tunes next. Its last update was
2026-11-15. The full record is
[`docs/WCRV2_MULTITASK_RETRAIN_PLAN.md`](docs/WCRV2_MULTITASK_RETRAIN_PLAN.md), which covers data
building, results, reproduction and inference. This file lists what is finished, what is still open,
and the traps that cost time.

## State

All of this work is merged on `main`. No open branches or running jobs belong to it.

| PR | Merged | What |
|---|---|---|
| #6 (`d999d19`) | 2026-10-06 | Two WCR v2 fine-tunes with age/sex inputs, NaN-masked BCE, dataset builders, eval with paired baselines, standalone inference, smoke test |
| #8 (`bb39014`) | 2026-10-14 | ONNX export for the auxiliary-input models (`scripts/export_wcrv2_aux_onnx.py`); only the AF model has been exported |
| #9 (`980e705`) | 2026-10-27 | Config and checkpoint loading works on newer Python/omegaconf (3.11+, omegaconf ≥ 2.1) and still on the 3.9 venv; BCE macro-AUC logging fixed |

## Delivered models (NAS, safe)

| Model | Folder under `/media/data1/models/DeepECG-SSL/` | Test AUROC | Notion card |
|---|---|---|---|
| EchoNext-12 (12 labels, age + sex) | `DeepECG_SSL_WCRv2_EchoNext12_v1/` | SHD composite 0.852 EchoNext / 0.763 MHI (v6: 0.829 / 0.726); LVEF ≤ 45 0.918 | [link](https://app.notion.com/p/3f10855e87b6810bb16ee25dc90ea3eb) |
| Incident AF 2 y + 5 y (age + sex) | `DeepECG_SSL_WCRv2_AFib_v2/` (includes ONNX) | 5 y 0.758 vs deployed WCR v1 `wcr_afib_5y` 0.744 on the same 130K MHI rows | [link](https://app.notion.com/p/3f10855e87b68115905be274af70088e) |

Each folder holds `checkpoint_best.pt`, `labels.json` (label order, age mean/std), configs, `eval/`,
`train.log`, `infer_wcrv2_aux.py` and a README. `DeepECG_SSL_WCRv2_AFib_v1` is superseded by v2.

## Open items, roughly in priority order

1. **Operating thresholds.** The thresholds in `eval/` were picked on the **test** set and are for
   reference only. Before either model is deployed, pick its thresholds on the validation set.
2. **Deployment of the AF model.** It needs a DeepECG Docker / PACS-AI entry, and model-policy
   exposure of the 5-year head so it replaces `wcr_afib_5y` (WCR v1) for the collaborating
   cardiologist. Use the `model-publishing` skill, then `pacs-ai-model`.
3. **Hugging Face.** Add a private mirror next to `heartwise/DeepECG-EchoNext`.
4. **ONNX for EchoNext-12.** Run `scripts/export_wcrv2_aux_onnx.py` on the EchoNext-12 folder, the
   same way the AF model was exported.
5. **Low-EF model.** The 16-head multi-task run `data/runs/wcrv2_mt_v1` (1 epoch, val macro AUROC
   0.827, LVEF ≤ 40 0.947) is a starting point. It exists **only on container-local `/volume`**, so
   copy it to the NAS if you want to keep it.

## Settled, do not re-derive

- **No WCR v2 incident-AF model existed before this work.** The deployed `wcr_afib_5y.pt` uses the
  WCR v1 encoder, as its checkpoint config shows.
- **Full fine-tuning peaks within 1–2 epochs** at lr 1e-5 and then overfits. Use lr 5e-6 and
  validate every 1 000 updates; this is how AF v2 beat v1.
- **MHI echo labels come from the v1.6 `echonext_*` columns.** Other label files collapse MHI aortic
  stenosis to AUROC 0.50.
- **The AF cohort follows the EHJ 2024 definition:** sinus ECGs, no prevalent AF, and no minimum
  follow-up for negatives (660,764 MHI ECGs, 5 y prevalence 10 %). The Notion iAF5 cohort (96K) is a
  subset of it, and its rows are scored separately in the eval.
- **Always pass age and sex at inference.** Leaving them unknown shifts probabilities by about 0.009
  on average.

## Traps

- **`/volume` is local NVMe and dies with the container.** That includes the datasets under `data/`
  (`wcrv2_multitask` 100 GB, `wcrv2_afib` 89 GB, `wcrv2_echonext12` 48 GB) and `data/runs/`. They can
  be rebuilt with the commands in section 6 of the plan doc. Models that matter are already on the NAS.
- **Use the Python 3.9 venv to train.** System Python 3.12 does not run fairseq-signals training with
  omegaconf 2.0.x / hydra 1.0.x. Use
  `PYTHONPATH=/volume/DeepECG-SSL-finetune /volume/venvs/fss39/bin/python -m fairseq_cli.hydra_train ...`.
  Console scripts are not installed. PR #9 makes loading and inference work on newer Python as well.
- **`default_factory` in `fairseq_signals/dataclass/configs.py`** works with omegaconf 2.0.6 only
  because `hydra_init` registers `default_factory()`. Keep that pairing if you touch either one.
- **MHI v1.6 label quirks:**
  - AF time deltas are int64 nanoseconds (divide by 8.64e13 for days).
  - `echonext_tr_max_gte_32` is never populated for MHI, so it is masked.
  - `PatientAge` covers only 18.5 % of ECGs. The builder pools DOB per patient to reach about 96 %.
- **Small-file reads from the NAS are slow** (about 30–70 files/s, shared with other users). The
  builder reads val/test from the consolidated arrays, which is about 10× faster.
- **There are no git credentials in the training container.** Pushes for #6 and #9 went through the
  GitHub API (MCP `push_files`), so local branches can look "modified" relative to `main` even when
  they match it. Compare blobs before you discard anything.

## Quick checks

```bash
cd /volume/DeepECG-SSL-finetune
# 2-minute end-to-end train + inference on synthetic data (needs a GPU and the WCR v2 encoder)
PYTHONPATH=$PWD /volume/venvs/fss39/bin/python scripts/tests/smoke_wcrv2_multitask.py --device 0
```

```python
import sys; sys.path.insert(0, "/volume/DeepECG-SSL-finetune/scripts")
from infer_wcrv2_aux import load, predict
m = load("/media/data1/models/DeepECG-SSL/DeepECG_SSL_WCRv2_AFib_v2", device="cuda:0")
probs = predict(m, ecg_mV, age_years, sex_male)   # ecg_mV: (N, 12, 2500) or (N, 2500, 12), 250 Hz; columns = m["labels"]
```
