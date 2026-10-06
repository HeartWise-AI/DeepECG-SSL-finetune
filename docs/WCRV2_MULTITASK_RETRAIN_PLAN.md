# WCR v2 retrain: EchoNext-12 and incident AF, with age and sex inputs

Context: HeartWise executive meeting, 2026-10-05. Three retraining threads were open on the
DeepECG-SSL v2 backbone (WCR v2, amplitude-preserved): EchoNext with all labels plus age and sex,
incident AF at 5 years on WCR v2, and low EF. They were first merged into one 16-head multi-task
model, then, on request, delivered as two separate models. This document records what was decided,
how the data were built, the results, and how to reproduce or use the models.

## 1. Findings that shaped the work

- **No WCR v2 incident-AF model existed.** The deployed `wcr_afib_5y.pt` was trained on the WCR v1
  encoder (checkpoint config), with binary focal loss and one label.
- **EchoNext on WCR v2 stopped at 4 labels** (Notion model card) and 7 labels (v6 bundle). The public
  EchoNext set has 12 binary labels (11 components + composite).
- **MHI echo labels must come from the v1.6 `echonext_*` columns.** v3 to v5 used another label file
  and MHI aortic stenosis collapsed to AUROC 0.50; v6 fixed it by switching to v1.6.
- **The repository had no auxiliary inputs and no missing-label support.** Added here.
- **Full fine-tuning peaks within 1 to 2 epochs** at lr 1e-5 and then overfits; validating every
  1 000 updates at lr 5e-6 found a better optimum for AF.

## 2. Delivered models

| Model | NAS folder | Labels (output order) | Selection |
|---|---|---|---|
| EchoNext-12 | `/media/data1/models/DeepECG-SSL/DeepECG_SSL_WCRv2_EchoNext12_v1/` | `lvef_lte_45, lvwt_gte_13, as_mod_plus, ar_mod_plus, mr_mod_plus, tr_mod_plus, pr_mod_plus, rv_dysf_mod_plus, peric_eff_mod_large, pasp_gte_45, tr_max_gte_32, shd_composite` | best epoch 2 (val macro AUROC 0.816) |
| Incident AF | `/media/data1/models/DeepECG-SSL/DeepECG_SSL_WCRv2_AFib_v2/` | `incident_af_2y, incident_af_5y` | lr 5e-6, best at update 6 000 (val 5 y AUROC 0.758) |

Each folder holds `checkpoint_best.pt`, `labels.json` (label order, age mean and std), the training
config, `eval/` (metrics, thresholds, paired baselines), `train.log`, `infer_wcrv2_aux.py` and a README.
Notion model cards (Models database) document inputs, outputs and inference for both.

### EchoNext-12 test AUROC (paired with v6 on identical rows)

| Label | EchoNext test (n 5 442) | v6 | MHI test (n ≈ 54 400) | v6 |
|---|---|---|---|---|
| lvef_lte_45 | 0.918 | 0.914 | 0.890 | 0.884 |
| as_mod_plus | 0.861 | 0.847 | 0.822 | 0.812 |
| ar_mod_plus | 0.773 | 0.781 | 0.804 | 0.796 |
| mr_mod_plus | 0.850 | 0.844 | 0.812 | 0.799 |
| tr_mod_plus | 0.866 | 0.854 | 0.821 | 0.808 |
| rv_dysf_mod_plus | 0.902 | 0.891 | 0.878 | 0.883 |
| pr_mod_plus | 0.853 | n/a | 0.825 | n/a |
| peric_eff_mod_large | 0.726 | n/a | 0.775 | n/a |
| pasp_gte_45 | 0.811 | n/a | 0.853 | n/a |
| tr_max_gte_32 | 0.802 | n/a | not labelled | n/a |
| lvwt_gte_13 | 0.785 | n/a | 0.765 | n/a |
| **shd_composite (12-label)** | **0.852** | 0.829 | **0.763** | 0.726 |

### Incident AF test AUROC (MHI test, n 130 256)

| Head | This model | Deployed WCR v1 `wcr_afib_5y`, same rows | iAF5 test rows (n 19 083) |
|---|---|---|---|
| incident_af_5y | **0.758** | 0.744 | 0.742 |
| incident_af_2y | 0.756 | n/a | 0.745 |

Thresholds in `eval/` are Youden and sensitivity-0.90 points on the **test** set and are for
reference only; set operating points on the validation set before deployment.

## 3. Data

Built by `scripts/preprocess/ecg/build_wcrv2_multitask_dataset.py`, then
`subset_wcrv2_dataset.py` for the EchoNext-12 subset. Outputs live on the training container under
`data/` (git-ignored): `{split}_x.npy` (N, 2500, 12, 1) float32 mV, `{split}_y.npy` (NaN = label not
available), `{split}_aux.npy` `[age_z, sex_male]` (NaN = unknown), row-aligned `{split}_meta.parquet`,
and fairseq manifests with an `aux_path:` line.

- **EchoNext** (PhysioNet 100K): the 12 flags, `age_at_ecg`, `sex`; Columbia splits; waveforms ×
  0.162889 to reach mV.
- **MHI v1.6**: `echonext_*` labels for ECGs within 365 days before the echo; `tr_max_gte_32` is
  masked (never populated in MHI); v1.6 `Split` (patient-level, no patient in two splits).
- **Incident AF** (EHJ 2024 definition, Notion *ECG-AI AF*): sinus-rhythm ECGs, prevalent AF excluded
  (`afib_dt_minaf` NaN or > 0), positives with first AF within the horizon, negatives with no AF
  recorded (`--af-negatives any`). 660 764 ECGs (paper: 669 782), 5-year prevalence 10.0 %. The
  DeepECG-PNG iAF5 cohort is a subset; its test rows are reported separately.
- **Age**: v1.6 metadata has age or date of birth on 18 % of ECGs. Pooling date of birth per patient
  across the v1.6 metadata and the DeepECG-PNG downstream parquets, then `afib_Age`, reaches 96.4 %
  of labelled MHI ECGs; about 9 600 patients (mostly 2008 to 2010) have no age anywhere on the NAS.
  Sex is 100 %. Unknown age is passed as NaN with a missingness flag.
- **MHI waveforms**: train from the per-ECG `waveform_path_psa` files (already mV); val and test from
  the consolidated raw-ADC arrays `X_val_v1.1.npy` / `X_test_v1.2.npy` × 0.00488, whose row order was
  verified against the per-ECG files. The train array is shuffled, so it could not be used.
- **Gotchas**: `afib_dt_*` are timedeltas stored as int64 nanoseconds with NaT = −9.22e18.

| Dataset | Train | Val | Test |
|---|---|---|---|
| EchoNext-12 (EchoNext + MHI) | 267 119 | 33 149 | 60 605 |
| Incident AF (MHI) | 462 846 | 67 662 | 130 256 |

## 4. Code

| File | Purpose |
|---|---|
| `fairseq_signals/criterions/masked_binary_cross_entropy_with_logits.py` | `criterion._name=masked_bce`: NaN targets excluded from loss and metrics; logs macro and per-label AUROC |
| `fairseq_signals/models/classification/ecg_transformer_aux_classifier.py` | `model._name=ecg_transformer_aux_classifier`: pooled WCR embedding ⊕ MLP over `[age_z, sex, missing flags]` |
| `fairseq_signals/data/ecg/raw_ecg_dataset.py` | `NpECGDataset` reads `aux_path:`; the collator passes `aux` in `net_input` |
| `fairseq_signals/utils/checkpoint_utils.py` | `weights_only=False` for torch ≥ 2.6; fix per-update pruning calling `suffix()` on a string |
| `scripts/preprocess/ecg/build_wcrv2_multitask_dataset.py` | dataset builder (`--only-labels`, `--af-negatives`, `--mhi-array-splits`, `--reuse-from`, `--refresh-aux`, `--dry-run`) |
| `scripts/preprocess/ecg/subset_wcrv2_dataset.py`, `merge_wcrv2_multitask_dirs.py` | carve label/row subsets; merge split folders |
| `examples/.../diagnosis_wcrv2_{echonext12,afib,multitask}.yaml` | training configs |
| `scripts/train_wcrv2_multitask.sh`, `scripts/launch_when_gpu_free.sh` | launchers |
| `scripts/eval_wcrv2_multitask.py` | per-label metrics, bootstrap CI, thresholds, paired baselines (v6, WCR v1 AF), iAF5 rows |
| `scripts/infer_wcrv2_aux.py` | standalone inference (waveform mV + age + sex); matches `fairseq-hydra-inference` to 0.0016 |
| `scripts/tests/smoke_wcrv2_multitask.py` | 2-minute end-to-end train + inference check on synthetic data |

## 5. Environment

On a Python 3.12 container fairseq-signals (`omegaconf<2.1`, `hydra-core<1.1`) does not import. Use a
Python 3.9 venv and run modules with the repo on `PYTHONPATH`:

```bash
uv venv --python 3.9 /volume/venvs/fss39
uv pip install --python /volume/venvs/fss39/bin/python torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /volume/venvs/fss39/bin/python omegaconf==2.0.6 hydra-core==1.0.7 numpy==1.26.4 \
    pandas scipy wfdb pyarrow scikit-learn wandb cython transformers==4.30.2 sacrebleu bitarray regex tqdm
# build only the Cython batching extension (the C++ extensions trip on CUDA 13 vs torch cu128)
export PYTHONPATH=/volume/DeepECG-SSL-finetune
/volume/venvs/fss39/bin/python -m fairseq_cli.hydra_train ...
```

`/volume` is container-local; copy models to `/media/data1/models/DeepECG-SSL/`.

## 6. Reproduce

```bash
cd /volume/DeepECG-SSL-finetune
# merged dataset (EchoNext + MHI), val/test from the consolidated arrays
python3 scripts/preprocess/ecg/build_wcrv2_multitask_dataset.py --out data/wcrv2_multitask --mhi-array-splits val test
# EchoNext-12 subset
python3 scripts/preprocess/ecg/subset_wcrv2_dataset.py --src data/wcrv2_multitask --out data/wcrv2_echonext12 \
    --labels lvef_lte_45 lvwt_gte_13 as_mod_plus ar_mod_plus mr_mod_plus tr_mod_plus pr_mod_plus \
             rv_dysf_mod_plus peric_eff_mod_large pasp_gte_45 tr_max_gte_32 shd_composite
# incident-AF dataset (EHJ definition), reusing local waveforms
python3 scripts/preprocess/ecg/build_wcrv2_multitask_dataset.py --out data/wcrv2_afib --no-echonext \
    --only-labels incident_af_2y incident_af_5y --af-negatives any --mhi-array-splits val test \
    --reuse-from data/wcrv2_multitask

# train
CUDA_VISIBLE_DEVICES=0 RUN=wcrv2_echonext12_v1 CONFIG=diagnosis_wcrv2_echonext12 DATA=$PWD/data/wcrv2_echonext12 \
    scripts/train_wcrv2_multitask.sh
CUDA_VISIBLE_DEVICES=1 RUN=wcrv2_afib_v2 CONFIG=diagnosis_wcrv2_afib DATA=$PWD/data/wcrv2_afib LR=5.0e-6 \
    scripts/train_wcrv2_multitask.sh checkpoint.save_interval_updates=1000 dataset.validate_interval_updates=1000 \
    checkpoint.patience=10 checkpoint.keep_interval_updates=1

# evaluate (with paired baselines)
PYTHONPATH=$PWD /volume/venvs/fss39/bin/python scripts/eval_wcrv2_multitask.py \
    --ckpt data/runs/wcrv2_echonext12_v1/checkpoints/checkpoint_best.pt --data data/wcrv2_echonext12 \
    --subsets echonext_test mhi_test --baselines
```

## 7. Inference

```python
import sys; sys.path.insert(0, "/volume/DeepECG-SSL-finetune/scripts")
from infer_wcrv2_aux import load, predict
m = load("/media/data1/models/DeepECG-SSL/DeepECG_SSL_WCRv2_EchoNext12_v1", device="cuda:0")
probs = predict(m, ecg_mV, age_years, sex_male)   # (N, 12); columns = m["labels"]; NaN age/sex allowed
```

Waveforms: 12 × 2500 at 250 Hz in mV (MHI raw ADC × 0.00488). Always pass age and sex: leaving them
unknown shifts probabilities by about 0.009 on average.

## 8. Open items

- Operating thresholds on the validation sets before deployment.
- DeepECG Docker entry and model-policy exposure of the 5-year AF head (replaces `wcr_afib_5y` for the
  collaborating cardiologist).
- Private HuggingFace mirror next to `heartwise/DeepECG-EchoNext`.
- The 16-head multi-task run (one epoch, val macro AUROC 0.827, LVEF ≤ 40 0.947) is kept on the
  training container as a starting point for a low-EF model.
