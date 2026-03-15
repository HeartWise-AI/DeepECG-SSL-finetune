#!/bin/bash
# =============================================================================
# Retrain DeepECG-SSL Without Amplitude-Destroying Normalization
#
# Uses fixed per-source scale factors (ADC -> mV) instead of z-score or
# per-sample spectral normalization, preserving inter-patient amplitude
# differences critical for voltage-dependent diagnoses (LVH, enlargement).
#
# Scale factors:
#   MHI:    0.00488  (MUSE GE ADC gain, verified)
#   MIMIC:  0.001    (dataset documentation, verified)
#   CODE15: 0.4694   (estimated via calibration tool, see Step 0)
#
# For unknown datasets, use:
#   python scripts/preprocess/ecg/calibrate_dataset_scale.py /path/to/data.npy
# =============================================================================

set -e

# Base paths
REPO_DIR="/volume/fairseq-signals"
DATA_DIR="/volume/fairseq-signals/data/ssl-amp-preserved"
PREPROCESS_SCRIPT="${REPO_DIR}/fairseq_signals/data/ecg/preprocess/preprocess_parquet.py"
MANIFEST_SCRIPT="${REPO_DIR}/scripts/preprocess/ecg/create_flat_manifest.py"
CMSC_CONVERT_SCRIPT="${REPO_DIR}/fairseq_signals/data/ecg/preprocess/convert_to_cmsc_manifest.py"
CALIBRATE_SCRIPT="${REPO_DIR}/scripts/preprocess/ecg/calibrate_dataset_scale.py"

# Source data paths
MHI_NPY_DIR="/media/data1/anolin/temp_new_dataset/ecg_npy/"
MHI_PARQUET="/media/data1/muse_ge/ECG_ad20241231_cat_labels_with_split.parquet"
MIMIC_NPY="/media/data1/achilsowa/datasets/npy/X_mimic.npy"
MIMIC_PARQUET="/media/data1/achilsowa/mimic/mimic_labelbox_review_completed_v3.parquet"
CODE15_NPY="/media/data1/achilsowa/code15/X_code15.npy"
CODE15_CSV="/media/data1/achilsowa/code15/exams.csv"

# Scale factors (ADC -> mV)
MHI_SCALE=0.00488     # MUSE GE documented gain
MIMIC_SCALE=0.001     # MIMIC dataset documentation
CODE15_SCALE=0.4694   # Estimated via calibration (original plan assumed 1.0)

# Preprocessing output paths
MHI_DEST="${DATA_DIR}/mhi"
MIMIC_DEST="${DATA_DIR}/mimic"
CODE15_DEST="${DATA_DIR}/code15"
MANIFEST_DIR="${DATA_DIR}/manifest"

# =============================================================================
# Step 0: Verify scale factors with calibration tool (optional)
# =============================================================================
echo "=== Step 0: Verifying scale factors ==="
python "${CALIBRATE_SCRIPT}" "${MHI_NPY_DIR}" --n-samples 2000 --known-scale ${MHI_SCALE}
echo ""
python "${CALIBRATE_SCRIPT}" "${MIMIC_NPY}" --n-samples 2000 --known-scale ${MIMIC_SCALE}
echo ""
python "${CALIBRATE_SCRIPT}" "${CODE15_NPY}" --n-samples 2000 --known-scale ${CODE15_SCALE}
echo ""

# =============================================================================
# Step 1: Preprocess MHI data (individual NPY files)
# =============================================================================
echo "=== Step 1: Preprocessing MHI data (scale=${MHI_SCALE}) ==="
python "${PREPROCESS_SCRIPT}" \
    "${MHI_NPY_DIR}" \
    --x-path "${MHI_PARQUET}" \
    --dest "${MHI_DEST}" \
    --scale ${MHI_SCALE} \
    --sample-rate 250 --sec 5 \
    --min-ecg-by-patient 2 \
    --patient-id-col new_PatientID \
    --fname-col npy_path \
    --workers 16 --seed 42

# =============================================================================
# Step 2: Preprocess MIMIC data (consolidated NPY)
# =============================================================================
echo "=== Step 2a: Creating MIMIC CSV for preprocessing ==="
python "${REPO_DIR}/scripts/preprocess/ecg/create_mimic_csv_for_preprocess.py" \
    --parquet-path "${MIMIC_PARQUET}" \
    --output-path "${DATA_DIR}/mimic_for_preprocess.csv"

echo "=== Step 2b: Preprocessing MIMIC data (scale=${MIMIC_SCALE}, resample 500->250 Hz) ==="
python "${PREPROCESS_SCRIPT}" \
    . \
    --x-path "${DATA_DIR}/mimic_for_preprocess.csv" \
    --npy-path "${MIMIC_NPY}" \
    --dest "${MIMIC_DEST}" \
    --scale ${MIMIC_SCALE} \
    --source-sample-rate 500 \
    --sample-rate 250 --sec 5 \
    --min-ecg-by-patient 2 \
    --patient-id-col new_PatientID \
    --fname-col preprocess_fname \
    --workers 16 --seed 42

# =============================================================================
# Step 3: Preprocess CODE15 data (consolidated NPY)
# =============================================================================
echo "=== Step 3a: Creating CODE15 CSV for preprocessing ==="
python "${REPO_DIR}/scripts/preprocess/ecg/create_code15_csv_for_preprocess.py" \
    --csv-path "${CODE15_CSV}" \
    --output-path "${DATA_DIR}/code15_for_preprocess.csv"

echo "=== Step 3b: Preprocessing CODE15 data (scale=${CODE15_SCALE}) ==="
python "${PREPROCESS_SCRIPT}" \
    . \
    --x-path "${DATA_DIR}/code15_for_preprocess.csv" \
    --npy-path "${CODE15_NPY}" \
    --dest "${CODE15_DEST}" \
    --scale ${CODE15_SCALE} \
    --sample-rate 250 --sec 5 \
    --min-ecg-by-patient 2 \
    --patient-id-col patient_id \
    --fname-col preprocess_fname \
    --workers 16 --seed 42

# =============================================================================
# Step 4: Create combined manifests
# =============================================================================
echo "=== Step 4a: Creating flat manifests per source ==="
mkdir -p "${MANIFEST_DIR}/per_source"

python "${MANIFEST_SCRIPT}" "${MHI_DEST}" --dest "${MANIFEST_DIR}/per_source" --split mhi
python "${MANIFEST_SCRIPT}" "${MIMIC_DEST}" --dest "${MANIFEST_DIR}/per_source" --split mimic
python "${MANIFEST_SCRIPT}" "${CODE15_DEST}" --dest "${MANIFEST_DIR}/per_source" --split code15

echo "=== Step 4b: Merging flat manifests ==="
{
    echo "${DATA_DIR}"
    tail -n +2 "${MANIFEST_DIR}/per_source/mhi.tsv" | sed 's|^|mhi/|'
    tail -n +2 "${MANIFEST_DIR}/per_source/mimic.tsv" | sed 's|^|mimic/|'
    tail -n +2 "${MANIFEST_DIR}/per_source/code15.tsv" | sed 's|^|code15/|'
} > "${MANIFEST_DIR}/train.tsv"

TOTAL=$(( $(wc -l < "${MANIFEST_DIR}/train.tsv") - 1 ))
echo "Combined manifest: ${TOTAL} entries"

echo "=== Step 4c: Converting to CMSC manifest ==="
python "${CMSC_CONVERT_SCRIPT}" \
    "${MANIFEST_DIR}" \
    --dest "${MANIFEST_DIR}" \
    --ext mat

# =============================================================================
# Step 5: SSL Pretraining
# =============================================================================
echo "=== Step 5: SSL Pretraining ==="
CUDA_VISIBLE_DEVICES=0,1,2 fairseq-hydra-train \
    common.fp16=true \
    task.data="${MANIFEST_DIR}/cmsc/" \
    checkpoint.save_dir="${DATA_DIR}/checkpoints-all" \
    common.wandb_project=wav2vec2-pretraining-amp-preserved \
    --config-dir "${REPO_DIR}/examples/w2v_cmsc/config/pretraining" \
    --config-name w2v_cmsc_rlm

# =============================================================================
# Step 6: Finetuning on 77 labels
# =============================================================================
echo "=== Step 6: Finetuning on 77 labels ==="
CUDA_VISIBLE_DEVICES=0 fairseq-hydra-train \
    common.fp16=true \
    task.data="${MANIFEST_DIR}/finetune/" \
    model.model_path="${DATA_DIR}/checkpoints-all/checkpoint_best.pt" \
    +task.npy_dataset=true \
    model.num_labels=77 \
    criterion._name=binary_cross_entropy_with_logits \
    checkpoint.save_dir="${DATA_DIR}/ft-77labels/" \
    --config-dir "${REPO_DIR}/examples/w2v_cmsc/config/finetuning/ecg_transformer" \
    --config-name diagnosis

# =============================================================================
# Step 7: Evaluation
# =============================================================================
echo "=== Step 7: Evaluation ==="
CUDA_VISIBLE_DEVICES=0 fairseq-hydra-inference \
    task.data="${MANIFEST_DIR}/finetune/" \
    common_eval.path="${DATA_DIR}/ft-77labels/checkpoint_best.pt" \
    common_eval.results_path="${DATA_DIR}/eval/" \
    +task.npy_dataset=true \
    model.num_labels=77 \
    dataset.valid_subset=test \
    --config-dir "${REPO_DIR}/examples/w2v_cmsc/config/finetuning/ecg_transformer" \
    --config-name eval

echo "=== Done! ==="
echo "Evaluation results saved to: ${DATA_DIR}/eval/"
