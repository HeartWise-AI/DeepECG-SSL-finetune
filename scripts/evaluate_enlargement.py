#!/usr/bin/env python3
"""
Evaluate per-label AUROC/AUPRC on the test set, highlighting enlargement labels.

Usage:
    python scripts/evaluate_enlargement.py \
        --results-dir /path/to/eval/ \
        --manifest /path/to/manifest/finetune/test.tsv \
        [--all-labels]
"""

import argparse
import ast
import numpy as np
import sys
import os

# The 77 labels in order (from preprocess_parquet.py y_labels list)
LABEL_NAMES = [
    'Acute pericarditis', 'QS complex in V1-V2-V3',
    'T wave inversion (anterior - V3-V4)', 'Right atrial enlargement',
    '2nd degree AV block - mobitz 1', 'Left posterior fascicular block',
    'Wolff-Parkinson-White (Pre-excitation syndrome)', 'Junctional rhythm',
    'Premature ventricular complex', "rSR' in V1-V2", 'Right superior axis',
    'ST elevation (inferior - II, III, aVF)', 'Afib',
    'ST elevation (anterior - V3-V4)', 'RV1 + SV6 > 11 mm', 'Sinusal',
    'Monomorph', 'Delta wave', 'R/S ratio in V1-V2 >1',
    'Third Degree AV Block', 'LV pacing',
    'Nonspecific intraventricular conduction delay',
    'ST depression (inferior - II, III, aVF)', 'Regular',
    'Premature atrial complex', '2nd degree AV block - mobitz 2',
    'Left anterior fascicular block', 'Q wave (septal- V1-V2)',
    'Prolonged QT', 'Left axis deviation', 'Left ventricular hypertrophy',
    'ST depression (septal- V1-V2)', 'Supraventricular tachycardia',
    'Atrial paced', 'Q wave (inferior - II, III, aVF)', 'no_qrs',
    'T wave inversion (lateral -I, aVL, V5-V6)', 'Right bundle branch block',
    'ST elevation (septal - V1-V2)', 'SV1 + RV5 or RV6 > 35 mm',
    'Right axis deviation', 'RaVL > 11 mm', 'Polymorph',
    'Ventricular tachycardia', 'QRS complex negative in III',
    'ST depression (lateral - I, avL, V5-V6)', '1st degree AV block',
    'Lead misplacement', 'Q wave (posterior - V7-V9)', 'Atrial flutter',
    'Ventricular paced', 'ST elevation (posterior - V7-V8-V9)',
    'Ectopic atrial rhythm (< 100 BPM)', 'Early repolarization',
    'Ventricular Rhythm', 'Irregularly irregular',
    'Atrial tachycardia (>= 100 BPM)', 'R complex in V5-V6',
    'ST elevation (lateral - I, aVL, V5-V6)', 'Brugada',
    'Bi-atrial enlargement', 'Q wave (lateral- I, aVL, V5-V6)',
    'ST upslopping', 'T wave inversion (inferior - II, III, aVF)',
    'Regularly irregular', 'Bradycardia', 'qRS in V5-V6-I, aVL',
    'Q wave (anterior - V3-V4)', 'Acute MI',
    'ST depression (anterior - V3-V4)', 'Right ventricular hypertrophy',
    'T wave inversion (septal- V1-V2)', 'ST downslopping',
    'Left bundle branch block', 'Low voltage', 'U wave',
    'Left atrial enlargement',
]

# Enlargement label indices (0-indexed in the 77-label list)
ENLARGEMENT_INDICES = {
    'Left ventricular hypertrophy': 30,
    'Left atrial enlargement': 76,
    'Right atrial enlargement': 3,
    'Right ventricular hypertrophy': 70,
    'Bi-atrial enlargement': 60,
}


def load_predictions(results_dir, manifest_path, subset="test"):
    """Load logits from inference output and targets from manifest's y_path."""
    import pickle

    # Load logits from inference output (memmap format with header)
    logits_path = os.path.join(results_dir, f"outputs_{subset}.npy")
    header_path = os.path.join(results_dir, f"outputs_{subset}_header.pkl")

    with open(header_path, "rb") as f:
        header = pickle.load(f)

    logits = np.memmap(logits_path, dtype=header["dtype"], mode="r",
                       shape=tuple(header["shape"]))

    # Parse manifest to get y_path and label_indexes
    manifest = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, val = line.split(":", 1)
                manifest[key] = val

    y_path = manifest["y_path"]
    label_indexes = ast.literal_eval(manifest["label_indexes"])

    # Load targets and select the same label columns
    all_targets = np.load(y_path)
    targets = all_targets[:, label_indexes]

    # Ensure same number of samples
    n = min(len(logits), len(targets))
    return logits[:n], targets[:n]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics(y_true, y_score):
    """Compute AUROC and AUPRC for a single label."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Skip if only one class present
    if len(np.unique(y_true)) < 2:
        return None, None, int(y_true.sum())

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    n_pos = int(y_true.sum())
    return auroc, auprc, n_pos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing outputs_test.npy from inference")
    parser.add_argument("--manifest", type=str, required=True,
                        help="Path to test manifest (e.g. manifest/finetune/test.tsv)")
    parser.add_argument("--subset", type=str, default="test",
                        help="Subset name (default: test)")
    parser.add_argument("--all-labels", action="store_true",
                        help="Print metrics for all 77 labels, not just enlargement")
    args = parser.parse_args()

    logits, targets = load_predictions(args.results_dir, args.manifest, args.subset)
    probs = sigmoid(logits)

    print(f"Loaded {logits.shape[0]} samples, {logits.shape[1]} labels")
    print()

    # Compute per-label metrics
    results = []
    for i in range(logits.shape[1]):
        auroc, auprc, n_pos = compute_metrics(targets[:, i], probs[:, i])
        results.append({
            'index': i,
            'name': LABEL_NAMES[i],
            'auroc': auroc,
            'auprc': auprc,
            'n_pos': n_pos,
        })

    # Print enlargement labels
    print("=" * 80)
    print("ENLARGEMENT LABELS (amplitude-sensitive)")
    print("=" * 80)
    print(f"{'Label':<40} {'AUROC':>8} {'AUPRC':>8} {'N_pos':>8}")
    print("-" * 80)

    enlargement_aurocs = []
    enlargement_auprcs = []
    for name, idx in ENLARGEMENT_INDICES.items():
        r = results[idx]
        auroc_str = f"{r['auroc']:.4f}" if r['auroc'] is not None else "N/A"
        auprc_str = f"{r['auprc']:.4f}" if r['auprc'] is not None else "N/A"
        print(f"{name:<40} {auroc_str:>8} {auprc_str:>8} {r['n_pos']:>8}")
        if r['auroc'] is not None:
            enlargement_aurocs.append(r['auroc'])
            enlargement_auprcs.append(r['auprc'])

    if enlargement_aurocs:
        print("-" * 80)
        print(f"{'MEAN (enlargement)':<40} {np.mean(enlargement_aurocs):>8.4f} {np.mean(enlargement_auprcs):>8.4f}")
    print()

    # Overall metrics
    all_aurocs = [r['auroc'] for r in results if r['auroc'] is not None]
    all_auprcs = [r['auprc'] for r in results if r['auprc'] is not None]
    print(f"{'MEAN (all 77 labels)':<40} {np.mean(all_aurocs):>8.4f} {np.mean(all_auprcs):>8.4f}")
    print()

    if args.all_labels:
        print("=" * 80)
        print("ALL LABELS")
        print("=" * 80)
        print(f"{'Label':<50} {'AUROC':>8} {'AUPRC':>8} {'N_pos':>8}")
        print("-" * 80)
        # Sort by AUROC descending
        sorted_results = sorted(results, key=lambda r: r['auroc'] if r['auroc'] is not None else 0, reverse=True)
        for r in sorted_results:
            auroc_str = f"{r['auroc']:.4f}" if r['auroc'] is not None else "N/A"
            auprc_str = f"{r['auprc']:.4f}" if r['auprc'] is not None else "N/A"
            marker = " ***" if r['name'] in ENLARGEMENT_INDICES else ""
            print(f"{r['name']:<50} {auroc_str:>8} {auprc_str:>8} {r['n_pos']:>8}{marker}")


if __name__ == "__main__":
    main()
