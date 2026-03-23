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
import os

import numpy as np

from eval_utils import (
    ENLARGEMENT_LABELS,
    LABEL_NAMES,
    compute_auroc_auprc,
    fmt_metric,
    load_memmap,
    load_targets,
    sigmoid,
)


def load_predictions(results_dir, manifest_path, subset="test"):
    """Load logits from inference output and targets from manifest's y_path."""
    logits_path = os.path.join(results_dir, f"outputs_{subset}.npy")
    logits = load_memmap(logits_path)

    targets = load_targets(manifest_path)

    # Ensure same number of samples
    n = min(len(logits), len(targets))
    return logits[:n], targets[:n]


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
    probs = sigmoid(np.asarray(logits, dtype=np.float32))

    print(f"Loaded {logits.shape[0]} samples, {logits.shape[1]} labels")
    print()

    # Compute per-label metrics
    results = []
    for i in range(logits.shape[1]):
        auroc, auprc = compute_auroc_auprc(targets[:, i], probs[:, i])
        n_pos = int(targets[:, i].sum())
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
    for name, idx in ENLARGEMENT_LABELS.items():
        r = results[idx]
        print(f"{name:<40} {fmt_metric(r['auroc']):>8} {fmt_metric(r['auprc']):>8} {r['n_pos']:>8}")
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
            marker = " ***" if r['name'] in ENLARGEMENT_LABELS else ""
            print(f"{r['name']:<50} {fmt_metric(r['auroc']):>8} {fmt_metric(r['auprc']):>8} {r['n_pos']:>8}{marker}")


if __name__ == "__main__":
    main()
