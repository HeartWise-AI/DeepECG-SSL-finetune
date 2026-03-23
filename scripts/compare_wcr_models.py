#!/usr/bin/env python3
"""Compare WCR v1 (z-score) vs WCR v2 (amp-preserved) across datasets."""

import numpy as np

from eval_utils import (
    CATEGORIES,
    ENLARGEMENT_LABELS,
    LABEL_NAMES,
    compute_auroc_auprc,
    fmt_metric,
    load_memmap,
    load_targets,
    sigmoid,
)


def eval_model(logits_path, targets, name):
    logits = load_memmap(logits_path)
    probs = sigmoid(np.asarray(logits, dtype=np.float32))
    n = min(len(probs), len(targets))
    probs, tgt = probs[:n], targets[:n]

    results = {}
    for i in range(tgt.shape[1]):
        auroc, auprc = compute_auroc_auprc(tgt[:, i], probs[:, i])
        results[i] = {'auroc': auroc, 'auprc': auprc, 'n_pos': int(tgt[:, i].sum())}
    return results


def print_comparison(v1_results, v2_results, dataset_name):
    print(f"\n{'='*100}")
    print(f"  {dataset_name}")
    print(f"{'='*100}")

    # Enlargement labels detail
    print(f"\n  ENLARGEMENT LABELS")
    print(f"  {'Label':<35} {'v1 AUROC':>9} {'v2 AUROC':>9} {'Δ AUROC':>9} {'v1 AUPRC':>9} {'v2 AUPRC':>9} {'Δ AUPRC':>9} {'N_pos':>7}")
    print(f"  {'-'*95}")

    enl_v1_aurocs, enl_v2_aurocs = [], []
    enl_v1_auprcs, enl_v2_auprcs = [], []
    for name, idx in ENLARGEMENT_LABELS.items():
        r1, r2 = v1_results[idx], v2_results[idx]
        a1, a2 = fmt_metric(r1['auroc']), fmt_metric(r2['auroc'])
        p1, p2 = fmt_metric(r1['auprc']), fmt_metric(r2['auprc'])
        da = f"{r2['auroc']-r1['auroc']:+.4f}" if r1['auroc'] is not None and r2['auroc'] is not None else "N/A"
        dp = f"{r2['auprc']-r1['auprc']:+.4f}" if r1['auprc'] is not None and r2['auprc'] is not None else "N/A"
        print(f"  {name:<35} {a1:>9} {a2:>9} {da:>9} {p1:>9} {p2:>9} {dp:>9} {r2['n_pos']:>7}")
        if r1['auroc'] is not None and r2['auroc'] is not None:
            enl_v1_aurocs.append(r1['auroc']); enl_v2_aurocs.append(r2['auroc'])
            enl_v1_auprcs.append(r1['auprc']); enl_v2_auprcs.append(r2['auprc'])

    if enl_v1_aurocs:
        m1a, m2a = np.mean(enl_v1_aurocs), np.mean(enl_v2_aurocs)
        m1p, m2p = np.mean(enl_v1_auprcs), np.mean(enl_v2_auprcs)
        print(f"  {'-'*95}")
        print(f"  {'MEAN (enlargement)':<35} {m1a:>9.4f} {m2a:>9.4f} {m2a-m1a:>+9.4f} {m1p:>9.4f} {m2p:>9.4f} {m2p-m1p:>+9.4f}")

    # Category summary
    print(f"\n  CATEGORY SUMMARY")
    print(f"  {'Category':<35} {'v1 AUROC':>9} {'v2 AUROC':>9} {'Δ AUROC':>9} {'v1 AUPRC':>9} {'v2 AUPRC':>9} {'Δ AUPRC':>9}")
    print(f"  {'-'*95}")

    for cat_name, indices in CATEGORIES.items():
        v1_aurocs = [v1_results[i]['auroc'] for i in indices if v1_results[i]['auroc'] is not None]
        v2_aurocs = [v2_results[i]['auroc'] for i in indices if v2_results[i]['auroc'] is not None]
        v1_auprcs = [v1_results[i]['auprc'] for i in indices if v1_results[i]['auprc'] is not None]
        v2_auprcs = [v2_results[i]['auprc'] for i in indices if v2_results[i]['auprc'] is not None]
        if v1_aurocs and v2_aurocs:
            m1a, m2a = np.mean(v1_aurocs), np.mean(v2_aurocs)
            m1p, m2p = np.mean(v1_auprcs), np.mean(v2_auprcs)
            print(f"  {cat_name:<35} {m1a:>9.4f} {m2a:>9.4f} {m2a-m1a:>+9.4f} {m1p:>9.4f} {m2p:>9.4f} {m2p-m1p:>+9.4f}")

    # Overall
    v1_all = [v1_results[i]['auroc'] for i in v1_results if v1_results[i]['auroc'] is not None]
    v2_all = [v2_results[i]['auroc'] for i in v2_results if v2_results[i]['auroc'] is not None]
    v1_all_p = [v1_results[i]['auprc'] for i in v1_results if v1_results[i]['auprc'] is not None]
    v2_all_p = [v2_results[i]['auprc'] for i in v2_results if v2_results[i]['auprc'] is not None]
    print(f"  {'-'*95}")
    print(f"  {'OVERALL (all labels)':<35} {np.mean(v1_all):>9.4f} {np.mean(v2_all):>9.4f} {np.mean(v2_all)-np.mean(v1_all):>+9.4f} {np.mean(v1_all_p):>9.4f} {np.mean(v2_all_p):>9.4f} {np.mean(v2_all_p)-np.mean(v1_all_p):>+9.4f}")


if __name__ == "__main__":
    # Dataset configs
    WCR_V1_BASE = '/media/data1/achilsowa/results/fairseq/outputs/2024-10-08/04-39-01/checkpoint_last-ft-labels-77-bce'
    WCR_V2_BASE = '/volume/fairseq-signals/data/ssl-amp-preserved/eval'

    datasets = [
        {
            'name': 'MHI Test Set',
            'v1_logits': f'{WCR_V1_BASE}/outputs_test.npy',
            'v2_logits': f'{WCR_V2_BASE}/outputs_test.npy',
            'v1_manifest': '/media/data1/achilsowa/datasets/fairseq/mhi-mimic-code15/manifest/finetune/unscaled-labels-77/test.tsv',
            'v2_manifest': '/volume/fairseq-signals/data/ssl-amp-preserved/manifest/finetune/test.tsv',
        },
        {
            'name': 'CLSA (External)',
            'v1_logits': f'{WCR_V1_BASE}/outputs_clsa_cleaned.npy',
            'v2_logits': f'{WCR_V2_BASE}/outputs_clsa_cleaned.npy',
            'v1_manifest': '/media/data1/achilsowa/datasets/fairseq/mhi-mimic-code15/manifest/finetune/unscaled-labels-77/clsa_cleaned.tsv',
            'v2_manifest': '/volume/fairseq-signals/data/ssl-amp-preserved/manifest/finetune/clsa_cleaned.tsv',
        },
        {
            'name': 'MIMIC-IV (External)',
            'v1_logits': f'{WCR_V1_BASE}/outputs_mimic_cleaned.npy',
            'v2_logits': f'{WCR_V2_BASE}/outputs_mimic_cleaned.npy',
            'v1_manifest': '/media/data1/achilsowa/datasets/fairseq/mhi-mimic-code15/manifest/finetune/unscaled-labels-77/mimic_cleaned.tsv',
            'v2_manifest': '/volume/fairseq-signals/data/ssl-amp-preserved/manifest/finetune/mimic_cleaned.tsv',
        },
    ]

    print("WCR v1 (z-score normalized) vs WCR v2 (amplitude-preserved)")
    print("=" * 100)

    for ds in datasets:
        targets_v1 = load_targets(ds['v1_manifest'])
        targets_v2 = load_targets(ds['v2_manifest'])

        v1 = eval_model(ds['v1_logits'], targets_v1, 'v1')
        v2 = eval_model(ds['v2_logits'], targets_v2, 'v2')
        print_comparison(v1, v2, ds['name'])
