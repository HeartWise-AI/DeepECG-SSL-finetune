#!/usr/bin/env python3
"""Compare WCR v1 (z-score) vs WCR v2 (amp-preserved) across datasets."""

import ast
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# 77 labels in model output order
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

CATEGORIES = {
    'Enlargement': [30, 76, 3, 70, 60],  # LVH, LAE, RAE, RVH, BAE
    'Rhythm Disorders': [12, 49, 50, 55, 56, 8, 24, 7, 52, 54, 43, 32, 19, 4, 25, 15, 23, 64, 65, 16, 42],
    'Conduction Disorder': [37, 73, 46, 26, 5, 21, 20, 33, 51],
    'Infarction/Ischemia': [68, 34, 27, 67, 61, 11, 13, 38, 58, 22, 69, 45, 31, 62, 72],
}

ENLARGEMENT_LABELS = {
    'Left ventricular hypertrophy': 30,
    'Left atrial enlargement': 76,
    'Right atrial enlargement': 3,
    'Right ventricular hypertrophy': 70,
    'Bi-atrial enlargement': 60,
}


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def load_memmap(path):
    header_path = path.replace('.npy', '_header.pkl')
    with open(header_path, 'rb') as f:
        header = pickle.load(f)
    return np.memmap(path, dtype=header['dtype'], mode='r', shape=tuple(header['shape']))


def load_targets(manifest_path):
    manifest = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, val = line.split(':', 1)
                manifest[key] = val

    y = np.load(manifest['y_path'])
    if 'label_indexes' in manifest:
        idx = ast.literal_eval(manifest['label_indexes'])
        y = y[:, idx]
    return y


def compute_auroc_auprc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None, None
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)


def eval_model(logits_path, targets, name):
    logits = np.array(load_memmap(logits_path), dtype=np.float32)
    probs = sigmoid(logits)
    n = min(len(probs), len(targets))
    probs, tgt = probs[:n], targets[:n]

    results = {}
    for i in range(77):
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
        a1 = f"{r1['auroc']:.4f}" if r1['auroc'] else "N/A"
        a2 = f"{r2['auroc']:.4f}" if r2['auroc'] else "N/A"
        p1 = f"{r1['auprc']:.4f}" if r1['auprc'] else "N/A"
        p2 = f"{r2['auprc']:.4f}" if r2['auprc'] else "N/A"
        da = f"{r2['auroc']-r1['auroc']:+.4f}" if r1['auroc'] and r2['auroc'] else "N/A"
        dp = f"{r2['auprc']-r1['auprc']:+.4f}" if r1['auprc'] and r2['auprc'] else "N/A"
        print(f"  {name:<35} {a1:>9} {a2:>9} {da:>9} {p1:>9} {p2:>9} {dp:>9} {r2['n_pos']:>7}")
        if r1['auroc'] and r2['auroc']:
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
    v1_all = [v1_results[i]['auroc'] for i in range(77) if v1_results[i]['auroc'] is not None]
    v2_all = [v2_results[i]['auroc'] for i in range(77) if v2_results[i]['auroc'] is not None]
    v1_all_p = [v1_results[i]['auprc'] for i in range(77) if v1_results[i]['auprc'] is not None]
    v2_all_p = [v2_results[i]['auprc'] for i in range(77) if v2_results[i]['auprc'] is not None]
    print(f"  {'-'*95}")
    print(f"  {'OVERALL (all labels)':<35} {np.mean(v1_all):>9.4f} {np.mean(v2_all):>9.4f} {np.mean(v2_all)-np.mean(v1_all):>+9.4f} {np.mean(v1_all_p):>9.4f} {np.mean(v2_all_p):>9.4f} {np.mean(v2_all_p)-np.mean(v1_all_p):>+9.4f}")


# ============================================================================
# Dataset configs: (v1_logits, v2_logits, manifest_for_targets, dataset_name)
# ============================================================================
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
