"""Shared utilities for ECG model evaluation scripts."""

import ast
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


# 77 labels in model output order (from preprocess_parquet.py y_labels list)
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

ENLARGEMENT_LABELS = {
    'Left ventricular hypertrophy': 30,
    'Left atrial enlargement': 76,
    'Right atrial enlargement': 3,
    'Right ventricular hypertrophy': 70,
    'Bi-atrial enlargement': 60,
}

CATEGORIES = {
    'Enlargement': [30, 76, 3, 70, 60],  # LVH, LAE, RAE, RVH, BAE
    'Rhythm Disorders': [12, 49, 50, 55, 56, 8, 24, 7, 52, 54, 43, 32, 19, 4, 25, 15, 23, 64, 65, 16, 42],
    'Conduction Disorder': [37, 73, 46, 26, 5, 21, 20, 33, 51],
    'Infarction/Ischemia': [68, 34, 27, 67, 61, 11, 13, 38, 58, 22, 69, 45, 31, 62, 72],
}


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def load_memmap(path):
    """Load a memmap array with its associated pickle header."""
    header_path = path.replace('.npy', '_header.pkl')
    with open(header_path, 'rb') as f:
        header = pickle.load(f)
    return np.memmap(path, dtype=header['dtype'], mode='r', shape=tuple(header['shape']))


def parse_manifest(manifest_path):
    """Parse a key:value manifest file into a dict."""
    manifest = {}
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, val = line.split(':', 1)
                manifest[key] = val
    return manifest


def load_targets(manifest_path):
    """Load target labels from a manifest file."""
    manifest = parse_manifest(manifest_path)
    y = np.load(manifest['y_path'])
    if 'label_indexes' in manifest:
        idx = ast.literal_eval(manifest['label_indexes'])
        y = y[:, idx]
    return y


def compute_auroc_auprc(y_true, y_score):
    """Compute AUROC and AUPRC for a single label. Returns (None, None) if only one class present."""
    if len(np.unique(y_true)) < 2:
        return None, None
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)


def fmt_metric(val):
    """Format a metric value for display, handling None."""
    return f"{val:.4f}" if val is not None else "N/A"
