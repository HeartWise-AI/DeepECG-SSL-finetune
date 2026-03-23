#!/usr/bin/env python3
"""
Estimate a fixed scale factor for an ECG dataset.

Computes the dataset-level scale factor that converts raw signals to
millivolts by comparing lead-0 mean power to a reference derived from
datasets with known ADC gains (MHI: 0.00488, MIMIC: 0.001).

For known devices, prefer the documented ADC gain:
    MUSE GE (MHI):  --scale 0.00488   (4.88 uV/ADC unit)
    MIMIC:           --scale 0.001
    PTB-XL (raw):    --scale ~0.005    (similar ADC to MUSE GE)

For unknown devices/datasets, this tool estimates the scale factor.

Examples:
    # From a consolidated NPY file
    python calibrate_dataset_scale.py /path/to/X_data.npy

    # From a directory of individual NPY files
    python calibrate_dataset_scale.py /path/to/npy_dir/

    # With more samples for higher accuracy
    python calibrate_dataset_scale.py /path/to/X_data.npy --n-samples 5000

    # Verify against a known scale factor
    python calibrate_dataset_scale.py /path/to/X_mhi.npy --known-scale 0.00488
"""

import argparse
import importlib.util
import os
import sys


def _load_calibration_module():
    """Load the calibration module without triggering the full fairseq_signals init."""
    module_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "fairseq_signals", "data", "ecg", "calibration.py",
    )
    module_path = os.path.normpath(module_path)
    spec = importlib.util.spec_from_file_location("calibration", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load calibration module from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    parser = argparse.ArgumentParser(
        description="Estimate a fixed scale factor for an ECG dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "source",
        help="Path to a consolidated .npy file or a directory of .npy files",
    )
    parser.add_argument(
        "--n-samples", type=int, default=2000,
        help="Number of random samples to use for estimation (default: 2000)",
    )
    parser.add_argument(
        "--lead", type=int, default=0,
        help="Lead index for power computation (default: 0 = lead I)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--known-scale", type=float, default=None,
        help="If provided, verify the estimated scale against this known value",
    )
    args = parser.parse_args()

    cal = _load_calibration_module()

    print(f"Source: {args.source}")
    print(f"Reference power: {cal.REFERENCE_LEAD0_POWER_MV:.6f} mV^2")
    print(f"Sampling {args.n_samples} signals, lead {args.lead}, seed {args.seed}")
    print()

    scale = cal.estimate_dataset_scale(
        source=args.source,
        n_samples=args.n_samples,
        lead=args.lead,
        seed=args.seed,
    )

    print(f"Estimated scale factor: {scale:.6f}")
    print(f"  Use with: preprocess_parquet.py --scale {scale:.6f}")

    if args.known_scale is not None:
        ratio = scale / args.known_scale
        pct_diff = abs(ratio - 1.0) * 100
        print()
        print(f"Known scale: {args.known_scale}")
        print(f"Ratio (estimated / known): {ratio:.4f}")
        print(f"Difference: {pct_diff:.1f}%")
        if pct_diff < 10:
            print("PASS: Estimated scale is within 10% of known value.")
        elif pct_diff < 25:
            print("WARNING: Estimated scale differs by 10-25%. Consider using known value.")
        else:
            print("FAIL: Estimated scale differs by >25%. Check data or reference power.")


if __name__ == "__main__":
    main()
