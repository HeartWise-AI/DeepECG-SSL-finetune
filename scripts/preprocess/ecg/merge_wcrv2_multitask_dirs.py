#!/usr/bin/env python3
"""Merge split files built in a secondary directory into the main WCR v2 multi-task dataset dir
and rewrite every manifest so paths point at the final location.

Used on 2026-10-05: train was built from per-ECG files (slow NAS path) in `data/wcrv2_multitask`,
val/test were built from the consolidated MHI arrays in `data/wcrv2_multitask_vt`.

    python3 scripts/preprocess/ecg/merge_wcrv2_multitask_dirs.py \
        --main data/wcrv2_multitask --from data/wcrv2_multitask_vt --splits val test echonext_test mhi_test
    python3 scripts/preprocess/ecg/build_wcrv2_multitask_dataset.py --out data/wcrv2_multitask --refresh-aux
"""
import argparse
import json
import shutil
from pathlib import Path

T, L = 2500, 12
FILES = ["{s}_x.npy", "{s}_y.npy", "{s}_aux.npy", "{s}_meta.parquet"]


def write_manifest(out_data, out_man, s, n_labels):
    with open(out_man / f"{s}.tsv", "w") as f:
        f.write(f"x_path:{out_data / f'{s}_x.npy'}\n")
        f.write(f"x_shape:({T}, {L}, 1)\n")
        f.write(f"y_path:{out_data / f'{s}_y.npy'}\n")
        f.write(f"aux_path:{out_data / f'{s}_aux.npy'}\n")
        f.write(f"label_indexes:{list(range(n_labels))}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    args = ap.parse_args()
    main_dir, src = Path(args.main).resolve(), Path(args.src).resolve()
    out_data, out_man = main_dir / "data", main_dir / "manifests"
    out_man.mkdir(exist_ok=True)
    for s in args.splits:
        for pat in FILES:
            f = src / "data" / pat.format(s=s)
            if not f.exists():
                raise SystemExit(f"missing {f}")
            dst = out_data / f.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(f), str(dst))
            print(f"moved {f.name}")
    info = json.load(open(main_dir / "labels.json"))
    n_labels = len(info["labels"])
    for s in ["train", "val", "test", "echonext_test", "mhi_test"]:
        if (out_data / f"{s}_x.npy").exists() and (out_data / f"{s}_y.npy").exists():
            write_manifest(out_data, out_man, s, n_labels)
            print(f"manifest {s}.tsv")
    src_info = src / "labels.json"
    if src_info.exists():
        si = json.load(open(src_info))
        info["mhi_waveform_source"] = {
            "train": "waveform_path_psa per-ECG files (PSA-adjusted mV, within ~10 % of ADC x 0.00488)",
            "val/test": "consolidated raw-ADC arrays X_val_v1.1 / X_test_v1.2 x 0.00488 (true mV)",
        }
        info["cohort"] = si.get("cohort", info.get("cohort"))
        json.dump(info, open(main_dir / "labels.json", "w"), indent=2)
    print("done")


if __name__ == "__main__":
    main()
