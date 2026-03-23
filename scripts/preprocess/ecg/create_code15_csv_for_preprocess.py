"""
Create a CSV for CODE15 data compatible with preprocess_parquet.py --npy-path.

Reads the CODE15 exams.csv and creates a CSV where the fname column has format
"patientid_rowindex.npy" so that preprocess_parquet.py can use --npy-path
with row-based indexing into the consolidated X_code15.npy.
"""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-path",
        default="/media/data1/achilsowa/code15/exams.csv",
        help="path to CODE15 exams.csv",
    )
    parser.add_argument(
        "--output-path",
        default="/volume/fairseq-signals/data/code15_for_preprocess.csv",
        help="output CSV path",
    )
    parser.add_argument(
        "--patient-id-col",
        default="patient_id",
        help="patient ID column name",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows, {df[args.patient_id_col].nunique()} unique patients")

    # Create fname column: patientid_rowindex.npy
    df["preprocess_fname"] = (
        df[args.patient_id_col].astype(str) + "_" + df.index.astype(str) + ".npy"
    )

    out = df[["preprocess_fname", args.patient_id_col]]
    out.to_csv(args.output_path, index=False)
    print(f"Wrote {len(out)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
