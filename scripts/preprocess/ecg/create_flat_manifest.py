"""
Create a flat TSV manifest from preprocessed mat directories.

Walks a data directory, finds segment mat files (*_0.mat, *_1.mat, etc.),
reads the feats shape, and writes a TSV manifest compatible with
convert_to_cmsc_manifest.py.

Usage:
    python create_flat_manifest.py /path/to/data \
        --dest /path/to/manifest/ \
        --ext mat
"""

import argparse
import glob
import os

import scipy.io


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR",
        help="root directory containing preprocessed mat files"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR",
        help="output directory for manifest TSV"
    )
    parser.add_argument(
        "--ext", default="mat", type=str, metavar="EXT",
        help="extension of data files"
    )
    parser.add_argument(
        "--split", default="train", type=str,
        help="split name for the output TSV file (e.g., train, valid, test)"
    )
    return parser


def main(args):
    root_path = os.path.realpath(args.root)
    dest_path = os.path.realpath(args.dest)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # Find all segment mat files (ending with _N.mat where N is a digit)
    search_pattern = os.path.join(root_path, "**", f"*.{args.ext}")
    all_files = sorted(glob.glob(search_pattern, recursive=True))

    # Filter to only segment files (e.g., *_0.mat, *_1.mat)
    # Exclude full-length files (e.g., basename.mat without _N suffix)
    segment_files = []
    for f in all_files:
        basename = os.path.splitext(os.path.basename(f))[0]
        # Check if it ends with _N where N is a digit
        parts = basename.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            segment_files.append(f)

    print(f"Found {len(segment_files)} segment files in {root_path}")

    output_path = os.path.join(dest_path, f"{args.split}.tsv")
    with open(output_path, "w") as f:
        # First line: root directory
        print(root_path, file=f)

        for fpath in segment_files:
            rel_path = os.path.relpath(fpath, root_path)
            try:
                data = scipy.io.loadmat(fpath)
                if "feats" not in data:
                    print(f"WARNING: no 'feats' key in {fpath}, skipping")
                    continue
                length = data["feats"].shape[-1]
                print(f"{rel_path}\t{length}", file=f)
            except Exception as e:
                print(f"WARNING: could not read {fpath}: {e}, skipping")

    print(f"Wrote manifest to {output_path}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
