"""
Merge several evaluate.py result CSVs (one per policy/job) into a single file.

Usage
-----
    python -m benchmark.merge_eval_results out.csv in1.csv in2.csv ... \\
        [--drop-policy Random] [--drop-policy GreedyMyopic]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("in_paths", type=Path, nargs="+")
    parser.add_argument(
        "--drop-policy", action="append", default=[],
        help="Policy name to drop from the merged output (repeatable).",
    )
    args = parser.parse_args()

    out_resolved = args.out_path.resolve()
    for p in args.in_paths:
        if p.resolve() == out_resolved:
            raise ValueError(
                f"Output path {args.out_path} is the same as input file {p} — "
                "this would overwrite one of the inputs before reading it. "
                "Pick a distinct output filename."
            )

    frames = [pd.read_csv(p) for p in args.in_paths]
    merged = pd.concat(frames, ignore_index=True)

    if args.drop_policy:
        before = len(merged)
        merged = merged[~merged["policy"].isin(args.drop_policy)]
        print(f"Dropped policies {args.drop_policy}: {before - len(merged)} rows removed")

    merged.to_csv(args.out_path, index=False)

    print(f"Merged {len(args.in_paths)} files -> {args.out_path}  ({len(merged)} rows)")
    print(f"Policies: {merged['policy'].unique().tolist()}")


if __name__ == "__main__":
    main()
