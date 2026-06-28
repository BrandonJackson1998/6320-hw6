#!/usr/bin/env python3
"""Split audit tables for Assignment 6."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bgg_common import (
    CATEGORY_COLUMNS,
    DEFAULT_PREP_DIR,
    NUMERIC_METADATA_COLUMNS,
    SLICE_COLUMNS,
    SPLIT_COL,
    TARGET_COL,
)


def split_counts(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame[SPLIT_COL].value_counts().rename_axis(SPLIT_COL).reset_index(name="rows")
    counts["percent"] = counts["rows"] / len(frame) * 100
    counts["positive_rate"] = [
        float(frame.loc[frame[SPLIT_COL] == split, TARGET_COL].mean()) for split in counts[SPLIT_COL]
    ]
    return counts.sort_values(SPLIT_COL).reset_index(drop=True)


def numeric_distributions_by_split(frame: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for split_name, split_frame in frame.groupby(SPLIT_COL, sort=False):
        for column in numeric_cols:
            values = pd.to_numeric(split_frame[column], errors="coerce").dropna()
            rows.append(
                {
                    SPLIT_COL: split_name,
                    "column": column,
                    "count": int(values.count()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "p25": float(values.quantile(0.25)),
                    "median": float(values.quantile(0.50)),
                    "p75": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def category_positive_rate_by_split(frame: pd.DataFrame, category_cols: list[str]) -> pd.DataFrame:
    rows = []
    for split_name, split_frame in frame.groupby(SPLIT_COL, sort=False):
        for column in category_cols:
            grouped = split_frame.groupby(column, observed=False)[TARGET_COL].agg(["count", "mean"]).reset_index()
            grouped[SPLIT_COL] = split_name
            grouped["column"] = column
            grouped = grouped.rename(columns={column: "category_value", "mean": "positive_rate"})
            rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create BGG split-audit tables.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_PREP_DIR / "data" / "prepared_bgg.csv",
        help="Prepared dataset CSV with split column.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREP_DIR / "split_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.data)
    split_counts(frame).to_csv(args.output_dir / "split_counts.csv", index=False)
    numeric_distributions_by_split(frame, SLICE_COLUMNS + ["GameWeight"]).to_csv(
        args.output_dir / "split_numeric_distributions.csv",
        index=False,
    )
    numeric_distributions_by_split(frame, NUMERIC_METADATA_COLUMNS).to_csv(
        args.output_dir / "split_numeric_metadata_distributions.csv",
        index=False,
    )
    category_positive_rate_by_split(frame, CATEGORY_COLUMNS).to_csv(
        args.output_dir / "split_category_positive_rates.csv",
        index=False,
    )
    print(f"Wrote split audit tables to {args.output_dir}")


if __name__ == "__main__":
    main()
