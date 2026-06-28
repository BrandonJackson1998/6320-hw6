#!/usr/bin/env python3
"""Prepare BGG portfolio dataset with stratified train/validation/test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from bgg_common import (
    DEFAULT_PREP_DIR,
    RANDOM_STATE,
    SPLIT_COL,
    TARGET_COL,
    add_target_and_text_features,
    feature_columns,
    load_games_frame,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare BGG data with stratified split by game row.")
    parser.add_argument("--games-csv", type=Path, default=None, help="Path to games.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREP_DIR, help="Prep output directory.")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    return parser.parse_args()


def assign_splits(frame: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float) -> pd.DataFrame:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")
    labeled = frame.copy()
    train, temp = train_test_split(
        labeled,
        test_size=(1.0 - train_frac),
        random_state=RANDOM_STATE,
        stratify=labeled[TARGET_COL],
    )
    relative_test = test_frac / (val_frac + test_frac)
    val, test = train_test_split(
        temp,
        test_size=relative_test,
        random_state=RANDOM_STATE,
        stratify=temp[TARGET_COL],
    )
    out = pd.concat([train, val, test], ignore_index=True)
    out[SPLIT_COL] = "train"
    out.loc[out.index.isin(val.index), SPLIT_COL] = "validation"
    out.loc[out.index.isin(test.index), SPLIT_COL] = "test"
    return out


def main() -> None:
    args = parse_args()
    games = load_games_frame(args.games_csv)
    prepared = add_target_and_text_features(games)
    prepared = assign_splits(prepared, args.train_frac, args.val_frac, args.test_frac)

    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(data_dir / "prepared_bgg.csv", index=False)
    for split_name in ("train", "validation", "test"):
        prepared[prepared[SPLIT_COL] == split_name].to_csv(data_dir / f"{split_name}.csv", index=False)

    split_counts = (
        prepared[SPLIT_COL]
        .value_counts()
        .rename_axis(SPLIT_COL)
        .reset_index(name="rows")
    )
    split_counts["positive_rate"] = [
        float(prepared.loc[prepared[SPLIT_COL] == split, TARGET_COL].mean())
        for split in split_counts[SPLIT_COL]
    ]
    split_counts.to_csv(args.output_dir / "split_counts.csv", index=False)

    manifest = {
        "source": "Kaggle threnjen/board-games-database-from-boardgamegeek",
        "unit_of_analysis": "one row per game (BGGId)",
        "split_policy": "stratified random hold-out by game row; seed 6320",
        "split_fractions": {
            "train": args.train_frac,
            "validation": args.val_frac,
            "test": args.test_frac,
        },
        "target": f"{TARGET_COL} = 1 when AvgRating >= 7.0",
        "rows_total": int(len(prepared)),
        "feature_columns": feature_columns(),
        "excluded_from_x": sorted(
            set(games.columns) - set(feature_columns()) - {TARGET_COL, SPLIT_COL, "AvgRating"}
        ),
        "positive_rate_overall": float(prepared[TARGET_COL].mean()),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Wrote prepared data to {data_dir}")
    print(split_counts.to_string(index=False))


if __name__ == "__main__":
    main()
