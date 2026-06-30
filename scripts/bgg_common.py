"""Shared helpers for CS 6320 Assignment 6 — BGG tabular model comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GAMES_CSV = REPO_ROOT.parent / "6320-hw2" / "part_b" / "data" / "bgg" / "games.csv"
DEFAULT_PREP_DIR = REPO_ROOT / "prep" / "bgg_split"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
RANDOM_STATE = 6320
RATING_THRESHOLD = 7.0
TARGET_COL = "high_rating"
SPLIT_COL = "split"
LABEL_COL = "AvgRating"

RATING_LEAKAGE_COLUMNS = [
    "AvgRating",
    "BayesAvgRating",
    "StdDev",
    "NumRatings",
    "UsersRated",
    "RatingRank",
]

DEMAND_TIMING_COLUMNS = [
    "NumOwned",
    "NumWant",
    "NumWish",
]

POPULARITY_COLUMNS = [
    "NumUserRatings",
    "NumComments",
    "NumAlternates",
    "NumExpansions",
    "NumImplementations",
]

RANK_COLUMNS = [f"Rank:{name}" for name in [
    "boardgame",
    "strategygames",
    "abstracts",
    "familygames",
    "thematic",
    "cgs",
    "wargames",
    "partygames",
    "childrensgames",
]]

IDENTITY_TEXT_COLUMNS = ["BGGId", "Name", "Description"]

DROP_FROM_FEATURES = set(
    IDENTITY_TEXT_COLUMNS
    + RATING_LEAKAGE_COLUMNS
    + DEMAND_TIMING_COLUMNS
    + POPULARITY_COLUMNS
    + RANK_COLUMNS
    + ["ImagePath", "Family", TARGET_COL, SPLIT_COL]
)

NUMERIC_METADATA_COLUMNS = [
    "YearPublished",
    "MinPlayers",
    "MaxPlayers",
    "ComAgeRec",
    "LanguageEase",
    "BestPlayers",
    "GameWeight",
    "NumWeightVotes",
    "MfgPlaytime",
    "ComMinPlaytime",
    "ComMaxPlaytime",
    "MfgAgeRec",
    "IsReimplementation",
    "Kickstarted",
]

CATEGORY_COLUMNS = [
    "Cat:Thematic",
    "Cat:Strategy",
    "Cat:War",
    "Cat:Family",
    "Cat:CGS",
    "Cat:Abstract",
    "Cat:Party",
    "Cat:Childrens",
]

SLICE_COLUMNS = ["YearPublished", "GameWeight", "MinPlayers", "MaxPlayers"]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def default_games_path() -> Path:
    env = __import__("os").environ.get("GAMES_CSV")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_GAMES_CSV.resolve()


def load_games_frame(games_csv: Path | None = None) -> pd.DataFrame:
    path = games_csv or default_games_path()
    if not path.exists():
        raise FileNotFoundError(
            f"games.csv not found at {path}. Set GAMES_CSV or copy from 6320-hw2/part_b/data/bgg/."
        )
    return pd.read_csv(path)


def add_target_and_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if LABEL_COL not in out.columns:
        raise ValueError(f"Expected {LABEL_COL} in games.csv")
    out = out[out[LABEL_COL].notna()].copy()
    out[TARGET_COL] = (out[LABEL_COL] >= RATING_THRESHOLD).astype(int)
    out["name_char_len"] = out["Name"].fillna("").astype(str).str.len()
    out["description_char_len"] = out["Description"].fillna("").astype(str).str.len()
    out["description_word_count"] = (
        out["Description"].fillna("").astype(str).str.split().str.len()
    )
    if "GoodPlayers" in out.columns:
        raw = out["GoodPlayers"].fillna("[]").astype(str)
        out["goodplayers_count"] = raw.apply(
            lambda value: 0 if value in {"[]", ""} else max(value.count("'") // 2, 0)
        )
    else:
        out["goodplayers_count"] = 0
    return out


def feature_columns() -> list[str]:
    return NUMERIC_METADATA_COLUMNS + CATEGORY_COLUMNS + [
        "name_char_len",
        "description_char_len",
        "description_word_count",
        "goodplayers_count",
    ]


def build_feature_matrix(
    frame: pd.DataFrame,
    numeric_medians: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    cols = feature_columns()
    missing = [col for col in cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")
    x_frame = frame[cols].copy()
    for col in NUMERIC_METADATA_COLUMNS:
        x_frame[col] = pd.to_numeric(x_frame[col], errors="coerce")
    for col in CATEGORY_COLUMNS:
        x_frame[col] = pd.to_numeric(x_frame[col], errors="coerce").fillna(0).astype(int)
    medians = (
        numeric_medians
        if numeric_medians is not None
        else x_frame[NUMERIC_METADATA_COLUMNS].median(numeric_only=True)
    )
    x_frame[NUMERIC_METADATA_COLUMNS] = x_frame[NUMERIC_METADATA_COLUMNS].fillna(medians)
    x_frame["description_word_count"] = x_frame["description_word_count"].fillna(0)
    x_frame["name_char_len"] = x_frame["name_char_len"].fillna(0)
    x_frame["description_char_len"] = x_frame["description_char_len"].fillna(0)
    x_frame["goodplayers_count"] = x_frame["goodplayers_count"].fillna(0)
    x_frame = x_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x_frame, frame[TARGET_COL].astype(int)


def load_prepared_split(prep_dir: Path | None = None) -> pd.DataFrame:
    root = prep_dir or DEFAULT_PREP_DIR
    path = root / "data" / "prepared_bgg.csv"
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {path}. Run prepare_bgg_data.py first.")
    return pd.read_csv(path)


def split_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return frame[frame[SPLIT_COL] == split_name].copy()
