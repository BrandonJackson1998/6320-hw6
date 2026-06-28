# CS 6320 — Assignment 6

**Name:** Brandon Jackson  
**Repo:** BGG portfolio tabular model comparison (Part A) and portfolio checkpoint (Part B).

## Contents

| Path | Purpose |
| --- | --- |
| `scripts/prepare_bgg_data.py` | Stratified train/validation/test split + manifest |
| `scripts/run_split_audit.py` | Split counts and distribution audits |
| `scripts/train_model_comparison.py` | Majority, logistic, GBT, MLP comparison |
| `scripts/plot_mlp_training.py` | MLP early-stop training curves |
| `run_local.sh` | End-to-end local pipeline |
| `writeup/CS6320_Assignment6_Jackson.md` | Part A + Part B writeup |

## Data

Default source: `../6320-hw2/part_b/data/bgg/games.csv`

Override:

```bash
export GAMES_CSV=/path/to/games.csv
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run locally

```bash
bash run_local.sh
```

## Portfolio model (Assignments 4–5 charter)

- **Target:** `high_rating = 1` when `AvgRating >= 7.0`
- **Representation:** native tabular metadata + category flags + simple text-length features
- **Split:** stratified 70/15/15 by game row, seed `6320`
- **Models compared:** majority, logistic regression, gradient boosted trees, MLP with validation early stopping
