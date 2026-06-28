#!/usr/bin/env python3
"""Assignment 6 — fair BGG tabular model comparison (majority, logistic, GBT, MLP)."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from bgg_common import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREP_DIR,
    RANDOM_STATE,
    build_feature_matrix,
    load_prepared_split,
    split_frame,
    write_json,
)


@dataclass(frozen=True)
class MetricBundle:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def as_dict(self, prefix: str) -> dict[str, float]:
        return {f"{prefix}_{key}": value for key, value in asdict(self).items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BGG tabular model comparison for Assignment 6.")
    parser.add_argument("--prep-dir", type=Path, default=DEFAULT_PREP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mlp-max-epochs", type=int, default=200)
    parser.add_argument("--mlp-patience", type=int, default=15)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def metrics_from_proba(y_true: np.ndarray, y_proba: np.ndarray) -> MetricBundle:
    preds = (y_proba >= 0.5).astype(int)
    return MetricBundle(
        accuracy=float(accuracy_score(y_true, preds)),
        precision=float(precision_score(y_true, preds, zero_division=0)),
        recall=float(recall_score(y_true, preds, zero_division=0)),
        f1=float(f1_score(y_true, preds, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
    )


def majority_metrics(y_true: np.ndarray) -> MetricBundle:
    majority = 1 if y_true.mean() >= 0.5 else 0
    preds = np.full_like(y_true, fill_value=majority)
    proba = np.full_like(y_true, fill_value=float(majority), dtype=float)
    return metrics_from_proba(y_true, proba)


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp_with_early_stop(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[TabularMLP, pd.DataFrame, dict[str, float]]:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = TabularMLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pos_count = max(int((y_train == 1).sum()), 1)
    neg_count = max(int((y_train == 0).sum()), 1)
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)

    best_state: dict[str, torch.Tensor] | None = None
    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_logits = model(x_train_t)
            val_logits = model(x_val_t)
            train_proba = torch.sigmoid(train_logits).cpu().numpy()
            val_proba = torch.sigmoid(val_logits).cpu().numpy()

        train_metrics = metrics_from_proba(y_train, train_proba)
        val_metrics = metrics_from_proba(y_val, val_proba)
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(loss.item()),
                "train_f1": train_metrics.f1,
                "validation_f1": val_metrics.f1,
                "validation_recall": val_metrics.recall,
                "validation_roc_auc": val_metrics.roc_auc,
            }
        )

        if val_metrics.f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_metrics.f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    selection = {
        "selected_epoch": best_epoch,
        "selected_validation_f1": best_val_f1,
        "stopped_epoch": history_rows[-1]["epoch"],
        "patience": patience,
    }
    return model, pd.DataFrame(history_rows), selection


def predict_mlp_proba(model: TabularMLP, x_array: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_array, dtype=torch.float32, device=device))
        return torch.sigmoid(logits).cpu().numpy()


def main() -> None:
    args = parse_args()
    prepared = load_prepared_split(args.prep_dir)
    train = split_frame(prepared, "train")
    val = split_frame(prepared, "validation")
    test = split_frame(prepared, "test")

    x_train, y_train = build_feature_matrix(train)
    x_val, y_val = build_feature_matrix(val)
    x_test, y_test = build_feature_matrix(test)

    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()
    y_test_np = y_test.to_numpy()

    x_train_raw = x_train.to_numpy(dtype=float)
    x_val_raw = x_val.to_numpy(dtype=float)
    x_test_raw = x_test.to_numpy(dtype=float)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_raw)
    x_val_scaled = scaler.transform(x_val_raw)
    x_test_scaled = scaler.transform(x_test_raw)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    majority_val = majority_metrics(y_val_np)
    majority_test = majority_metrics(y_test_np)
    write_json(
        args.output_dir / "majority_baseline.json",
        {
            "model": "majority_class",
            "validation": majority_val.as_dict("validation"),
            "test": majority_test.as_dict("test"),
        },
    )
    rows.append(
        {
            "model": "Majority class",
            "key_settings": "Always predict not-high (prevalence baseline)",
            "validation_f1": majority_val.f1,
            "validation_recall": majority_val.recall,
            "validation_roc_auc": majority_val.roc_auc,
            "test_f1": majority_test.f1,
            "test_recall": majority_test.recall,
            "test_roc_auc": majority_test.roc_auc,
            "practical_notes": "Sanity check; F1=0 on positive class",
        }
    )

    logistic = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight=None,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    logistic.fit(x_train_scaled, y_train_np)
    logistic_val = metrics_from_proba(y_val_np, logistic.predict_proba(x_val_scaled)[:, 1])
    logistic_test = metrics_from_proba(y_test_np, logistic.predict_proba(x_test_scaled)[:, 1])
    logistic_dir = args.output_dir / "logistic_regression"
    logistic_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        logistic_dir / "summary.json",
        {
            "model": "LogisticRegression(l2, unweighted)",
            "preprocessing": "train-fitted StandardScaler on all features",
            "seed": RANDOM_STATE,
            "validation": logistic_val.as_dict("validation"),
            "test": logistic_test.as_dict("test"),
        },
    )
    rows.append(
        {
            "model": "Logistic regression",
            "key_settings": "L2, unweighted; train-fitted StandardScaler",
            "validation_f1": logistic_val.f1,
            "validation_recall": logistic_val.recall,
            "validation_roc_auc": logistic_val.roc_auc,
            "test_f1": logistic_test.f1,
            "test_recall": logistic_test.recall,
            "test_roc_auc": logistic_test.roc_auc,
            "practical_notes": "Transparent linear baseline; under-recalls minority class",
        }
    )

    gbt = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    gbt.fit(x_train_raw, y_train_np)
    gbt_val = metrics_from_proba(y_val_np, gbt.predict_proba(x_val_raw)[:, 1])
    gbt_test = metrics_from_proba(y_test_np, gbt.predict_proba(x_test_raw)[:, 1])
    gbt_dir = args.output_dir / "gradient_boosted_trees"
    gbt_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        gbt_dir / "summary.json",
        {
            "model": "HistGradientBoostingClassifier",
            "settings": {"max_depth": 6, "learning_rate": 0.08, "max_iter": 300},
            "preprocessing": "median-imputed raw features (trees invariant to scaling)",
            "seed": RANDOM_STATE,
            "validation": gbt_val.as_dict("validation"),
            "test": gbt_test.as_dict("test"),
        },
    )
    rows.append(
        {
            "model": "Gradient boosted trees",
            "key_settings": "HistGBM depth=6, lr=0.08, 300 est.; raw imputed features",
            "validation_f1": gbt_val.f1,
            "validation_recall": gbt_val.recall,
            "validation_roc_auc": gbt_val.roc_auc,
            "test_f1": gbt_test.f1,
            "test_recall": gbt_test.recall,
            "test_roc_auc": gbt_test.roc_auc,
            "practical_notes": "Strong nonlinear tabular benchmark; partial feature importances",
        }
    )

    mlp_model, mlp_history, mlp_selection = train_mlp_with_early_stop(
        x_train_scaled,
        y_train_np,
        x_val_scaled,
        y_val_np,
        max_epochs=args.mlp_max_epochs,
        patience=args.mlp_patience,
        learning_rate=args.mlp_lr,
        weight_decay=args.mlp_weight_decay,
        seed=RANDOM_STATE,
    )
    mlp_val = metrics_from_proba(y_val_np, predict_mlp_proba(mlp_model, x_val_scaled))
    mlp_test = metrics_from_proba(y_test_np, predict_mlp_proba(mlp_model, x_test_scaled))
    mlp_dir = args.output_dir / "mlp_early_stop"
    mlp_dir.mkdir(parents=True, exist_ok=True)
    mlp_history.to_csv(mlp_dir / "mlp_history.csv", index=False)
    write_json(
        mlp_dir / "summary.json",
        {
            "model": "TabularMLP(64-32, dropout=0.2)",
            "optimizer": "Adam",
            "learning_rate": args.mlp_lr,
            "weight_decay": args.mlp_weight_decay,
            "regularization": "dropout=0.2 + weight_decay + BCE pos_weight for imbalance",
            "selection": mlp_selection,
            "preprocessing": "train-fitted StandardScaler",
            "seed": RANDOM_STATE,
            "validation": mlp_val.as_dict("validation"),
            "test": mlp_test.as_dict("test"),
        },
    )
    rows.append(
        {
            "model": "MLP (early stop)",
            "key_settings": (
                f"64→32, dropout 0.2, Adam wd={args.mlp_weight_decay}; "
                f"best epoch {int(mlp_selection['selected_epoch'])} by val F1"
            ),
            "validation_f1": mlp_val.f1,
            "validation_recall": mlp_val.recall,
            "validation_roc_auc": mlp_val.roc_auc,
            "test_f1": mlp_test.f1,
            "test_recall": mlp_test.recall,
            "test_roc_auc": mlp_test.roc_auc,
            "practical_notes": "Neural tabular; harder to explain than trees/logistic",
        }
    )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.output_dir / "comparison_table.csv", index=False)
    write_json(
        args.output_dir / "comparison_manifest.json",
        {
            "seed": RANDOM_STATE,
            "split_policy": "stratified 70/15/15 by game row (Assignment 5)",
            "feature_manifest": "Assignment 4/5 charter exclusions enforced",
            "models": [row["model"] for row in rows],
        },
    )

    print(comparison.to_string(index=False))
    print(f"\nSaved comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
