#!/usr/bin/env python3
"""Plot MLP training history for Assignment 6."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HISTORY = REPO_ROOT / "outputs" / "mlp_early_stop" / "mlp_history.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "plots" / "mlp_training_curves.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BGG MLP early-stop curves.")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history = pd.read_csv(args.history)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(history["epoch"], history["train_f1"], label="train F1", linewidth=2)
    axes[0].plot(history["epoch"], history["validation_f1"], label="validation F1", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("F1 (high_rating)")
    axes[0].set_title("MLP F1 — train vs validation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    best_epoch = int(history.loc[history["validation_f1"].idxmax(), "epoch"])
    axes[1].plot(history["epoch"], history["validation_recall"], label="validation recall", linewidth=2)
    axes[1].plot(history["epoch"], history["validation_roc_auc"], label="validation ROC-AUC", linewidth=2)
    axes[1].axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("MLP validation metrics + selected checkpoint")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
