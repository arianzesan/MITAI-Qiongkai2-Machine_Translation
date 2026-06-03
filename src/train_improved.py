"""Train mT5-small using original English-Bangla data plus filtered synthetic pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train_baseline import main as train_baseline_main


def combine_training_data(original_train: Path, synthetic_train: Path, output: Path) -> None:
    original = pd.read_csv(original_train)
    synthetic = pd.read_csv(synthetic_train)
    combined = pd.concat([original[["english", "bangla"]], synthetic[["english", "bangla"]]], ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(f"Saved combined training data with {len(combined)} pairs to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create original+synthetic training CSV.")
    parser.add_argument("--original-train", required=True, type=Path)
    parser.add_argument("--synthetic-train", required=True, type=Path)
    parser.add_argument("--combined-out", default=Path("data/processed/train_original_plus_synthetic.csv"), type=Path)
    args = parser.parse_args()
    combine_training_data(args.original_train, args.synthetic_train, args.combined_out)
    print("Train the improved model by passing this combined CSV to src/train_baseline.py.")


if __name__ == "__main__":
    main()

