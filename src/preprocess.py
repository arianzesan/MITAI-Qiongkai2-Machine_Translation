"""Dataset loading and preprocessing helpers for English-Bangla mT5 experiments."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from datasets import load_dataset


DATASET_PARQUET = "hf://datasets/ai4bharat/samanantar/bn/train-00000-of-00005.parquet"
SEED = 42


def clean_text(value: object) -> str:
    """Normalize text while keeping the original sentence content intact."""
    if value is None:
        return ""
    text = str(value).replace("\u200c", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(str(text).split())


def stream_clean_pairs(total_unique: int) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Stream clean unique English-Bangla sentence pairs from Samanantar."""
    dataset = load_dataset("parquet", data_files=DATASET_PARQUET, split="train", streaming=True)
    seen: set[tuple[str, str]] = set()
    pairs: list[dict[str, str]] = []
    scanned = empty_removed = duplicate_removed = 0

    for row in dataset:
        scanned += 1
        english = clean_text(row.get("src"))
        bangla = clean_text(row.get("tgt"))
        if not english or not bangla:
            empty_removed += 1
            continue
        key = (english, bangla)
        if key in seen:
            duplicate_removed += 1
            continue
        seen.add(key)
        pairs.append({"english": english, "bangla": bangla})
        if len(pairs) >= total_unique:
            break

    stats = {
        "candidate_rows_scanned": scanned,
        "empty_rows_removed": empty_removed,
        "duplicate_pairs_removed": duplicate_removed,
        "clean_unique_pairs": len(pairs),
    }
    return pairs, stats


def split_pairs(
    pairs: list[dict[str, str]],
    train_size: int,
    validation_size: int,
    test_size: int,
    seed: int = SEED,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Create a deterministic low-resource train/validation/test split."""
    import random

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    train = shuffled[:train_size]
    validation = shuffled[train_size : train_size + validation_size]
    test = shuffled[train_size + validation_size : train_size + validation_size + test_size]
    return train, validation, test


def dataset_statistics(rows_by_split: dict[str, Iterable[dict[str, str]]]) -> pd.DataFrame:
    records = []
    for split, rows_iter in rows_by_split.items():
        rows = list(rows_iter)
        records.append(
            {
                "Split": split,
                "Sentence pairs": len(rows),
                "Avg English length (words)": round(
                    sum(word_count(row["english"]) for row in rows) / max(1, len(rows)), 2
                ),
                "Avg Bangla length (words)": round(
                    sum(word_count(row["bangla"]) for row in rows) / max(1, len(rows)), 2
                ),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a deterministic Samanantar English-Bangla subset.")
    parser.add_argument("--total", type=int, default=10500)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--validation-size", type=int, default=500)
    parser.add_argument("--test-size", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs, preprocessing_stats = stream_clean_pairs(args.total)
    train, validation, test = split_pairs(pairs, args.train_size, args.validation_size, args.test_size)

    pd.DataFrame(train).to_csv(args.out_dir / "train.csv", index=False)
    pd.DataFrame(validation).to_csv(args.out_dir / "validation.csv", index=False)
    pd.DataFrame(test).to_csv(args.out_dir / "test.csv", index=False)
    dataset_statistics({"Train": train, "Validation": validation, "Test": test}).to_csv(
        args.out_dir / "dataset_statistics.csv", index=False
    )
    pd.DataFrame([preprocessing_stats]).to_csv(args.out_dir / "preprocessing_stats.csv", index=False)


if __name__ == "__main__":
    main()

