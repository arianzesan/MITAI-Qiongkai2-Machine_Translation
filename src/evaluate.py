"""SacreBLEU evaluation utility for generated English-Bangla translations."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import sacrebleu


def corpus_bleu(predictions: list[str], references: list[str], tokenizer: str = "flores200") -> dict[str, object]:
    score = sacrebleu.corpus_bleu(predictions, [references], tokenize=tokenizer)
    return {
        "bleu": float(score.score),
        "signature": score.format(signature=True),
        "tokenizer": tokenizer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions with SacreBLEU.")
    parser.add_argument("--predictions", required=True, type=Path, help="CSV containing a prediction column.")
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument("--reference-column", default="reference")
    parser.add_argument("--out", default=Path("results/bleu_scores.csv"), type=Path)
    parser.add_argument("--tokenizer", default="flores200")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    result = corpus_bleu(
        df[args.prediction_column].fillna("").astype(str).tolist(),
        df[args.reference_column].fillna("").astype(str).tolist(),
        tokenizer=args.tokenizer,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(args.out, index=False)
    print(result)


if __name__ == "__main__":
    main()

