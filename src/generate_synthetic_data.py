"""Generate and filter synthetic English-Bangla pairs with a reverse mT5 model."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def clean_text(value: object) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.replace("\u200c", "")).strip()


def ascii_letter_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return 0.0 if not chars else sum(c.isascii() and c.isalpha() for c in chars) / len(chars)


def punctuation_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return 1.0 if not chars else sum(not c.isalnum() for c in chars) / len(chars)


def max_token_repetition_ratio(text: str) -> float:
    tokens = text.lower().split()
    return 1.0 if not tokens else max(Counter(tokens).values()) / len(tokens)


def keep_synthetic(english: str, bangla: str) -> tuple[bool, str]:
    english = clean_text(english)
    tokens = english.split()
    if not english:
        return False, "empty"
    if len(tokens) < 3 or len(english) < 10:
        return False, "too_short"
    if punctuation_ratio(english) > 0.45:
        return False, "mostly_punctuation"
    if ascii_letter_ratio(english) < 0.55:
        return False, "not_english_like"
    if max_token_repetition_ratio(english) > 0.45 and len(tokens) >= 5:
        return False, "repeated_generic"
    ratio = len(tokens) / max(1, len(str(bangla).split()))
    if ratio < 0.25 or ratio > 4.0:
        return False, "bad_length_ratio"
    return True, "kept"


def generate(model, tokenizer, bangla_sentences: list[str], batch_size: int, max_length: int) -> list[str]:
    predictions: list[str] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(bangla_sentences), batch_size):
            batch = [f"translate Bengali to English: {x}" for x in bangla_sentences[start : start + batch_size]]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(model.device)
            output = model.generate(**enc, max_new_tokens=max_length, num_beams=4, do_sample=False, no_repeat_ngram_size=3)
            predictions.extend(tokenizer.batch_decode(output, skip_special_tokens=True))
    return [clean_text(x) for x in predictions]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate filtered synthetic English-Bangla pairs.")
    parser.add_argument("--reverse-model", required=True, type=Path)
    parser.add_argument("--monolingual-bangla", required=True, type=Path, help="CSV with a bangla column.")
    parser.add_argument("--out", default=Path("data/processed/synthetic_filtered.csv"), type=Path)
    parser.add_argument("--raw-out", default=Path("data/processed/synthetic_raw.csv"), type=Path)
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.reverse_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.reverse_model).to(device)
    mono = pd.read_csv(args.monolingual_bangla)
    bangla = mono["bangla"].fillna("").astype(str).tolist()
    synthetic = generate(model, tokenizer, bangla, args.batch_size, args.max_length)

    raw = pd.DataFrame({"synthetic_english": synthetic, "bangla": bangla})
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.raw_out, index=False)

    kept = []
    seen = set()
    for english, target_bangla in zip(synthetic, bangla):
        ok, _reason = keep_synthetic(english, target_bangla)
        key = clean_text(english).lower()
        if ok and key not in seen:
            kept.append({"english": clean_text(english), "bangla": clean_text(target_bangla)})
            seen.add(key)
        if len(kept) >= args.target:
            break
    pd.DataFrame(kept).to_csv(args.out, index=False)
    print(f"Saved {len(kept)} filtered synthetic pairs to {args.out}")


if __name__ == "__main__":
    main()

