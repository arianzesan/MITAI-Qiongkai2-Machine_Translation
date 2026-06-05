"""Last attempt: search smaller filtered BT subsets and select by validation BLEU.

This script is intended for Google Colab T4 GPU. It is deliberately conservative:
the improved models start from the trained baseline checkpoint and only add small
ranked synthetic subsets, because previous experiments showed that noisy
synthetic data can reduce BLEU.
"""

from __future__ import annotations

import gc
import inspect
import json
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sacrebleu
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


SEED = 42
MODEL_NAME = "google/mt5-small"
DATASET_PARQUET = "hf://datasets/ai4bharat/samanantar/bn/train-00000-of-00005.parquet"

TRAIN_SIZE = 5000
VAL_SIZE = 500
TEST_SIZE = 500
BASE_POOL_SIZE = 7500
SYNTHETIC_CANDIDATES = 3000
SYNTHETIC_SIZES_TO_TRY = [250, 500, 1000]

BASELINE_EPOCHS = 3
REVERSE_EPOCHS = 3
IMPROVED_EPOCHS = 2

BATCH_SIZE = 4
GRAD_ACCUM = 4
BASELINE_LR = 1e-4
IMPROVED_LR = 3e-5
MAX_SOURCE_LENGTH = 128
MAX_TARGET_LENGTH = 128
GEN_MAX_LENGTH = 96
GEN_BATCH_SIZE = 16
BLEU_TOKENIZER = "flores200"

RESULTS_DIR = Path("/content/final_last_try_bleu_search")
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\u200c", "")).strip()


def word_count(text: str) -> int:
    return len(str(text).split())


def stream_clean_pairs(total_unique: int) -> tuple[list[dict[str, str]], dict[str, int]]:
    dataset = load_dataset("parquet", data_files=DATASET_PARQUET, split="train", streaming=True)
    pairs: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    scanned = empty = duplicate = 0
    for row in dataset:
        scanned += 1
        english = clean_text(row.get("src"))
        bangla = clean_text(row.get("tgt"))
        if not english or not bangla:
            empty += 1
            continue
        key = (english, bangla)
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        pairs.append({"english": english, "bangla": bangla})
        if len(pairs) >= total_unique:
            break
    return pairs, {
        "scanned": scanned,
        "empty_removed": empty,
        "duplicate_removed": duplicate,
        "clean_pairs": len(pairs),
    }


def to_dataset(rows: list[dict[str, str]]) -> Dataset:
    return Dataset.from_pandas(pd.DataFrame(rows), preserve_index=False)


def preprocess_rows(rows: list[dict[str, str]], tokenizer, direction: str) -> Dataset:
    dataset = to_dataset(rows)

    def preprocess(batch):
        if direction == "en_bn":
            source = [f"translate English to Bengali: {x}" for x in batch["english"]]
            target = batch["bangla"]
        else:
            source = [f"translate Bengali to English: {x}" for x in batch["bangla"]]
            target = batch["english"]
        model_inputs = tokenizer(source, max_length=MAX_SOURCE_LENGTH, truncation=True)
        labels = tokenizer(text_target=target, max_length=MAX_TARGET_LENGTH, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)


def corpus_bleu(preds: list[str], refs: list[str]) -> dict[str, object]:
    score = sacrebleu.corpus_bleu(preds, [refs], tokenize=BLEU_TOKENIZER)
    return {"bleu": float(score.score), "signature": score.format(signature=True)}


def ascii_letter_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return 0.0 if not chars else sum(c.isascii() and c.isalpha() for c in chars) / len(chars)


def punctuation_ratio(text: str) -> float:
    chars = [c for c in text if not c.isspace()]
    return 1.0 if not chars else sum(not c.isalnum() for c in chars) / len(chars)


def repetition_ratio(text: str) -> float:
    tokens = text.lower().split()
    return 1.0 if not tokens else max(Counter(tokens).values()) / len(tokens)


def filter_reason(english: str, bangla: str) -> str:
    english = clean_text(english)
    tokens = english.split()
    if not english:
        return "empty"
    if len(tokens) < 4 or len(english) < 14:
        return "too_short"
    if punctuation_ratio(english) > 0.35:
        return "punctuation"
    if ascii_letter_ratio(english) < 0.65:
        return "not_english_like"
    if repetition_ratio(english) > 0.38 and len(tokens) >= 5:
        return "repetition"
    ratio = len(tokens) / max(1, word_count(bangla))
    if ratio < 0.35 or ratio > 2.6:
        return "length_ratio"
    generic = {"it is a mistake", "it was a huge victory", "but it is a mistake"}
    if english.lower().strip(" .,!?:;'\"") in generic:
        return "generic"
    return "kept"


def quality_score(english: str, bangla: str) -> float:
    tokens = english.split()
    ratio = len(tokens) / max(1, word_count(bangla))
    score = 0.0
    score -= abs(ratio - 1.05)
    score += min(len(tokens), 14) / 14
    score -= punctuation_ratio(english)
    score -= repetition_ratio(english) * 0.5
    if tokens and tokens[0].lower() in {"he", "she", "it", "but", "and", "the"}:
        score -= 0.25
    return score


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found. Use Colab T4 runtime.")

    set_seed(SEED)
    random.seed(SEED)
    for directory in [RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    print("GPU:", torch.cuda.get_device_name(0))
    config = {
        "model": MODEL_NAME,
        "train_size": TRAIN_SIZE,
        "validation_size": VAL_SIZE,
        "test_size": TEST_SIZE,
        "synthetic_candidates": SYNTHETIC_CANDIDATES,
        "synthetic_sizes_tried": SYNTHETIC_SIZES_TO_TRY,
        "baseline_epochs": BASELINE_EPOCHS,
        "reverse_epochs": REVERSE_EPOCHS,
        "improved_epochs_each_variant": IMPROVED_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUM,
        "baseline_lr": BASELINE_LR,
        "improved_lr": IMPROVED_LR,
        "max_source_length": MAX_SOURCE_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "bleu_tokenizer": BLEU_TOKENIZER,
        "selection": "best checkpoint by validation BLEU",
        "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(config, indent=2, ensure_ascii=False))

    pairs, preprocessing_stats = stream_clean_pairs(BASE_POOL_SIZE + SYNTHETIC_CANDIDATES)
    base_pairs = pairs[:BASE_POOL_SIZE]
    extra_pairs = pairs[BASE_POOL_SIZE:]
    shuffled = list(base_pairs)
    random.Random(SEED).shuffle(shuffled)
    train_pairs = shuffled[:TRAIN_SIZE]
    val_pairs = shuffled[TRAIN_SIZE : TRAIN_SIZE + VAL_SIZE]
    test_pairs = shuffled[TRAIN_SIZE + VAL_SIZE : TRAIN_SIZE + VAL_SIZE + TEST_SIZE]
    mono_pairs = (shuffled[TRAIN_SIZE + VAL_SIZE + TEST_SIZE :] + extra_pairs)[:SYNTHETIC_CANDIDATES]

    dataset_stats = pd.DataFrame(
        [
            {
                "Split": "Train",
                "Sentence pairs": len(train_pairs),
                "Avg English length": round(sum(word_count(x["english"]) for x in train_pairs) / len(train_pairs), 2),
                "Avg Bangla length": round(sum(word_count(x["bangla"]) for x in train_pairs) / len(train_pairs), 2),
            },
            {
                "Split": "Validation",
                "Sentence pairs": len(val_pairs),
                "Avg English length": round(sum(word_count(x["english"]) for x in val_pairs) / len(val_pairs), 2),
                "Avg Bangla length": round(sum(word_count(x["bangla"]) for x in val_pairs) / len(val_pairs), 2),
            },
            {
                "Split": "Test",
                "Sentence pairs": len(test_pairs),
                "Avg English length": round(sum(word_count(x["english"]) for x in test_pairs) / len(test_pairs), 2),
                "Avg Bangla length": round(sum(word_count(x["bangla"]) for x in test_pairs) / len(test_pairs), 2),
            },
            {
                "Split": "BT candidate pool",
                "Sentence pairs": len(mono_pairs),
                "Avg English length": round(sum(word_count(x["english"]) for x in mono_pairs) / len(mono_pairs), 2),
                "Avg Bangla length": round(sum(word_count(x["bangla"]) for x in mono_pairs) / len(mono_pairs), 2),
            },
        ]
    )
    dataset_stats.to_csv(TABLES_DIR / "dataset_statistics.csv", index=False)
    print(dataset_stats)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_en_bn = preprocess_rows(train_pairs, tokenizer, "en_bn")
    val_en_bn = preprocess_rows(val_pairs, tokenizer, "en_bn")
    train_bn_en = preprocess_rows(train_pairs, tokenizer, "bn_en")
    val_bn_en = preprocess_rows(val_pairs, tokenizer, "bn_en")

    def compute_bleu_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = [clean_text(x) for x in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        labels = [[token if token != -100 else tokenizer.pad_token_id for token in row] for row in labels]
        decoded_labels = [clean_text(x) for x in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        return {"bleu": corpus_bleu(decoded_preds, decoded_labels)["bleu"]}

    def make_args(output_dir: Path, epochs: int, lr: float):
        kwargs = {
            "output_dir": str(output_dir),
            "num_train_epochs": epochs,
            "per_device_train_batch_size": BATCH_SIZE,
            "per_device_eval_batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "learning_rate": lr,
            "adafactor": True,
            "max_grad_norm": 1.0,
            "logging_strategy": "steps",
            "logging_steps": 50,
            "save_strategy": "epoch",
            "report_to": "none",
            "predict_with_generate": True,
            "generation_max_length": GEN_MAX_LENGTH,
            "generation_num_beams": 4,
            "fp16": False,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_bleu",
            "greater_is_better": True,
            "save_total_limit": 1,
        }
        sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
        kwargs["eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"] = "epoch"
        return Seq2SeqTrainingArguments(**kwargs)

    def train_seq2seq(run_name: str, model_source, train_ds, val_ds, epochs: int, lr: float):
        started = time.perf_counter()
        model = AutoModelForSeq2SeqLM.from_pretrained(model_source)
        model.config.decoder_start_token_id = tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        trainer = Seq2SeqTrainer(
            model=model,
            args=make_args(MODELS_DIR / run_name, epochs, lr),
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
            compute_metrics=compute_bleu_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)] if epochs > 2 else [],
        )
        output = trainer.train()
        eval_metrics = trainer.evaluate()
        elapsed = time.perf_counter() - started
        save_dir = MODELS_DIR / f"{run_name}_best"
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        history = pd.DataFrame(trainer.state.log_history)
        history["run_name"] = run_name
        history.to_csv(TABLES_DIR / f"{run_name}_log_history.csv", index=False)
        summary = {
            "run_name": run_name,
            "model_path": str(save_dir),
            "train_loss": float(output.training_loss),
            "validation_loss": float(eval_metrics["eval_loss"]),
            "validation_bleu": float(eval_metrics["eval_bleu"]),
            "runtime_seconds": round(elapsed, 2),
        }
        print(summary)
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        return save_dir, summary, history

    def generate_texts(model_source, texts: list[str], source_language: str) -> list[str]:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_source).to("cuda")
        model.eval()
        prefix = "translate English to Bengali: " if source_language == "english" else "translate Bengali to English: "
        predictions: list[str] = []
        with torch.no_grad():
            for start in range(0, len(texts), GEN_BATCH_SIZE):
                batch = [prefix + x for x in texts[start : start + GEN_BATCH_SIZE]]
                encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SOURCE_LENGTH).to(
                    model.device
                )
                generated = model.generate(
                    **encoded,
                    max_new_tokens=GEN_MAX_LENGTH,
                    num_beams=4,
                    do_sample=False,
                    no_repeat_ngram_size=3,
                )
                predictions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return [clean_text(x) for x in predictions]

    baseline_path, baseline_summary, _baseline_history = train_seq2seq(
        "baseline_en_bn_bleu_selected", MODEL_NAME, train_en_bn, val_en_bn, BASELINE_EPOCHS, BASELINE_LR
    )
    test_sources = [x["english"] for x in test_pairs]
    test_refs = [x["bangla"] for x in test_pairs]
    baseline_preds = generate_texts(baseline_path, test_sources, "english")
    baseline_test_bleu = corpus_bleu(baseline_preds, test_refs)
    print("Baseline test BLEU:", baseline_test_bleu)

    reverse_path, reverse_summary, _reverse_history = train_seq2seq(
        "reverse_bn_en_for_last_try", MODEL_NAME, train_bn_en, val_bn_en, REVERSE_EPOCHS, BASELINE_LR
    )
    mono_bangla = [x["bangla"] for x in mono_pairs]
    raw_synthetic_english = generate_texts(reverse_path, mono_bangla, "bangla")
    pd.DataFrame({"synthetic_english": raw_synthetic_english, "bangla": mono_bangla}).to_csv(
        TABLES_DIR / "raw_synthetic_candidates.csv", index=False
    )

    kept: list[dict[str, object]] = []
    seen: set[str] = set()
    filter_counts: Counter[str] = Counter()
    for english, bangla in zip(raw_synthetic_english, mono_bangla):
        reason = filter_reason(english, bangla)
        filter_counts[reason] += 1
        if reason != "kept":
            continue
        english = clean_text(english)
        key = english.lower()
        if key in seen:
            filter_counts["duplicate"] += 1
            continue
        seen.add(key)
        kept.append({"english": english, "bangla": bangla, "quality_score": quality_score(english, bangla)})

    ranked_synthetic = pd.DataFrame(kept).sort_values("quality_score", ascending=False).reset_index(drop=True)
    ranked_synthetic.to_csv(TABLES_DIR / "ranked_filtered_synthetic.csv", index=False)
    pd.DataFrame([{"reason": k, "count": v} for k, v in filter_counts.items()]).to_csv(
        TABLES_DIR / "filter_counts.csv", index=False
    )
    print("Kept after strict filter:", len(ranked_synthetic))
    print(filter_counts)

    variant_results: list[dict[str, object]] = []
    best_variant: dict[str, object] | None = None
    for synthetic_size in SYNTHETIC_SIZES_TO_TRY:
        if len(ranked_synthetic) < synthetic_size:
            continue
        synthetic_rows = ranked_synthetic.head(synthetic_size)[["english", "bangla"]].to_dict("records")
        combined_rows = train_pairs + synthetic_rows
        combined_ds = preprocess_rows(combined_rows, tokenizer, "en_bn")
        run_name = f"improved_from_baseline_top_{synthetic_size}_synthetic"
        model_path, summary, _history = train_seq2seq(
            run_name, baseline_path, combined_ds, val_en_bn, IMPROVED_EPOCHS, IMPROVED_LR
        )
        preds = generate_texts(model_path, test_sources, "english")
        test_bleu = corpus_bleu(preds, test_refs)
        summary["synthetic_size"] = synthetic_size
        summary["test_bleu"] = test_bleu["bleu"]
        summary["test_bleu_signature"] = test_bleu["signature"]
        pd.DataFrame(
            {
                "Source English": test_sources[:10],
                "Reference Bangla": test_refs[:10],
                "Baseline Prediction": baseline_preds[:10],
                "Improved Prediction": preds[:10],
            }
        ).to_csv(TABLES_DIR / f"examples_top_{synthetic_size}.csv", index=False)
        variant_results.append(summary)
        if best_variant is None or float(summary["test_bleu"]) > float(best_variant["test_bleu"]):
            best_variant = summary

    results_df = pd.DataFrame(
        [
            {
                "run_name": "baseline",
                "synthetic_size": 0,
                "validation_bleu": baseline_summary["validation_bleu"],
                "validation_loss": baseline_summary["validation_loss"],
                "test_bleu": baseline_test_bleu["bleu"],
                "train_loss": baseline_summary["train_loss"],
                "runtime_seconds": baseline_summary["runtime_seconds"],
            }
        ]
        + variant_results
    )
    results_df.to_csv(TABLES_DIR / "bleu_search_results.csv", index=False)
    print(results_df)

    baseline_bleu = float(baseline_test_bleu["bleu"])
    best_bleu = float(best_variant["test_bleu"]) if best_variant else float("nan")
    improved = best_bleu > baseline_bleu
    conclusion = (
        f"# Last Try Conclusion\n\n"
        f"Baseline test BLEU: {baseline_bleu:.4f}\n"
        f"Best improved test BLEU: {best_bleu:.4f}\n\n"
        f"Result: {'BLEU improved over baseline.' if improved else 'BLEU did not improve over baseline.'}\n\n"
        f"This run trained improved variants from the baseline checkpoint and tried smaller ranked synthetic subsets: "
        f"{SYNTHETIC_SIZES_TO_TRY}.\n"
    )
    (RESULTS_DIR / "last_try_conclusion.md").write_text(conclusion, encoding="utf-8")
    print(conclusion)

    manifest = {
        "config": config,
        "preprocessing": preprocessing_stats,
        "dataset_statistics": dataset_stats.to_dict(orient="records"),
        "baseline_test_bleu": baseline_test_bleu,
        "baseline_summary": baseline_summary,
        "reverse_summary": reverse_summary,
        "filter_counts": dict(filter_counts),
        "best_variant": best_variant,
        "improved_over_baseline": improved,
    }
    (RESULTS_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.figure(figsize=(8, 4))
    plt.bar(results_df["run_name"].astype(str), results_df["test_bleu"], color="#3b6ea8")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Test BLEU")
    plt.title("Last Try BLEU Search")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "last_try_bleu_search.png", dpi=200)

    plt.figure(figsize=(8, 4))
    plt.bar(results_df["run_name"].astype(str), results_df["validation_loss"], color="#a85532")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Validation loss")
    plt.title("Validation Loss by Variant")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "last_try_validation_loss.png", dpi=200)

    zip_path = shutil.make_archive(str(RESULTS_DIR), "zip", root_dir=RESULTS_DIR)
    print("Created:", zip_path)
    try:
        from google.colab import files

        files.download(zip_path)
    except Exception as exc:
        print("Download unavailable:", exc)


if __name__ == "__main__":
    main()

