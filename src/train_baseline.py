"""Fine-tune baseline mT5-small for English-to-Bangla translation.

This script is a compact reproducible entry point. The Colab notebooks in
`notebooks/` contain the exact measured experiment runs used in the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def tokenize(rows: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(rows, preserve_index=False)

    def preprocess(batch):
        source = [f"translate English to Bengali: {x}" for x in batch["english"]]
        model_inputs = tokenizer(source, max_length=max_length, truncation=True)
        labels = tokenizer(text_target=batch["bangla"], max_length=max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return ds.map(preprocess, batched=True, remove_columns=ds.column_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline English-to-Bangla mT5-small.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/baseline"))
    parser.add_argument("--model-name", default="google/mt5-small")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    train_ds = tokenize(pd.read_csv(args.train), tokenizer, args.max_length)
    val_ds = tokenize(pd.read_csv(args.validation), tokenizer, args.max_length)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        adafactor=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
        fp16=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

