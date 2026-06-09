from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import math

from datasets import DatasetDict, load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed

from cbr_monetary_policy_nlp.config import SEED
from cbr_monetary_policy_nlp.data_io import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage_07_dapt_rubert"))
    parser.add_argument("--base-model", type=str, default="DeepPavlov/rubert-base-cased")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=2.0)
    args = parser.parse_args()

    set_seed(SEED)
    output_dir = ensure_dir(args.output_dir)

    raw_dataset = load_dataset("text", data_files={"train": str(args.text_path)})
    raw_dataset = raw_dataset.filter(lambda x: x["text"] is not None and len(x["text"].strip()) >= 80)
    split_dataset = raw_dataset["train"].train_test_split(test_size=0.05, seed=SEED)
    datasets = DatasetDict({"train": split_dataset["train"], "validation": split_dataset["test"]})

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)

    def tokenize(batch):
        return tokenizer(batch["text"], return_special_tokens_mask=True, truncation=False)

    tokenized = datasets.map(tokenize, batched=True, num_proc=2, remove_columns=["text"])

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = (len(concatenated["input_ids"]) // args.max_seq_length) * args.max_seq_length
        result = {k: [t[i:i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)] for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(group_texts, batched=True, batch_size=1000, num_proc=2)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=str(output_dir), eval_strategy="steps", eval_steps=500, save_strategy="steps", save_steps=500,
        logging_steps=100, learning_rate=5e-5, weight_decay=0.01, warmup_ratio=0.06, num_train_epochs=args.epochs,
        per_device_train_batch_size=8, per_device_eval_batch_size=8, gradient_accumulation_steps=4,
        fp16=True, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, report_to="none", seed=SEED,
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=lm_datasets["train"], eval_dataset=lm_datasets["validation"], data_collator=collator)
    before = trainer.evaluate()
    train_result = trainer.train()
    after = trainer.evaluate()

    final_dir = ensure_dir(output_dir / "final_model")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = {
        "base_model": args.base_model,
        "max_seq_length": args.max_seq_length,
        "train_chunks": len(lm_datasets["train"]),
        "validation_chunks": len(lm_datasets["validation"]),
        "eval_before_loss": before["eval_loss"],
        "eval_before_perplexity": math.exp(before["eval_loss"]) if before["eval_loss"] < 20 else float("inf"),
        "eval_after_loss": after["eval_loss"],
        "eval_after_perplexity": math.exp(after["eval_loss"]) if after["eval_loss"] < 20 else float("inf"),
        "train_result": str(train_result),
        "final_model_dir": str(final_dir),
    }
    write_json(metrics, output_dir / "dapt_training_metrics.json")


if __name__ == "__main__":
    main()
