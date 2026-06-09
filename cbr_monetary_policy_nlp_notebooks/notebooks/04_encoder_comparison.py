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

import gc
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, Trainer

from cbr_monetary_policy_nlp.config import SEED, seed_everything
from cbr_monetary_policy_nlp.data_io import ensure_dir, save_table, write_json
from cbr_monetary_policy_nlp.evaluation import prediction_table
from cbr_monetary_policy_nlp.labels import ID2LABEL, NUM_LABELS
from cbr_monetary_policy_nlp.training import (
    BestStateDictCallback,
    compute_class_weights,
    make_compute_metrics,
    make_single_task_dataset,
    make_training_args,
    softmax_np,
)
from cbr_monetary_policy_nlp.transformer_models import SingleTaskTransformerClassifier


ENCODERS = [
    {"encoder_name": "ruBERT", "model_name": "DeepPavlov/rubert-base-cased", "pooling": "cls", "batch_size": 8, "eval_batch_size": 16, "learning_rate": 2e-5, "epochs": 4},
    {"encoder_name": "XLM-R-base", "model_name": "xlm-roberta-base", "pooling": "cls", "batch_size": 8, "eval_batch_size": 16, "learning_rate": 2e-5, "epochs": 4},
    {"encoder_name": "MiniLM", "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "pooling": "mean", "batch_size": 16, "eval_batch_size": 32, "learning_rate": 2e-5, "epochs": 4},
    {"encoder_name": "rubert-tiny2", "model_name": "cointegrated/rubert-tiny2", "pooling": "cls", "batch_size": 16, "eval_batch_size": 32, "learning_rate": 3e-5, "epochs": 5},
]


def run_encoder_task(split_df: pd.DataFrame, cfg: dict, task: str, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = {"max_length": 128, "weight_decay": 0.01, "warmup_ratio": 0.1, "dropout": 0.2, "gradient_accumulation_steps": 1, "early_stopping_patience": 2, **cfg}
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    ds, train_df, val_df, test_df = make_single_task_dataset(split_df, task, tokenizer, cfg["max_length"])
    weights = compute_class_weights(train_df[f"{task}_id"].astype(int).values, NUM_LABELS[task])
    model = SingleTaskTransformerClassifier(cfg["model_name"], NUM_LABELS[task], cfg["pooling"], cfg["dropout"], weights)

    run_name = f"{cfg['encoder_name']}__{cfg['pooling']}__{task}"
    args = make_training_args(output_dir / "trainer_tmp" / run_name, cfg, run_name, seed=SEED)
    best = BestStateDictCallback("eval_f1_macro", patience=cfg["early_stopping_patience"])
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"], compute_metrics=make_compute_metrics(), callbacks=[best])
    trainer.train()
    if best.best_state_dict is not None:
        trainer.model.load_state_dict(best.best_state_dict)

    rows = []
    predictions = []
    for eval_split, dataset, frame in [("validation", ds["validation"], val_df), ("test", ds["test"], test_df)]:
        output = trainer.predict(dataset)
        logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        probs = softmax_np(logits)
        pred = probs.argmax(axis=1)
        labels = output.label_ids
        rows.append({
            "encoder_name": cfg["encoder_name"], "model_name": cfg["model_name"], "pooling": cfg["pooling"],
            "task": task, "eval_split": eval_split, "accuracy": accuracy_score(labels, pred),
            "f1_macro": f1_score(labels, pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, pred, average="weighted", zero_division=0),
            "best_validation_f1_macro": best.best_metric, "best_epoch": best.best_epoch, "status": "ok",
        })
        predictions.append(prediction_table(frame, labels, pred, task, ID2LABEL[task], probs, {"encoder_name": cfg["encoder_name"], "eval_split": eval_split, "pooling": cfg["pooling"]}))

    del trainer, model, tokenizer, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows), pd.concat(predictions, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("cbr_monetary_policy_nlp"))
    parser.add_argument("--split-file", type=str, default="main_doc_split_combined_with_synthetic_train.csv")
    parser.add_argument("--tasks", nargs="+", default=["sentiment", "topic", "stance"])
    args = parser.parse_args()

    seed_everything(SEED)
    split_df = pd.read_csv(args.project_dir / "splits" / args.split_file)
    output_dir = ensure_dir(args.project_dir / "outputs" / "stage_04_encoder_comparison")

    metric_tables = []
    pred_tables = []
    for cfg in ENCODERS:
        for task in args.tasks:
            metrics, preds = run_encoder_task(split_df, cfg, task, output_dir)
            metric_tables.append(metrics)
            pred_tables.append(preds)

    save_table(pd.concat(metric_tables, ignore_index=True), output_dir / "encoder_comparison_metrics_long.csv")
    save_table(pd.concat(pred_tables, ignore_index=True), output_dir / "encoder_comparison_predictions.csv")
    write_json({"seed": SEED, "encoders": ENCODERS, "tasks": args.tasks}, output_dir / "encoder_comparison_config.json")


if __name__ == "__main__":
    main()
