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
from cbr_monetary_policy_nlp.labels import ID2LABEL, NUM_LABELS, TASKS
from cbr_monetary_policy_nlp.training import (
    BestStateDictCallback,
    compute_class_weights,
    make_compute_metrics,
    make_multitask_dataset,
    make_single_task_dataset,
    make_training_args,
    softmax_np,
)
from cbr_monetary_policy_nlp.transformer_models import MultiTaskTrainerMixin, MultiTaskTransformer, SingleTaskTransformerClassifier


class MultiTaskTrainer(MultiTaskTrainerMixin, Trainer):
    pass


def evaluate_single_task(trainer: Trainer, dataset, eval_df: pd.DataFrame, task: str, metadata: dict) -> pd.DataFrame:
    output = trainer.predict(dataset)
    logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
    probs = softmax_np(logits)
    y_true = output.label_ids
    y_pred = probs.argmax(axis=1)
    return prediction_table(eval_df, y_true, y_pred, task, ID2LABEL[task], probs=probs, metadata=metadata)


def evaluate_multitask(trainer: Trainer, dataset, eval_df: pd.DataFrame, metadata: dict) -> list[pd.DataFrame]:
    output = trainer.predict(dataset)
    logits_tuple = output.predictions
    labels_tuple = output.label_ids
    tables = []
    for task, logits, labels in zip(TASKS, logits_tuple, labels_tuple):
        probs = softmax_np(logits)
        pred = probs.argmax(axis=1)
        tables.append(prediction_table(eval_df, labels, pred, task, ID2LABEL[task], probs=probs, metadata=metadata))
    return tables


def train_single_task(split_df: pd.DataFrame, cfg: dict, task: str, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    ds, train_df, val_df, test_df = make_single_task_dataset(split_df, task, tokenizer, cfg["max_length"])
    weights = compute_class_weights(train_df[f"{task}_id"].values, NUM_LABELS[task])
    model = SingleTaskTransformerClassifier(
        cfg["model_name"], NUM_LABELS[task], pooling=cfg.get("pooling", "cls"), dropout=cfg.get("dropout", 0.2), class_weights=weights
    )

    run_name = f"{cfg['encoder_name']}__single_task__{task}"
    args = make_training_args(output_dir / "trainer_tmp" / run_name, cfg, run_name, seed=SEED)
    best = BestStateDictCallback(metric_name="eval_f1_macro", patience=cfg.get("early_stopping_patience", 2))
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"], compute_metrics=make_compute_metrics(), callbacks=[best])
    trainer.train()
    if best.best_state_dict is not None:
        trainer.model.load_state_dict(best.best_state_dict)

    pred_tables = []
    rows = []
    for split_name, dataset, frame in [("validation", ds["validation"], val_df), ("test", ds["test"], test_df)]:
        pred = evaluate_single_task(
            trainer, dataset, frame, task, {"architecture": "single_task", "encoder_name": cfg["encoder_name"], "eval_split": split_name}
        )
        rows.append({
            "architecture": "single_task", "encoder_name": cfg["encoder_name"], "task": task, "eval_split": split_name,
            "accuracy": accuracy_score(pred["true_id"], pred["pred_id"]),
            "f1_macro": f1_score(pred["true_id"], pred["pred_id"], average="macro", zero_division=0),
            "f1_weighted": f1_score(pred["true_id"], pred["pred_id"], average="weighted", zero_division=0),
            "best_validation_f1_macro": best.best_metric,
            "best_epoch": best.best_epoch,
        })
        pred_tables.append(pred)

    del trainer, model, tokenizer, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows), pd.concat(pred_tables, ignore_index=True)


def train_multitask(split_df: pd.DataFrame, cfg: dict, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    ds, train_df, val_df, test_df = make_multitask_dataset(split_df, tokenizer, cfg["max_length"])
    weights = {task: compute_class_weights(train_df[f"{task}_id"].values, NUM_LABELS[task]) for task in TASKS}
    model = MultiTaskTransformer(
        cfg["model_name"], NUM_LABELS["topic"], NUM_LABELS["stance"], NUM_LABELS["sentiment"],
        pooling=cfg.get("pooling", "cls"), dropout=cfg.get("dropout", 0.2),
        topic_weights=weights["topic"], stance_weights=weights["stance"], sentiment_weights=weights["sentiment"],
    )

    run_name = f"{cfg['encoder_name']}__multitask"
    args = make_training_args(output_dir / "trainer_tmp" / run_name, cfg, run_name, seed=SEED)
    trainer = MultiTaskTrainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"])
    trainer.train()

    pred_tables = []
    rows = []
    for split_name, dataset, frame in [("validation", ds["validation"], val_df), ("test", ds["test"], test_df)]:
        tables = evaluate_multitask(trainer, dataset, frame, {"architecture": "multitask", "encoder_name": cfg["encoder_name"], "eval_split": split_name})
        for task, pred in zip(TASKS, tables):
            rows.append({
                "architecture": "multitask", "encoder_name": cfg["encoder_name"], "task": task, "eval_split": split_name,
                "accuracy": accuracy_score(pred["true_id"], pred["pred_id"]),
                "f1_macro": f1_score(pred["true_id"], pred["pred_id"], average="macro", zero_division=0),
                "f1_weighted": f1_score(pred["true_id"], pred["pred_id"], average="weighted", zero_division=0),
            })
        pred_tables.extend(tables)

    del trainer, model, tokenizer, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(rows), pd.concat(pred_tables, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("cbr_monetary_policy_nlp"))
    parser.add_argument("--split-file", type=str, default="main_doc_split_combined_with_synthetic_train.csv")
    parser.add_argument("--model-name", type=str, default="DeepPavlov/rubert-base-cased")
    parser.add_argument("--encoder-name", type=str, default="ruBERT")
    args = parser.parse_args()

    seed_everything(SEED)
    split_df = pd.read_csv(args.project_dir / "splits" / args.split_file)
    output_dir = ensure_dir(args.project_dir / "outputs" / "stage_03_transformer_single_vs_multitask")

    cfg = {
        "encoder_name": args.encoder_name, "model_name": args.model_name, "pooling": "cls", "max_length": 128,
        "batch_size": 8, "eval_batch_size": 16, "learning_rate": 2e-5, "epochs": 4,
        "weight_decay": 0.01, "warmup_ratio": 0.1, "dropout": 0.2, "gradient_accumulation_steps": 1,
        "early_stopping_patience": 2,
    }

    metric_tables = []
    prediction_tables = []
    for task in TASKS:
        metrics, preds = train_single_task(split_df, cfg, task, output_dir)
        metric_tables.append(metrics)
        prediction_tables.append(preds)

    multitask_metrics, multitask_preds = train_multitask(split_df, cfg, output_dir)
    metric_tables.append(multitask_metrics)
    prediction_tables.append(multitask_preds)

    all_metrics = pd.concat(metric_tables, ignore_index=True)
    all_predictions = pd.concat(prediction_tables, ignore_index=True)
    save_table(all_metrics, output_dir / "single_vs_multitask_metrics.csv")
    save_table(all_predictions, output_dir / "single_vs_multitask_predictions.csv")
    write_json(cfg, output_dir / "single_vs_multitask_config.json")


if __name__ == "__main__":
    main()
