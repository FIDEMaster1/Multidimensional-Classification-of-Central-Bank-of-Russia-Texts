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
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, Trainer

from cbr_monetary_policy_nlp.config import SEED, seed_everything
from cbr_monetary_policy_nlp.data_io import ensure_dir, save_table, write_json
from cbr_monetary_policy_nlp.evaluation import prediction_table, save_classification_report, save_confusion_matrix
from cbr_monetary_policy_nlp.labels import ID2LABEL, NUM_LABELS, TASKS
from cbr_monetary_policy_nlp.training import BestStateDictCallback, compute_class_weights, make_compute_metrics, make_single_task_dataset, make_training_args, softmax_np
from cbr_monetary_policy_nlp.transformer_models import SingleTaskTransformerClassifier


def save_model(model: SingleTaskTransformerClassifier, tokenizer, task: str, cfg: dict, output_dir: Path) -> None:
    model_dir = ensure_dir(output_dir / "models" / task)
    tokenizer.save_pretrained(model_dir / "tokenizer")
    model.encoder.save_pretrained(model_dir / "encoder")
    torch.save(model.state_dict(), model_dir / "custom_single_task_classifier_state_dict.pt")
    write_json({"task": task, "cfg": cfg, "id2label": {str(k): v for k, v in ID2LABEL[task].items()}}, model_dir / "final_model_config.json")


def train_task(split_df: pd.DataFrame, task: str, cfg: dict, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    ds, train_df, val_df, test_df = make_single_task_dataset(split_df, task, tokenizer, cfg["max_length"])
    weights = compute_class_weights(train_df[f"{task}_id"].values, NUM_LABELS[task])
    model = SingleTaskTransformerClassifier(cfg["model_name"], NUM_LABELS[task], cfg["pooling"], cfg["dropout"], weights)

    run_name = f"final__{task}__{cfg['encoder_name']}"
    args = make_training_args(output_dir / "trainer_tmp" / task / run_name, cfg, run_name, seed=SEED)
    best = BestStateDictCallback("eval_f1_macro", patience=cfg.get("early_stopping_patience", 2))
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"], compute_metrics=make_compute_metrics(), callbacks=[best])
    trainer.train()
    if best.best_state_dict is not None:
        trainer.model.load_state_dict(best.best_state_dict)

    metrics = []
    predictions = []
    for eval_split, dataset, frame in [("train", ds["train"], train_df), ("validation", ds["validation"], val_df), ("test", ds["test"], test_df)]:
        output = trainer.predict(dataset)
        logits = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        probs = softmax_np(logits)
        y_true = output.label_ids
        y_pred = probs.argmax(axis=1)
        pred = prediction_table(frame, y_true, y_pred, task, ID2LABEL[task], probs, {"eval_split": eval_split, "encoder_name": cfg["encoder_name"]})
        predictions.append(pred)
        metrics.append({
            "task": task, "eval_split": eval_split, "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "best_validation_f1_macro": best.best_metric, "best_epoch": best.best_epoch,
            "n_train": len(train_df), "n_train_synthetic": int((train_df["source"] == "synthetic").sum()),
        })

        labels = list(range(NUM_LABELS[task]))
        class_names = [ID2LABEL[task][i] for i in labels]
        save_classification_report(y_true, y_pred, labels, class_names, output_dir / "reports" / task / f"{eval_split}_report.json")
        save_confusion_matrix(y_true, y_pred, labels, class_names, output_dir / "confusion_matrices" / task / f"{eval_split}_cm.csv")
        save_table(pred, output_dir / "predictions" / task / f"{eval_split}_predictions_with_probs.csv")

    save_model(trainer.model, tokenizer, task, cfg, output_dir)
    del trainer, model, tokenizer, ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("cbr_monetary_policy_nlp"))
    parser.add_argument("--split-file", type=str, default="main_doc_split_combined_with_synthetic_train.csv")
    parser.add_argument("--model-name", type=str, default="DeepPavlov/rubert-base-cased")
    args = parser.parse_args()

    seed_everything(SEED)
    split_df = pd.read_csv(args.project_dir / "splits" / args.split_file)
    output_dir = ensure_dir(args.project_dir / "outputs" / "stage_08_final_rubert_models")

    base_cfg = {
        "model_name": args.model_name, "encoder_name": "ruBERT_final", "pooling": "cls", "max_length": 128,
        "batch_size": 8, "eval_batch_size": 16, "learning_rate": 2e-5, "epochs": 4,
        "weight_decay": 0.01, "warmup_ratio": 0.1, "dropout": 0.2, "gradient_accumulation_steps": 1,
        "early_stopping_patience": 2,
    }

    all_metrics = []
    all_predictions = []
    for task in ["sentiment", "topic", "stance"]:
        cfg = {**base_cfg, "encoder_name": f"ruBERT_final_{task}"}
        metrics, predictions = train_task(split_df, task, cfg, output_dir)
        all_metrics.append(metrics)
        all_predictions.append(predictions)

    save_table(pd.concat(all_metrics, ignore_index=True), output_dir / "final_rubert_metrics_all_tasks.csv")
    save_table(pd.concat(all_predictions, ignore_index=True), output_dir / "final_rubert_predictions_all_tasks.csv")
    write_json({"seed": SEED, "tasks": TASKS, "base_cfg": base_cfg}, output_dir / "final_rubert_config.json")


if __name__ == "__main__":
    main()
