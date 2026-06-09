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

import joblib
import pandas as pd

from cbr_monetary_policy_nlp.classical_models import get_classical_models, run_classical_experiment
from cbr_monetary_policy_nlp.config import SEED, seed_everything
from cbr_monetary_policy_nlp.data_io import ensure_dir, save_table, write_json
from cbr_monetary_policy_nlp.evaluation import prediction_table, save_classification_report, save_confusion_matrix
from cbr_monetary_policy_nlp.labels import ID2LABEL, TASKS
from cbr_monetary_policy_nlp.splitting import validate_split_df


def evaluate_and_save_predictions(split_df: pd.DataFrame, fitted_models: dict, output_dir: Path) -> None:
    pred_dir = ensure_dir(output_dir / "predictions")
    report_dir = ensure_dir(output_dir / "reports")
    cm_dir = ensure_dir(output_dir / "confusion_matrices")

    for (task, model_name), model in fitted_models.items():
        for eval_split in ["validation", "test"]:
            eval_df = split_df[split_df["split"] == eval_split].copy()
            if eval_df.empty:
                continue
            y_true = eval_df[f"{task}_id"].astype(int).values
            y_pred = model.predict(eval_df["text"].astype(str).tolist())
            labels = list(range(len(ID2LABEL[task])))
            class_names = [ID2LABEL[task][i] for i in labels]

            report = save_classification_report(
                y_true, y_pred, labels, class_names, report_dir / f"{model_name}__{task}__{eval_split}_report.json"
            )
            save_confusion_matrix(
                y_true, y_pred, labels, class_names, cm_dir / f"{model_name}__{task}__{eval_split}_cm.csv"
            )
            pred = prediction_table(
                eval_df, y_true, y_pred, task=task, id2label=ID2LABEL[task], metadata={"model": model_name, "eval_split": eval_split}
            )
            save_table(pred, pred_dir / f"{model_name}__{task}__{eval_split}_predictions.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("cbr_monetary_policy_nlp"))
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()

    seed_everything(SEED)
    project_dir = args.project_dir
    split_dir = project_dir / "splits"
    stage_dir = ensure_dir(project_dir / "outputs" / "stage_02_classical_baselines")
    model_dir = ensure_dir(stage_dir / "models")

    split_files = {
        "main_doc_manual": "main_doc_split_manual.csv",
        "main_doc_with_synthetic": "main_doc_split_combined_with_synthetic_train.csv",
        "sentence_diagnostic": "sentence_diagnostic_split_manual.csv",
        "time_holdout": "time_holdout_split_manual.csv",
    }

    all_metrics = []
    models = get_classical_models(SEED)

    for split_name, filename in split_files.items():
        path = split_dir / filename
        if not path.exists():
            continue

        split_df = pd.read_csv(path)
        validate_split_df(split_df, check_document_leakage=split_name != "sentence_diagnostic")
        metrics_df, fitted_models = run_classical_experiment(split_df, models=models, tasks=TASKS, seed=SEED)
        metrics_df["split_name"] = split_name
        metrics_df["dataset_variant"] = "manual_plus_synthetic_train" if "synthetic" in set(split_df["source"]) else "manual_only"
        all_metrics.append(metrics_df)

        split_output_dir = ensure_dir(stage_dir / split_name)
        save_table(metrics_df, split_output_dir / "metrics_long.csv")
        evaluate_and_save_predictions(split_df, fitted_models, split_output_dir)

        if args.save_models:
            for key, model in fitted_models.items():
                task, model_name = key
                joblib.dump(model, model_dir / f"{split_name}__{task}__{model_name}.joblib")

    combined = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    save_table(combined, stage_dir / "classical_baselines_metrics_long.csv")
    write_json({"seed": SEED, "models": list(models.keys()), "tasks": TASKS}, stage_dir / "classical_baselines_config.json")


if __name__ == "__main__":
    main()
