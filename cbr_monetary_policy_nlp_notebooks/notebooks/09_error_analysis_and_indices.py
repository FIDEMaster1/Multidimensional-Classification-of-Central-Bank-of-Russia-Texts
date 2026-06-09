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

import pandas as pd

from cbr_monetary_policy_nlp.data_io import ensure_dir, save_table, write_json
from cbr_monetary_policy_nlp.error_analysis import save_error_analysis
from cbr_monetary_policy_nlp.indices import add_time_columns, aggregate_communication_indices


def load_prediction_tables(prediction_dir: Path) -> pd.DataFrame:
    files = list(prediction_dir.glob("**/*predictions_with_probs.csv"))
    if not files:
        files = list(prediction_dir.glob("**/*predictions*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {prediction_dir}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def build_wide_prediction_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [col for col in ["sample_id", "doc_id", "category", "published_at", "title", "text"] if col in pred_df.columns]
    keep = key_cols + ["task", "pred_label", "confidence"]
    wide = pred_df[keep].copy()
    wide = wide.pivot_table(index=key_cols, columns="task", values="pred_label", aggfunc="first").reset_index()
    return wide.rename(columns={"topic": "topic", "stance": "stance", "sentiment": "sentiment"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", type=Path, default=Path("cbr_monetary_policy_nlp"))
    parser.add_argument("--prediction-dir", type=Path, default=None)
    parser.add_argument("--group-cols", nargs="+", default=["year", "category"])
    args = parser.parse_args()

    stage_dir = ensure_dir(args.project_dir / "outputs" / "stage_09_error_analysis_and_indices")
    prediction_dir = args.prediction_dir or (args.project_dir / "outputs" / "stage_08_final_rubert_models" / "predictions")

    pred_df = load_prediction_tables(prediction_dir)
    eval_pred = pred_df[pred_df.get("eval_split", "test").isin(["validation", "test"])] if "eval_split" in pred_df.columns else pred_df
    save_error_analysis(eval_pred, stage_dir / "error_analysis")

    wide = build_wide_prediction_table(pred_df)
    if "published_at" in wide.columns:
        wide = add_time_columns(wide, date_col="published_at")

    available_group_cols = [col for col in args.group_cols if col in wide.columns]
    if available_group_cols and {"topic", "stance", "sentiment"}.issubset(wide.columns):
        indices = aggregate_communication_indices(wide, group_cols=available_group_cols)
        save_table(indices, stage_dir / "communication_indices.csv")
    else:
        indices = pd.DataFrame()

    save_table(wide, stage_dir / "sentence_predictions_wide.csv")
    write_json({"prediction_dir": str(prediction_dir), "group_cols": available_group_cols, "n_sentences": len(wide)}, stage_dir / "error_analysis_and_indices_config.json")


if __name__ == "__main__":
    main()
