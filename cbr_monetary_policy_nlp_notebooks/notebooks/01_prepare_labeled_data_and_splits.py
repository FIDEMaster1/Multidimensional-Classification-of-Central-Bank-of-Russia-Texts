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

from cbr_monetary_policy_nlp.config import SEED, ProjectPaths, seed_everything
from cbr_monetary_policy_nlp.data_io import read_json_records, save_table, write_json
from cbr_monetary_policy_nlp.labels import LABEL_SCHEMA, TASKS, add_label_ids, save_label_schema
from cbr_monetary_policy_nlp.splitting import (
    add_synthetic_to_train,
    make_document_split,
    make_sentence_split,
    time_holdout_split,
    validate_split_df,
)
from cbr_monetary_policy_nlp.text_cleaning import (
    clean_label,
    clean_text,
    drop_empty_texts,
    is_boilerplate_candidate,
    parse_russian_date,
)


def load_manual_data(path: Path) -> pd.DataFrame:
    records = read_json_records(path)
    df = pd.DataFrame(records).copy()
    required = ["doc_id", "section", "category", "title", "published_at", "sentence", *TASKS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Manual data missing columns: {missing}")

    df["text"] = df["sentence"].apply(clean_text)
    df["source"] = "manual"
    df["is_synthetic"] = False
    df["synthetic_type"] = None
    df["sample_id"] = [f"manual_{i:06d}" for i in range(len(df))]

    for col in TASKS:
        df[col] = df[col].apply(clean_label)
    for col in ["doc_id", "section", "category", "title", "published_at"]:
        df[col] = df[col].apply(clean_text)

    columns = [
        "sample_id", "source", "is_synthetic", "synthetic_type", "doc_id", "section", "category",
        "title", "published_at", "text", *TASKS,
    ]
    return df[columns]


def load_synthetic_data(path: Path) -> pd.DataFrame:
    records = read_json_records(path)
    df = pd.DataFrame(records).copy()
    required = ["text", *TASKS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Synthetic data missing columns: {missing}")

    df["text"] = df["text"].apply(clean_text)
    for col in TASKS:
        df[col] = df[col].apply(clean_label)

    df["source"] = "synthetic"
    df["is_synthetic"] = True
    df["synthetic_type"] = df.get("synthetic_type", "rare_joint")
    df["doc_id"] = None
    df["section"] = None
    df["category"] = "synthetic"
    df["title"] = None
    df["published_at"] = None
    df["sample_id"] = [f"synthetic_{i:06d}" for i in range(len(df))]

    columns = [
        "sample_id", "source", "is_synthetic", "synthetic_type", "doc_id", "section", "category",
        "title", "published_at", "text", *TASKS,
    ]
    return df[columns]


def validate_labels(df: pd.DataFrame, name: str) -> None:
    for task in TASKS:
        unknown = sorted(set(df[task].dropna().unique()) - set(LABEL_SCHEMA[task]))
        if unknown:
            raise ValueError(f"{name}: unknown labels in {task}: {unknown}")


def save_distribution_tables(df: pd.DataFrame, name: str, output_dir: Path) -> None:
    for col in [*TASKS, "source", "category", "year"]:
        if col not in df.columns:
            continue
        table = (
            df[col].fillna("NA").astype(str).value_counts().rename_axis(col).reset_index(name="count")
        )
        table["share_pct"] = (table["count"] / len(df) * 100).round(2)
        save_table(table, output_dir / f"{name}_distribution_{col}.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-path", type=Path, required=True)
    parser.add_argument("--synthetic-path", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path("cbr_monetary_policy_nlp"))
    args = parser.parse_args()

    seed_everything(SEED)
    paths = ProjectPaths.from_root(args.project_dir)
    paths.make_dirs()
    (paths.root / "outputs" / "tables").mkdir(parents=True, exist_ok=True)

    manual_df = load_manual_data(args.manual_path)
    synthetic_df = load_synthetic_data(args.synthetic_path)

    validate_labels(manual_df, "manual")
    validate_labels(synthetic_df, "synthetic")

    manual_df = add_label_ids(drop_empty_texts(manual_df))
    synthetic_df = add_label_ids(drop_empty_texts(synthetic_df))

    manual_df["date"] = manual_df["published_at"].apply(parse_russian_date)
    manual_df["year"] = manual_df["date"].dt.year
    manual_df["is_boilerplate_candidate"] = manual_df["text"].apply(is_boilerplate_candidate)
    synthetic_df["date"] = pd.NaT
    synthetic_df["year"] = pd.NA
    synthetic_df["is_boilerplate_candidate"] = False

    combined_df = pd.concat([manual_df, synthetic_df], ignore_index=True)

    save_label_schema(paths.data)
    save_table(manual_df, paths.data / "manual_clean_encoded.csv")
    save_table(synthetic_df, paths.data / "synthetic_clean_encoded.csv")
    save_table(combined_df, paths.data / "combined_clean_encoded.csv")

    tables_dir = paths.root / "outputs" / "tables"
    save_distribution_tables(manual_df, "manual", tables_dir)
    save_distribution_tables(synthetic_df, "synthetic", tables_dir)
    save_distribution_tables(combined_df, "combined", tables_dir)

    main_doc_manual = make_document_split(manual_df, doc_col="doc_id", seed=SEED)
    main_doc_with_synthetic = add_synthetic_to_train(main_doc_manual, synthetic_df)
    sentence_diagnostic = make_sentence_split(manual_df, stratify_col="sentiment", seed=SEED)
    time_holdout = time_holdout_split(manual_df, date_col="date")

    split_tables = {
        "main_doc_split_manual.csv": main_doc_manual,
        "main_doc_split_combined_with_synthetic_train.csv": main_doc_with_synthetic,
        "sentence_diagnostic_split_manual.csv": sentence_diagnostic,
        "time_holdout_split_manual.csv": time_holdout,
    }

    for filename, split_df in split_tables.items():
        validate_split_df(split_df, check_document_leakage="sentence_diagnostic" not in filename)
        save_table(split_df, paths.splits / filename)

    write_json(
        {
            "seed": SEED,
            "n_manual": len(manual_df),
            "n_synthetic": len(synthetic_df),
            "label_schema": LABEL_SCHEMA,
            "splits": list(split_tables.keys()),
            "note": "Synthetic examples are added only to train in the combined split.",
        },
        paths.configs / "data_preparation_config.json",
    )


if __name__ == "__main__":
    main()
