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

import re
import pandas as pd

from cbr_monetary_policy_nlp.data_io import ensure_dir, read_jsonl, save_table, write_jsonl
from cbr_monetary_policy_nlp.text_cleaning import clean_text, make_text_norm

WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+")
DIGIT_RE = re.compile(r"\d")
RU_RE = re.compile(r"[А-Яа-яЁё]")
PAGE_RE = re.compile(r"\[PAGE\s+\d+\]", re.IGNORECASE)


def remove_page_header(text: str) -> str:
    text = clean_text(text)
    text = re.sub(r"^\[PAGE\s+\d+\]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Банк России\s+Годовой отчет за\s+\d{4}\s+год\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^Основные направления.*?денежно-кредитной политики.*?\s+", "", text, flags=re.IGNORECASE)
    return clean_text(text)


def is_boilerplate(text: str) -> bool:
    tl = text.lower()
    patterns = [
        "официальный сайт банка россии", "при использовании материала ссылка", "страница была полезной",
        "утвержден советом директоров банка россии", "электронная версия", "107016, москва",
    ]
    return len(text) < 700 and any(p in tl for p in patterns)


def is_toc_or_table(text: str) -> bool:
    tl = text.lower()
    dot_leaders = len(re.findall(r"\.{5,}\s*\d{1,4}", text))
    numeric_chunks = len(re.findall(r"(?<!\w)-?\d+[,.]?\d*", text))
    digit_share = len(DIGIT_RE.findall(text)) / max(len(text), 1)
    cyr_share = len(RU_RE.findall(text)) / max(len(text), 1)
    if "оглавление" in tl or "содержание" in tl or dot_leaders >= 2:
        return True
    if numeric_chunks >= 25 and digit_share > 0.18 and cyr_share < 0.55:
        return True
    return False


def clean_paragraph(text: str) -> str:
    text = remove_page_header(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return clean_text(text)


def keep_paragraph(text: str, min_words: int = 20) -> bool:
    if not text or is_boilerplate(text) or is_toc_or_table(text):
        return False
    words = WORD_RE.findall(text)
    if len(words) < min_words:
        return False
    cyr_share = len(RU_RE.findall(text)) / max(len(text), 1)
    return cyr_share >= 0.35


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dapt-dir", type=Path, default=Path("data/dapt_corpus"))
    args = parser.parse_args()

    dapt_dir = args.dapt_dir
    clean_dir = ensure_dir(dapt_dir / "cleaned")
    docs = read_jsonl(dapt_dir / "cbr_long_documents_dapt_docs.jsonl")
    paragraphs = read_jsonl(dapt_dir / "cbr_long_documents_dapt_paragraphs.jsonl")

    if "file_sha1" in docs.columns:
        docs = docs.drop_duplicates(subset=["file_sha1"], keep="first")
    kept_doc_ids = set(docs["doc_id"])
    paragraphs = paragraphs[paragraphs["doc_id"].isin(kept_doc_ids)].copy()
    paragraphs["text_clean"] = paragraphs["text"].apply(clean_paragraph)
    paragraphs["text_norm"] = paragraphs["text_clean"].apply(make_text_norm)
    paragraphs = paragraphs[paragraphs["text_clean"].apply(keep_paragraph)].copy()
    paragraphs = paragraphs.drop_duplicates(subset=["text_norm"]).reset_index(drop=True)

    cleaned = paragraphs[["doc_id", "paragraph_id", "text_clean"]].rename(columns={"text_clean": "text"})
    save_table(docs, clean_dir / "dapt_docs_clean.csv")
    save_table(cleaned, clean_dir / "dapt_paragraphs_clean.csv")
    write_jsonl(cleaned.to_dict("records"), clean_dir / "dapt_paragraphs_clean.jsonl")

    txt_path = clean_dir / "dapt_corpus_for_mlm.txt"
    txt_path.write_text("\n".join(cleaned["text"].astype(str).tolist()), encoding="utf-8")

    print({"n_documents": len(docs), "n_clean_paragraphs": len(cleaned), "txt_path": str(txt_path)})


if __name__ == "__main__":
    main()
