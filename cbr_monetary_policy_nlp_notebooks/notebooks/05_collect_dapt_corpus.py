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

import hashlib
import mimetypes
import re
import time
import zipfile
from urllib.parse import unquote, urljoin, urlparse

import fitz
import pandas as pd
import requests
from bs4 import BeautifulSoup
from docx import Document

from cbr_monetary_policy_nlp.data_io import ensure_dir, save_table, write_jsonl
from cbr_monetary_policy_nlp.text_cleaning import clean_text

CBR_BASE = "https://www.cbr.ru"
SOURCE_PAGES = {
    "annual_report": "https://www.cbr.ru/about_br/publ/god/",
    "monetary_policy_guidelines": "https://www.cbr.ru/about_br/publ/ondkp/",
}
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8"}


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_filename(value: str, max_len: int = 150) -> str:
    value = unquote(str(value)).replace("\\", "_").replace("/", "_")
    value = re.sub(r"[^\w\-.а-яА-ЯёЁ]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value[:max_len]


def normalize_url(href: str, base_url: str) -> str:
    return urljoin(base_url, str(href or "").strip().replace("\\", "/")).replace("http://www.cbr.ru", CBR_BASE)


def classify_url(url: str) -> str:
    lower = url.lower()
    for ext in ["pdf", "docx", "doc", "zip"]:
        if f".{ext}" in lower:
            return ext
    return "html"


def request_get(session: requests.Session, url: str, retries: int = 3) -> requests.Response:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(url, timeout=90, allow_redirects=True)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(attempt)
    raise RuntimeError(f"Failed to GET {url}: {last_error!r}")


def collect_candidate_links(session: requests.Session) -> pd.DataFrame:
    rows = []
    for source_type, source_page in SOURCE_PAGES.items():
        soup = BeautifulSoup(request_get(session, source_page).text, "html.parser")
        for a in soup.find_all("a"):
            href = a.get("href")
            text = clean_text(a.get_text(" "))
            url = normalize_url(href, source_page)
            if not url.startswith(CBR_BASE):
                continue
            if any(x in url.lower() for x in [".pdf", ".doc", ".docx", ".zip", "/collection/collection/file/", "/content/document/file/"]) or text:
                rows.append({"source_type": source_type, "source_page": source_page, "landing_url": url, "anchor_text": text, "url_type": classify_url(url)})
    return pd.DataFrame(rows).drop_duplicates(subset=["source_type", "landing_url"]).reset_index(drop=True)


def resolve_download_links(session: requests.Session, links_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in links_df.to_dict("records"):
        if row["url_type"] in {"pdf", "doc", "docx", "zip"}:
            rows.append({**row, "download_url": row["landing_url"], "title": row["anchor_text"], "file_type": row["url_type"]})
            continue

        try:
            soup = BeautifulSoup(request_get(session, row["landing_url"]).text, "html.parser")
        except Exception:
            continue

        title = clean_text(soup.find("h1").get_text(" ")) if soup.find("h1") else row["anchor_text"]
        for a in soup.find_all("a"):
            url = normalize_url(a.get("href"), row["landing_url"])
            file_type = classify_url(url)
            if file_type in {"pdf", "doc", "docx", "zip"}:
                rows.append({**row, "download_url": url, "title": title, "file_type": file_type})
    return pd.DataFrame(rows).drop_duplicates(subset=["source_type", "download_url"]).reset_index(drop=True)


def download_files(session: requests.Session, links_df: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    rows = []
    for i, row in enumerate(links_df.to_dict("records")):
        try:
            response = request_get(session, row["download_url"])
            suffix = Path(urlparse(response.url).path).suffix.lower() or mimetypes.guess_extension(response.headers.get("Content-Type", "").split(";")[0]) or ".bin"
            filename = safe_filename(f"{row['source_type']}__{i:04d}__{Path(urlparse(response.url).path).name or 'file'}")
            if not Path(filename).suffix:
                filename += suffix
            path = raw_dir / filename
            path.write_bytes(response.content)
            rows.append({**row, "download_status": "ok", "file_path": str(path), "file_name": path.name, "file_extension": path.suffix.lower(), "file_sha1": sha1_file(path), "file_size_bytes": path.stat().st_size})
        except Exception as exc:
            rows.append({**row, "download_status": "failed", "error": repr(exc)})
    return pd.DataFrame(rows)


def extract_pdf(path: Path) -> str:
    parts = []
    with fitz.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                parts.append(f"[PAGE {page_no}] {text}")
    return "\n\n".join(parts)


def extract_docx(path: Path) -> str:
    doc = Document(path)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def split_paragraphs(text: str) -> list[str]:
    return [clean_text(x) for x in re.split(r"\n{2,}", text) if len(clean_text(x)) >= 80]


def build_text_corpus(downloads_df: pd.DataFrame, output_dir: Path) -> None:
    docs = []
    paragraphs = []
    for row in downloads_df[downloads_df["download_status"] == "ok"].to_dict("records"):
        path = Path(row["file_path"])
        try:
            if path.suffix.lower() == ".pdf":
                text = extract_pdf(path)
                method = "pymupdf"
            elif path.suffix.lower() == ".docx":
                text = extract_docx(path)
                method = "python-docx"
            elif path.suffix.lower() == ".zip":
                continue
            else:
                continue

            doc_id = row["file_sha1"]
            docs.append({**row, "doc_id": doc_id, "extraction_method": method, "n_chars": len(text)})
            for i, paragraph in enumerate(split_paragraphs(text)):
                paragraphs.append({"doc_id": doc_id, "paragraph_id": f"{doc_id}_{i:06d}", "text": paragraph})
        except Exception as exc:
            docs.append({**row, "doc_id": row.get("file_sha1", ""), "extraction_method": "failed", "error": repr(exc)})

    write_jsonl(docs, output_dir / "cbr_long_documents_dapt_docs.jsonl")
    write_jsonl(paragraphs, output_dir / "cbr_long_documents_dapt_paragraphs.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/dapt_corpus"))
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    raw_dir = ensure_dir(output_dir / "raw_files")
    session = requests.Session()
    session.headers.update(HEADERS)

    candidates = collect_candidate_links(session)
    save_table(candidates, output_dir / "candidate_links.csv")
    resolved = resolve_download_links(session, candidates)
    save_table(resolved, output_dir / "resolved_download_links.csv")
    downloads = download_files(session, resolved, raw_dir)
    save_table(downloads, output_dir / "download_manifest.csv")
    build_text_corpus(downloads, output_dir)


if __name__ == "__main__":
    main()
