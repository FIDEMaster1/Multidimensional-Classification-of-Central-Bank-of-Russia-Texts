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

import time
from dataclasses import asdict

from bs4 import BeautifulSoup

from cbr_monetary_policy_nlp.cbr_parsing import (
    CBR_BASE,
    DocItem,
    collect_html_items_from_page,
    fetch_documents,
    make_session,
    normalize_cbr_url,
    parse_ru_date_from_text,
    request_get,
    save_document_records,
)
from cbr_monetary_policy_nlp.data_io import ensure_dir, write_jsonl

URL_MP_DEC = "https://cbr.ru/dkp/mp_dec/#t1"
URL_DECISION_KEY_RATE = "https://cbr.ru/dkp/mp_dec/decision_key_rate/#y2026"


def render_page_with_load_more(url: str, max_clicks: int = 100) -> str:
    # Playwright is used only for pages where the archive is loaded dynamically.
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url.split("#")[0], wait_until="networkidle", timeout=120_000)

        for _ in range(max_clicks):
            clicked = False
            buttons = page.locator("text=Загрузить еще")
            for i in range(buttons.count()):
                button = buttons.nth(i)
                if button.is_visible():
                    button.scroll_into_view_if_needed()
                    page.wait_for_timeout(500)
                    button.click(timeout=10_000)
                    page.wait_for_timeout(1200)
                    clicked = True
                    break
            if not clicked:
                break

        html = page.content()
        browser.close()
        return html


def collect_mp_decisions(use_playwright: bool = True) -> list[DocItem]:
    session = make_session()
    source_page = URL_MP_DEC.split("#")[0]

    if use_playwright:
        html = render_page_with_load_more(source_page)
    else:
        html = request_get(session, source_page).text

    return collect_html_items_from_page(html, source_page=source_page, section="mp_decisions")


def collect_decision_key_rate() -> list[DocItem]:
    session = make_session()
    source_page = URL_DECISION_KEY_RATE.split("#")[0]
    html = request_get(session, source_page).text
    soup = BeautifulSoup(html, "lxml")
    main = soup.find("main") or soup

    items = []
    seen = set()
    current_meeting_date = None
    last_date = None

    for node in main.descendants:
        if isinstance(node, str):
            text = " ".join(node.split())
            if not text:
                continue
            if "Заседание Совета директоров от" in text:
                date = parse_ru_date_from_text(text)
                current_meeting_date = date
                last_date = date
            else:
                parsed = parse_ru_date_from_text(text)
                if parsed:
                    last_date = parsed
            continue

        if getattr(node, "name", None) != "a" or not node.has_attr("href"):
            continue

        title = node.get_text(" ", strip=True)
        if title.startswith("Резюме обсуждения ключевой ставки"):
            category = "summary"
        elif title.startswith("Заявление Председателя Банка России"):
            category = "statement"
        else:
            continue

        url = normalize_cbr_url(node.get("href"), source_page)
        if not url or url in seen:
            continue
        seen.add(url)

        import hashlib

        items.append(
            DocItem(
                doc_id=hashlib.sha1(url.encode("utf-8")).hexdigest(),
                section="decision_key_rate",
                category=category,
                title=title,
                published_at=last_date or current_meeting_date,
                landing_url=url,
                source_page=source_page,
            )
        )

    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw/cbr_html"))
    parser.add_argument("--no-playwright", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    session = make_session()

    mp_items = collect_mp_decisions(use_playwright=not args.no_playwright)
    key_rate_items = collect_decision_key_rate()
    items = mp_items + key_rate_items

    write_jsonl([asdict(item) for item in items], output_dir / "document_manifest.jsonl")

    records, errors = fetch_documents(items, output_dir=output_dir, session=session)
    save_document_records(records, output_dir)
    write_jsonl(errors, output_dir / "errors.jsonl")

    print(json.dumps({"n_items": len(items), "n_documents": len(records), "n_errors": len(errors)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
