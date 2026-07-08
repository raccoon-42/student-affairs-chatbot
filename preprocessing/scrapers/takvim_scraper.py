"""Scrape the academic calendar PDFs from the main IZTECH site.

The page links one PDF per academic year ("2025-2026 Akademik Takvimi").
Downloads land in preprocessing/data/raw/takvim/ with ASCII-slug
filenames, plus a manifest.json recording title, source URL and local
path for each calendar.

Run again whenever the university updates the page; files are
re-downloaded unconditionally so revised PDFs replace stale ones.
"""
import json
import re

import requests
from bs4 import BeautifulSoup

from config import settings

PAGE_URL = "https://iyte.edu.tr/akademik/akademik-takvim/"

OUTPUT_DIR = settings.ROOT / "preprocessing" / "data" / "raw" / "takvim"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def extract_calendars(html):
    """Yield (title, pdf_url) for every linked academic-year calendar."""
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("a", href=True):
        title = " ".join(link.get_text().split())
        if "Akademik Takvim" in title and link["href"].lower().endswith(".pdf"):
            yield title, link["href"]


def main():
    print(f"Fetching {PAGE_URL}")
    resp = requests.get(PAGE_URL, timeout=30)
    resp.raise_for_status()

    docs = dict(extract_calendars(resp.text))
    if not docs:
        raise ValueError("no calendar PDFs found on page")

    manifest = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for title, url in docs.items():
        filename = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") + ".pdf"
        path = OUTPUT_DIR / filename

        print(f"{title}")
        pdf = requests.get(url, timeout=60)
        pdf.raise_for_status()
        path.write_bytes(pdf.content)

        manifest.append({
            "title": title,
            "source_url": url,
            "file": filename,
        })

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDownloaded {len(manifest)} calendars, manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
