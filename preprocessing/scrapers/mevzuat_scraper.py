"""Scrape every regulation PDF from the student-affairs mevzuat page.

The page lists three sections — Yönetmelikler, Yönergeler, Esaslar ve
İlkeler — each a <p><strong>header</strong></p> followed by a <ul> of
PDF links. Downloads land in preprocessing/data/raw/mevzuat/<category>/
with ASCII-slug filenames, plus a manifest.json recording the original
title, source URL, category and local path for each document.

Run again whenever the university updates the page; files are
re-downloaded unconditionally so revised PDFs replace stale ones.
"""
import json
import re
import unicodedata

from bs4 import BeautifulSoup

from config import settings
from preprocessing.scrapers.fetch import fetch

PAGE_URL = "https://ogrenciisleri.iyte.edu.tr/yonetmelikler-yonergeler-isleyis-esaslari/"

CATEGORIES = {
    "Yönetmelikler": "yonetmelikler",
    "Yönergeler": "yonergeler",
    "Esaslar ve İlkeler": "esaslar",
}

OUTPUT_DIR = settings.ROOT / "preprocessing" / "data" / "raw" / "mevzuat"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

TURKISH_MAP = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")


def slugify(title):
    text = title.translate(TURKISH_MAP)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text


def extract_documents(html):
    """Yield (category_slug, title, pdf_url) for every listed document."""
    soup = BeautifulSoup(html, "html.parser")
    for header, slug in CATEGORIES.items():
        strong = soup.find("strong", string=lambda s: s and s.strip() == header)
        if strong is None:
            raise ValueError(f"section header not found on page: {header}")
        ul = strong.find_parent("p").find_next_sibling("ul")
        for link in ul.find_all("a", href=True):
            title = " ".join(link.get_text().split())
            yield slug, title, link["href"]


def main():
    print(f"Fetching {PAGE_URL}")
    resp = fetch(PAGE_URL)

    manifest = []
    for category, title, url in extract_documents(resp.text):
        target_dir = OUTPUT_DIR / category
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = slugify(title) + ".pdf"
        path = target_dir / filename

        print(f"[{category}] {title}")
        pdf = fetch(url, timeout=60)
        path.write_bytes(pdf.content)

        manifest.append({
            "title": title,
            "category": category,
            "source_url": url,
            "file": f"{category}/{filename}",
        })

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDownloaded {len(manifest)} documents, manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
