"""Scrape the İYTE degree-program directories into a catalog.

Two sources: the lisans page groups 17 programs under faculty headings
(Elementor markup: an h4 per faculty, then an icon-list of program
links), the lisansüstü page on lee.iyte.edu.tr groups programs under h2
level headings (Yüksek Lisans / Doktora, each with a Disiplinlerarası
variant). Like forms, this is a link catalog — the deliverable is the
program's existence, level, faculty and department site URL; nothing is
downloaded. Both pages walk in document order, tracking the current
heading, so reordered entries survive but a renamed heading fails loudly.

Run again whenever the university adds a program.
"""
import json
import re

import requests
from bs4 import BeautifulSoup

from config import settings

LISANS_URL = "https://iyte.edu.tr/akademik/lisans-programlari/"
LISANSUSTU_URL = "https://lee.iyte.edu.tr/lisansustu-programlar/"

# h2 section heading on the lisansüstü page -> program level
LEVEL_HEADINGS = {
    "Yüksek Lisans Programları": "yukseklisans",
    "Disiplinlerarası Yüksek Lisans Programları": "yukseklisans",
    "Doktora Programları": "doktora",
    "Disiplinlerarası Doktora Programları": "doktora",
}

OUTPUT_DIR = settings.ROOT / "preprocessing" / "data" / "raw" / "programlar"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

DEPT_SITE = re.compile(r"https?://[a-z]+\.iyte\.edu\.tr", re.IGNORECASE)


def fetch_content(url):
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    article = soup.find("article")
    if article is None:
        raise ValueError(f"no <article> content area on {url} — layout changed?")
    return article


def extract_lisans(article):
    """Walk in document order: an h4 naming a faculty applies to every
    program link after it, until the next faculty h4."""
    faculty = None
    for element in article.descendants:
        if getattr(element, "name", None) == "h4" and "Fakültesi" in element.get_text():
            faculty = " ".join(element.get_text().split())
        elif getattr(element, "name", None) == "a" and element.get("href"):
            title = " ".join(element.get_text().split())
            url = element["href"]
            if not title or "Fakültesi" in title or not DEPT_SITE.match(url):
                continue
            if faculty is None:
                raise ValueError(f"program {title!r} before any faculty heading")
            yield {"title": title, "level": "lisans", "faculty": faculty,
                   "source_url": url, "page_url": LISANS_URL}


def extract_lisansustu(article):
    """Same walk, but h2 level headings partition the page instead of
    faculties; the same program name may appear under both levels."""
    level = None
    for element in article.descendants:
        if getattr(element, "name", None) == "h2":
            heading = " ".join(element.get_text().split())
            level = LEVEL_HEADINGS.get(heading, level)
        elif getattr(element, "name", None) == "a" and element.get("href"):
            title = " ".join(element.get_text().split())
            url = element["href"]
            if not title or level is None or not DEPT_SITE.match(url):
                continue
            if url.rstrip("/").endswith("lee.iyte.edu.tr"):
                continue  # enstitü self-links are not programs
            yield {"title": title, "level": level, "faculty": None,
                   "source_url": url, "page_url": LISANSUSTU_URL}


def main():
    catalog = []
    seen = set()
    for url, extract in ((LISANS_URL, extract_lisans), (LISANSUSTU_URL, extract_lisansustu)):
        print(f"Fetching {url}")
        count = 0
        for entry in extract(fetch_content(url)):
            key = (entry["title"], entry["level"])
            if key in seen:  # some pages repeat entries in the markup
                continue
            seen.add(key)
            catalog.append(entry)
            count += 1
        print(f"  {count} programs")

    levels = {entry["level"] for entry in catalog}
    if not {"lisans", "yukseklisans", "doktora"} <= levels:
        raise ValueError(f"expected all three levels, got {levels} — headings changed?")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{len(catalog)} programs total, manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
