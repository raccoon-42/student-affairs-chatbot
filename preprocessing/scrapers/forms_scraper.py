"""Scrape the student-affairs forms page into a link catalog.

Unlike mevzuat, the PDFs themselves are worthless for retrieval — a form
is blank fields; the retrieval key is its title and the deliverable is
its URL. So nothing is downloaded: the manifest IS the corpus. Every PDF
link inside the page's content area (the menu repeats site-wide links,
so it is excluded) becomes one entry with its title, URL and, when the
filename carries one, the İYTE-ÖİDB form code.

The page URL is a CLI argument so other form sources (enstitü, SKS) can
be added later; entries from different pages merge into one manifest
keyed by URL. Run again whenever the university updates a page.
"""
import json
import re
import sys

from bs4 import BeautifulSoup

from config import settings
from preprocessing.scrapers.fetch import fetch

DEFAULT_PAGE_URL = "https://ogrenciisleri.iyte.edu.tr/formlar/"

OUTPUT_DIR = settings.ROOT / "preprocessing" / "data" / "raw" / "formlar"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

FORM_CODE = re.compile(r"İYTE-[ÖO]İDB-(\d+)", re.IGNORECASE)


def extract_forms(html, page_url):
    """Yield one entry per PDF link in the page's content area."""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find("article", class_="post_article") or soup
    for link in content.find_all("a", href=True):
        url = link["href"].split("#")[0]
        if not url.lower().endswith((".pdf", ".doc", ".docx")):
            continue
        title = " ".join(link.get_text().split())
        if not title:
            continue
        code = FORM_CODE.search(url)
        # not every form filename carries an ÖİDB code — the title decides
        is_form = code or "form" in title.lower()
        yield {
            "title": title,
            "source_url": url,
            "form_code": f"İYTE-ÖİDB-{code.group(1)}" if code else None,
            "category": "form" if is_form else "bilgilendirme",
            "page_url": page_url,
        }


def main():
    page_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PAGE_URL

    existing = []
    if MANIFEST_PATH.exists():
        existing = [entry for entry in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                    if entry.get("page_url") != page_url]

    print(f"Fetching {page_url}")
    resp = fetch(page_url)

    scraped = list(extract_forms(resp.text, page_url))
    if not scraped:
        sys.exit(f"No form links found on {page_url} — page layout changed?")
    for entry in scraped:
        print(f"[{entry['category']}] {entry['title']}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = existing + scraped
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n{len(scraped)} links from this page, {len(manifest)} total in {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
