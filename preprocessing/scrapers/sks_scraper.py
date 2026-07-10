"""Scrape campus-life pages from the SKS site (spor, topluluklar, yemekhane).

Unlike mevzuat these are HTML content pages, not PDFs. Each page's
content area is saved as raw HTML — tables (sports schedules, meal
prices) carry their structure in the markup, and the LLM parser needs
that structure to produce readable chunks; plain text extraction
destroys it. Downloads land in preprocessing/data/raw/sks/<slug>.html
plus a manifest.json recording title, topic, source URL and local path.

The page list is curated by hand: the site also hosts announcements,
tender notices and staff pages that would only pollute retrieval.
Run again whenever the university updates a page; files are
re-downloaded unconditionally so revised pages replace stale ones.
"""
import json

from bs4 import BeautifulSoup

from config import settings
from preprocessing.scrapers.fetch import fetch

# (topic, url) — the slug in the manifest comes from the URL's last segment
PAGES = [
    ("spor", "https://sks.iyte.edu.tr/spor/"),
    ("spor", "https://sks.iyte.edu.tr/spor-tesisleri-calisma-saatleri-ve-ucretleri/"),
    # rezervasyon carries the reservation-system link AND the kılavuz PDF;
    # sks-spor-uyelik-ve-randevu-sistemi is just the same PDF link again,
    # and spor/topluluklar is byte-identical to kultur/topluluklar
    ("spor", "https://sks.iyte.edu.tr/spor/rezervasyon/"),
    ("topluluklar", "https://sks.iyte.edu.tr/kultur/topluluklar/"),
    ("topluluklar", "https://sks.iyte.edu.tr/duyuru/ogrenci-topluluklarina-nasil-uye-olunur/"),
    ("topluluklar", "https://sks.iyte.edu.tr/kultur/kultur-ogrenci-topluluklari/"),
    ("yemekhane", "https://sks.iyte.edu.tr/beslenme-hizmetleri/"),
    ("yemekhane", "https://sks.iyte.edu.tr/beslenme-hizmetleri/yemekhane-hizmeti/"),
    ("yemekhane", "https://sks.iyte.edu.tr/burs/iyte-yemek-bursu/"),
]
# Deliberately excluded: yemek-bedeli-katki-payi and kantin-fiyatlari —
# price lists change every year and stale amounts mislead; the parser
# also strips fee amounts from the pages that remain (Ali, 2026-07-09).

OUTPUT_DIR = settings.ROOT / "preprocessing" / "data" / "raw" / "sks"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def extract_content(html):
    """The page's content area as (title, cleaned HTML)."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article", class_="post_article")
    if article is None:
        raise ValueError("no article.post_article on page — layout changed?")
    heading = article.find("h1")
    title = " ".join(heading.get_text().split()) if heading else ""
    for tag in article.find_all(["script", "style", "noscript", "img", "iframe"]):
        tag.decompose()
    return title, str(article)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slugs = [url.rstrip("/").rsplit("/", 1)[-1] for _, url in PAGES]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"slug collision in PAGES — two URLs share a last segment: {slugs}")

    manifest = []
    for (topic, url), slug in zip(PAGES, slugs):
        print(f"[{topic}] {url}")
        resp = fetch(url)
        title, content = extract_content(resp.text)
        path = OUTPUT_DIR / f"{slug}.html"
        path.write_text(content, encoding="utf-8")
        manifest.append({
            "title": title or slug,
            "topic": topic,
            "source_url": url,
            "file": path.name,
        })

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDownloaded {len(manifest)} pages, manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
