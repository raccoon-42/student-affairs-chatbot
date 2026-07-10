"""Scrape process-guide pages from the ÖİDB (öğrenci işleri) site.

These are the "how does X work" pages the regulations don't spell out
step by step: ders seçimi, yatay geçiş variants, çift ana dal / yan dal
applications, lisansüstü başvuru, askerlik. Same WordPress theme as the
SKS site, so the content extraction is shared. Saved as raw HTML because
several pages carry tables the LLM parser needs intact.

The page list is curated by hand, checked against the other corpora so
the same text isn't indexed twice. Deliberately excluded:
- lisans-programlarina-yatay-gecis — its A/B/C sections duplicate the
  three per-type pages below
- ogrenci-bilgi-sistemi — a 67-char stub holding only the UBYS link
- katki-payi-ve-ogrenim-ucretleri — fee amounts change every year and
  stale amounts mislead (Ali's no-amounts rule, 2026-07-09)
- iç kontrol / faaliyet raporu / vizyon-misyon bureaucracy
The yönerge overlap (yatay geçiş, ÇAP/yan dal) is deliberate: mevzuat
carries the rules, these pages carry the mechanics (where to apply,
which documents, which departments participate).

Run again whenever the university updates a page; files are
re-downloaded unconditionally so revised pages replace stale ones.
"""
import json

from bs4 import BeautifulSoup

from config import settings
from preprocessing.scrapers.fetch import fetch
from preprocessing.scrapers.sks_scraper import extract_content

# (topic, url) — the slug in the manifest comes from the URL's last segment
PAGES = [
    ("ders-secimi", "https://ogrenciisleri.iyte.edu.tr/ders-secimi-ile-ilgili-bilgilendirme/"),
    ("yatay-gecis", "https://ogrenciisleri.iyte.edu.tr/kurum-ici-yatay-gecis/"),
    ("yatay-gecis", "https://ogrenciisleri.iyte.edu.tr/kurumlar-arasi-yatay-gecis/"),
    ("yatay-gecis", "https://ogrenciisleri.iyte.edu.tr/merkezi-yerlestirme-puani-ile-yatay-gecis/"),
    ("yatay-gecis", "https://ogrenciisleri.iyte.edu.tr/lisansustu-programlara-yatay-gecis/"),
    ("cift-anadal-yandal", "https://ogrenciisleri.iyte.edu.tr/cift-anadal/"),
    ("cift-anadal-yandal", "https://ogrenciisleri.iyte.edu.tr/yan-dal/"),
    ("basvuru", "https://ogrenciisleri.iyte.edu.tr/lisansustu-programlara-basvuru-bilgileri/"),
    ("basvuru", "https://ogrenciisleri.iyte.edu.tr/lisans-programlarina-yabanci-uyruklu-ogrenci-basvuru-bilgileri/"),
    ("askerlik", "https://ogrenciisleri.iyte.edu.tr/lisansustu-askerlik-islemleri/"),
]

OUTPUT_DIR = settings.ROOT / "preprocessing" / "data" / "raw" / "rehber"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

SITE_URL = "https://ogrenciisleri.iyte.edu.tr/"

# Step-by-step PDFs linked from the site MENU, not from any page's
# content — without this they are unreachable from every chunk. Each
# becomes a single authored link chunk (kind: "link", no raw file, no
# LLM parse): the description is the retrieval text and the chip links
# the PDF itself. The href is re-discovered from the menu on every
# scrape because the wp-content upload path changes with each re-upload.
PDF_LINKS = [
    ("ders-secimi", "ÖĞRENCİ DERS KAYIT KILAVUZU", "Öğrenci Ders Kayıt Kılavuzu (PDF)",
     "ÖBS/UBYS üzerinden ders seçiminin nasıl yapıldığını ekran görüntüleriyle "
     "adım adım gösteren kılavuz. Ders kaydı, ders seçme ekranı, şube seçimi ve "
     "danışman onayına gönderme adımlarını görsel olarak anlatır."),
    ("ders-secimi", "DERS KAYITLANMA ADIMLARI", "Ders Kayıtlanma Adımları (PDF)",
     "Ders kayıt sürecinin işlem adımlarını sırasıyla özetleyen belge: katkı payı, "
     "ders seçimi ve danışman onayı."),
]


def find_menu_pdfs(html):
    """{menu text -> pdf href} for the PDF_LINKS entries, from the nav."""
    soup = BeautifulSoup(html, "html.parser")
    found = {}
    for a in soup.find_all("a", href=True):
        text = " ".join(a.get_text().split())
        for _, menu_text, _, _ in PDF_LINKS:
            if text == menu_text and a["href"].split("#")[0].lower().endswith(".pdf"):
                found[menu_text] = a["href"].split("#")[0]
    missing = [menu_text for _, menu_text, _, _ in PDF_LINKS if menu_text not in found]
    if missing:
        raise ValueError(f"menu PDF link(s) not found on {SITE_URL}: {missing} — menu changed?")
    return found


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

    print(f"[pdf-links] {SITE_URL}")
    pdf_hrefs = find_menu_pdfs(fetch(SITE_URL).text)
    for topic, menu_text, title, description in PDF_LINKS:
        print(f"  {title} -> {pdf_hrefs[menu_text]}")
        manifest.append({
            "title": title,
            "topic": topic,
            "source_url": pdf_hrefs[menu_text],
            "kind": "link",
            "description": description,
        })

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDownloaded {len(manifest) - len(PDF_LINKS)} pages + {len(PDF_LINKS)} PDF links, "
          f"manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
