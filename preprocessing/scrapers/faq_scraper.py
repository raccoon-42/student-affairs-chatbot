"""Scrape the three student-affairs FAQ sources into Q&A pairs.

Two PDFs (lisans, lisansüstü) and one HTML page (İngilizce hazırlık).
All three share the same text shape — ALL-CAPS category headers,
question lines ending with "?", answer paragraphs below — so one
line-based parser handles them; fetching is per-format.

Output: preprocessing/data/processed/faq/faq.json, one object per Q&A pair.
Run again whenever the university updates the pages.
"""
import io
import json
import re

import requests

from config import settings

SOURCES = [
    {
        "audience": "hazirlik",
        "format": "html",
        "url": "https://ydyo.iyte.edu.tr/egitim/ydyo-sorular/",
    },
    {
        "audience": "lisans",
        "format": "pdf",
        "url": "https://ogrenciisleri.iyte.edu.tr/wp-content/uploads/sites/99/2025/05/S%C4%B1k%C3%A7a-Sorulan-Sorular-Lisans-13.04.2025.pdf",
    },
    {
        "audience": "lisansustu",
        "format": "pdf",
        "url": "https://ogrenciisleri.iyte.edu.tr/wp-content/uploads/sites/99/2025/05/Lisans%C3%BCst%C3%BC-S%C4%B1k%C3%A7a-Sorulan-Sorular-16.05.2025.pdf",
    },
]

OUTPUT_PATH = settings.ROOT / "preprocessing" / "data" / "processed" / "faq" / "faq.json"


def parse_faq_lines(lines, audience, source_url):
    """Turn a flat list of text lines into Q&A pairs.

    A question is a line ending with "?", optionally followed by a
    parenthetical — e.g. "... nasıl yapılır? (Eylül SBS)". A rare
    question wrapped over two lines loses its first half to the
    previous answer — accepted, the sources almost never wrap.
    """
    faqs = []
    category = None
    question = None
    answer_lines = []

    def flush():
        nonlocal question, answer_lines
        if question and answer_lines:
            faqs.append({
                "question": question,
                "answer": " ".join(answer_lines),
                "audience": audience,
                "category": category,
                "source_url": source_url,
            })
        question, answer_lines = None, []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if _is_category_header(line):
            flush()
            category = line
        elif _is_question(line):
            flush()
            question = line
        elif question:
            answer_lines.append(line)
        # lines before the first question (page intro, links) are dropped

    flush()
    return faqs


_QUESTION_RE = re.compile(r"\?\s*(\([^()]*\)\s*)?\*{0,2}\)?\s*$")


def _is_question(line):
    return bool(_QUESTION_RE.search(line))


def _is_category_header(line):
    return len(line) > 3 and line == line.upper() and not _is_question(line)


def _get(url):
    # the university servers are slow and occasionally drop connections
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            print(f"retrying {url}...")


def fetch_pdf_lines(url):
    import pdfplumber

    response = _get(url)
    lines = []
    with pdfplumber.open(io.BytesIO(response.content)) as pdf:
        for page in pdf.pages:
            lines.extend((page.extract_text() or "").split("\n"))
    return lines


def fetch_html_lines(url):
    from bs4 import BeautifulSoup

    response = _get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    content = soup.find("main") or soup.body or soup
    return content.get_text("\n").split("\n")


def main():
    all_faqs = []
    for source in SOURCES:
        fetch = fetch_pdf_lines if source["format"] == "pdf" else fetch_html_lines
        lines = fetch(source["url"])
        faqs = parse_faq_lines(lines, source["audience"], source["url"])
        print(f"{source['audience']}: {len(faqs)} Q&A pairs")
        all_faqs.extend(faqs)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_faqs, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(all_faqs)} Q&A pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
