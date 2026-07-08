"""LLM-based regulation (yonerge) parser.

Modern alternative to regulation_parser.py: that one just dumps
page.extract_text(), which leaves words glued together ("Dayanakve",
"esaslarıbelirlemektir") and no structure. Here pdfplumber only extracts
raw text and an LLM restructures it into one JSON object per MADDE with
fixed spacing, so each article becomes a self-contained chunk carrying
its section and title.

Run: uv run python preprocessing/parsers/regulation_parser_llm.py <pdf-path>
or:  uv run python preprocessing/parsers/regulation_parser_llm.py --html <url> <output-name>
or:  uv run python preprocessing/parsers/regulation_parser_llm.py --all
--all walks the scrape manifest and parses every mevzuat PDF that has
no output yet, so it is safe to rerun after a partial failure.
--html is for documents whose İYTE PDF is a scan with no text layer:
point it at the consolidated text on mevzuat.gov.tr instead (the site
403s non-browser agents, so the fetch sends a browser User-Agent).
Outputs land in preprocessing/data/processed/mevzuat/<name>.{txt,json}.
Requires OPENROUTER_API_KEY in the environment / .env.
"""
import json
import re
import sys
import time
from pathlib import Path

import pdfplumber
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import settings

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "mevzuat"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "mevzuat"

# Documents whose İYTE PDF is a scan with no text layer: --all parses
# these from the consolidated text on mevzuat.gov.tr instead.
HTML_SOURCES = {
    "iyte-lisansustu-egitim-ogretim-yonetmeligi":
        "https://www.mevzuat.gov.tr/anasayfa/MevzuatFihristDetayIframe"
        "?MevzuatTur=8&MevzuatNo=38653&MevzuatTertip=5",
}

PROMPT = """Aşağıda bir üniversite yönetmeliğinin PDF'inden çıkarılmış ham metin var.
PDF çıkarımı bazı kelimeleri bitişik yazmış (ör. "Dayanakve" -> "Dayanak ve"). Bunu yapılandırılmış JSON'a dönüştür.

Kurallar:
- Çıktı SADECE bir JSON dizisi olsun, başka hiçbir şey yazma.
- Her MADDE için bir nesne: {"bolum": ..., "madde": ..., "baslik": ..., "metin": ...}
- "bolum": maddenin ait olduğu bölüm başlığı (ör. "BİRİNCİ BÖLÜM - Amaç, Kapsam, Dayanak ve Tanımlar").
- "madde": madde numarası (tamsayı).
- "baslik": maddenin üstündeki başlık (ör. "Amaç ve kapsam"). Başlık yoksa null.
- "metin": maddenin tam metni. Bitişik yazılmış kelimeleri ayır, satır sonlarına
  bölünmüş cümleleri birleştir. Fıkra "(1)" ve bent "a)" numaralandırmasını koru.
  Metni asla özetleme veya kısaltma, birebir aktar.
- Hiçbir maddeyi atlama.

Ham metin:
"""


BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36")


def extract_raw_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


def extract_html_text(url):
    from bs4 import BeautifulSoup

    resp = requests.get(url, headers={"User-Agent": BROWSER_UA}, timeout=60)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser").get_text("\n")


def call_llm(raw_text):
    # transient TLS resets happen on long uploads — retry before giving up
    for attempt in range(3):
        try:
            response = requests.post(
                f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
                json={
                    "model": settings.OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": PROMPT + raw_text}],
                    "temperature": 0,
                },
                timeout=600,
            )
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == 2:
                raise
            wait = 5 * (attempt + 1)
            print(f"{e.__class__.__name__}, retrying in {wait}s...")
            time.sleep(wait)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array in LLM response:\n{content[:500]}")
    return json.loads(match.group(0))


def render_article(article):
    # Short "esaslar" documents may have no MADDE numbers or BÖLÜM headers
    # at all — the whole text comes back as one entry with nulls.
    if article.get("madde") is not None:
        header = f"MADDE {article['madde']}"
        if article.get("baslik"):
            header += f" - {article['baslik']}"
    else:
        header = article.get("baslik") or "HÜKÜMLER"
    if article.get("bolum"):
        header += f" ({article['bolum']})"
    return f"{header}\n{article['metin'].strip()}\n"


def restructure(raw_text, stem):
    """Send extracted text through the LLM and write the outputs.

    Writes nothing on failure, so an --all rerun retries the document
    instead of skipping past an empty output file.
    """
    if len(raw_text.strip()) < 200:
        print(f"FAILED {stem}: only {len(raw_text.strip())} chars extracted — "
              f"scanned PDF with no text layer? Try --html with a mevzuat.gov.tr URL.")
        return

    print(f"Extracted {len(raw_text)} chars, sending to {settings.OPENROUTER_MODEL}...")
    articles = call_llm(raw_text)
    print(f"LLM returned {len(articles)} articles.")
    if not articles:
        print(f"FAILED {stem}: no MADDE structure found — announcement page or "
              f"unparseable layout, nothing written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_json = OUTPUT_DIR / f"{stem}.json"
    output_txt = OUTPUT_DIR / f"{stem}.txt"
    output_json.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    output_txt.write_text("\n".join(render_article(a) for a in articles), encoding="utf-8")
    print(f"Wrote {output_txt.name} and {output_json.name}")


def parse_pdf(pdf_path):
    restructure(extract_raw_text(pdf_path), pdf_path.stem)


def main():
    if not settings.OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set.")
    usage = f"Usage: {sys.argv[0]} <pdf-path> | --html <url> <output-name> | --all"

    if len(sys.argv) == 4 and sys.argv[1] == "--html":
        restructure(extract_html_text(sys.argv[2]), sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == "--all":
        manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest:
            pdf_path = RAW_DIR / entry["file"]
            if (OUTPUT_DIR / f"{pdf_path.stem}.json").exists():
                print(f"Skipping {pdf_path.stem} (already parsed)")
                continue
            print(f"\n=== {entry['title']} ===")
            html_url = HTML_SOURCES.get(pdf_path.stem)
            if html_url:
                restructure(extract_html_text(html_url), pdf_path.stem)
            else:
                parse_pdf(pdf_path)
    elif len(sys.argv) == 2:
        parse_pdf(Path(sys.argv[1]))
    else:
        sys.exit(usage)


if __name__ == "__main__":
    main()
