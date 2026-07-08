"""LLM-based academic calendar parser.

Modern alternative to academic_calendar_parser.py: instead of hand-tuned
pdfplumber table settings plus if/else formatting rules, pdfplumber only
extracts raw text and an LLM structures it into JSON events. Python then
renders the retrieval-friendly Turkish lines deterministically, so dates
and casing never depend on LLM formatting.

Fixes over the rule-based parser:
- full dates ("3 Haziran 2024 Pazartesi") instead of "03.Haz.24 Pzt"
- events wrapped across table rows come back as one line
- every line carries its term, so chunks keep context
- no .lower() mangling of Turkish dotted-I

Run: uv run python preprocessing/parsers/academic_calendar_parser_llm.py <pdf-path>
e.g. preprocessing/data/raw/takvim/2025-2026-akademik-takvimi.pdf
Outputs land in preprocessing/data/processed/takvim/<name>.{txt,json}.
Requires OPENROUTER_API_KEY in the environment / .env.
"""
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

import pdfplumber
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import settings

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "takvim"

MONTHS_TR = [
    "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran",
    "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık",
]
WEEKDAYS_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]

PROMPT = """Aşağıda bir üniversitenin akademik takvim PDF'inden çıkarılmış ham metin var.
Bunu yapılandırılmış JSON'a dönüştür.

Kurallar:
- Çıktı SADECE bir JSON dizisi olsun, başka hiçbir şey yazma.
- Her takvim olayı için bir nesne: {"term": ..., "start_date": ..., "end_date": ..., "description": ...}
- "term": olayın ait olduğu yarıyıl başlığı (ör. "2024-2025 Güz Yarıyılı", "2024-2025 Bahar Yarıyılı").
- "start_date" ve "end_date": ISO formatında (YYYY-MM-DD). Tek günlük olaylarda end_date null olsun.
- Tarihi belirsiz olan olaylarda (ör. "*" veya "ÖSYM'ce belirlenecek") start_date null olsun ve
  description'a tarihin nasıl belirleneceğini ekle.
- "description": olayın tam açıklaması. Satır sonlarına bölünmüş cümleleri birleştir.
  Türkçe karakterleri ve kısaltmaların büyük/küçük harflerini (EABD, YDYO, ÖİDB, ÖBS, LUBES vb.) koru.
- Hiçbir olayı atlama.

Ham metin:
"""


def extract_raw_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


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
                timeout=300,
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
    # Strip a possible ```json fence before parsing
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array in LLM response:\n{content[:500]}")
    return json.loads(match.group(0))


def format_date_tr(iso_date):
    d = date.fromisoformat(iso_date)
    return f"{d.day} {MONTHS_TR[d.month - 1]} {d.year} {WEEKDAYS_TR[d.weekday()]}"


def render_line(event):
    term = event.get("term") or "Akademik Takvim"
    description = event["description"].strip().rstrip(".")
    start, end = event.get("start_date"), event.get("end_date")
    if start and end:
        when = f"{format_date_tr(start)} ile {format_date_tr(end)} tarihleri arasında"
    elif start:
        when = f"{format_date_tr(start)} tarihinde"
    else:
        when = "Tarihi ayrıca duyurulacak:"
    return f"[{term}] {when} {description}."


def main():
    if not settings.OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set.")
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <pdf-path>")

    pdf_path = Path(sys.argv[1])
    output_json = OUTPUT_DIR / f"{pdf_path.stem}.json"
    output_txt = OUTPUT_DIR / f"{pdf_path.stem}.txt"

    raw_text = extract_raw_text(pdf_path)
    print(f"Extracted {len(raw_text)} chars, sending to {settings.OPENROUTER_MODEL}...")
    events = call_llm(raw_text)
    print(f"LLM returned {len(events)} events.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    output_txt.write_text("\n".join(render_line(e) for e in events) + "\n", encoding="utf-8")
    print(f"Wrote {output_txt.name} and {output_json.name}")


if __name__ == "__main__":
    main()
