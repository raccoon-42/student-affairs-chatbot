"""LLM-based regulation (yonerge) parser.

Modern alternative to regulation_parser.py: that one just dumps
page.extract_text(), which leaves words glued together ("Dayanakve",
"esaslarıbelirlemektir") and no structure. Here pdfplumber only extracts
raw text and an LLM restructures it into one JSON object per MADDE with
fixed spacing, so each article becomes a self-contained chunk carrying
its section and title.

Run: uv run python preprocessing/parsers/regulation_parser_llm.py
Requires OPENROUTER_API_KEY in the environment / .env.
"""
import json
import re
import sys
from pathlib import Path

import pdfplumber
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import settings

PDF_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "yonerge.pdf"
OUTPUT_TXT = Path(__file__).resolve().parent.parent / "data" / "processed" / "yonerge-llm.txt"
OUTPUT_JSON = Path(__file__).resolve().parent.parent / "data" / "processed" / "yonerge-llm.json"

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


def extract_raw_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n\n".join(page.extract_text() or "" for page in pdf.pages)


def call_llm(raw_text):
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
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array in LLM response:\n{content[:500]}")
    return json.loads(match.group(0))


def render_article(article):
    header = f"MADDE {article['madde']}"
    if article.get("baslik"):
        header += f" - {article['baslik']}"
    header += f" ({article['bolum']})"
    return f"{header}\n{article['metin'].strip()}\n"


def main():
    if not settings.OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set.")

    raw_text = extract_raw_text(PDF_PATH)
    print(f"Extracted {len(raw_text)} chars, sending to {settings.OPENROUTER_MODEL}...")
    articles = call_llm(raw_text)
    print(f"LLM returned {len(articles)} articles.")

    OUTPUT_JSON.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUT_TXT.write_text("\n".join(render_article(a) for a in articles), encoding="utf-8")
    print(f"Wrote {OUTPUT_TXT.name} and {OUTPUT_JSON.name}")


if __name__ == "__main__":
    main()
