"""LLM-based parser for scraped SKS pages.

The raw HTML keeps table structure (sports schedules, price lists) that
plain text extraction destroys — a schedule cell means nothing without
its weekday column. The LLM turns each page into self-contained chunks:
one per section, facility, price table or student society, with tables
flattened into readable sentences ("Kapalı Spor Salonu Büyük Salon:
Voleybol Salı 16.00-20.00 ...").

Run: uv run python preprocessing/parsers/sks_parser_llm.py <html-path>
or:  uv run python preprocessing/parsers/sks_parser_llm.py --all
--all walks the scrape manifest and parses every page that has no
output yet, so it is safe to rerun after a partial failure.
Outputs land in preprocessing/data/processed/sks/<slug>.json.
Requires OPENROUTER_API_KEY in the environment / .env.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import settings
from preprocessing.parsers.llm_call import json_array

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "sks"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "sks"

PROMPT = """Aşağıda bir üniversitenin Sağlık Kültür Spor (SKS) sayfasının HTML içeriği var.
Bunu bir chatbot'un arama dizinine girecek, kendi başına anlaşılır parçalara dönüştür.

Kurallar:
- Çıktı SADECE bir JSON dizisi olsun, başka hiçbir şey yazma.
- Her parça için bir nesne: {"baslik": ..., "metin": ...}
- Her parça TEK bir konuyu kapsasın: bir bölüm, bir tesis, bir ücret tablosu,
  bir öğrenci topluluğu. Sayfa tek konuysa tek parça da olabilir.
- "baslik": parçanın konusu (ör. "Kapalı Spor Salonu Çalışma Programı",
  "IEEE Topluluğu", "Yemekhane Kart Sistemi").
- "metin": parçanın tam içeriği düz metin olarak. Tabloları okunur cümlelere çevir:
  program tablolarında gün + saat + etkinliği birlikte yaz
  (ör. "Salı 16.00-20.00: Voleybol"). Saat, tarih, isim gibi bilgileri atlama;
  hiçbir şey uydurma.
- FİYAT VE ÜCRET TUTARLARINI METNE ALMA — bunlar her yıl değişir ve bayatlar.
  Ücret tablolarını tamamen atla; bir hizmetin ücretli olduğunu söylemek serbest,
  tutar yazma.
- Menü, gezinme, buton metni gibi sayfa iskeletini alma.
- HTML'deki önemli bağlantıları "(bağlantı: URL)" olarak metnin içinde bırak.

HTML:
"""


def call_llm(html):
    return json_array(PROMPT + html, timeout=600)


def render_chunk(chunk):
    return f"{chunk['baslik']}\n{chunk['metin'].strip()}\n"


def parse_page(html_path):
    """Send one page through the LLM and write the output.

    Writes nothing on failure, so an --all rerun retries the page
    instead of skipping past an empty output file.
    """
    html = Path(html_path).read_text(encoding="utf-8")
    print(f"{Path(html_path).stem}: {len(html)} chars, sending to {settings.PARSER_MODEL}...")
    chunks = call_llm(html)
    print(f"LLM returned {len(chunks)} chunks.")
    if not chunks:
        print(f"FAILED {Path(html_path).stem}: no chunks, nothing written.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"{Path(html_path).stem}.json"
    output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {output.name}")


def main():
    if not settings.OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set.")
    usage = f"Usage: {sys.argv[0]} <html-path> | --all"

    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        manifest = json.loads((RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
        for entry in manifest:
            html_path = RAW_DIR / entry["file"]
            if (OUTPUT_DIR / f"{html_path.stem}.json").exists():
                print(f"Skipping {html_path.stem} (already parsed)")
                continue
            print(f"\n=== {entry['title']} ===")
            parse_page(html_path)
    elif len(sys.argv) == 2:
        parse_page(Path(sys.argv[1]))
    else:
        sys.exit(usage)


if __name__ == "__main__":
    main()
