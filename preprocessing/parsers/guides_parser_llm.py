"""LLM-based parser for scraped ÖİDB process-guide pages.

Turns each page into self-contained chunks: one per process, step
sequence or department list, with tables flattened into readable
sentences. Process steps must survive in order — "katkı payı, ders
seçimi, danışman onayı" is the answer to "ders kaydı nasıl yapılır".

Run: uv run python -m preprocessing.parsers.guides_parser_llm <html-path>
or:  uv run python -m preprocessing.parsers.guides_parser_llm --all
--all walks the scrape manifest and parses every page that has no
output yet, so it is safe to rerun after a partial failure.
Outputs land in preprocessing/data/processed/rehber/<slug>.json.
Requires OPENROUTER_API_KEY in the environment / .env.
"""
import json
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import settings

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "rehber"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "rehber"

PROMPT = """Aşağıda bir üniversitenin Öğrenci İşleri Daire Başkanlığı bilgilendirme sayfasının
HTML içeriği var. Bunu bir chatbot'un arama dizinine girecek, kendi başına anlaşılır
parçalara dönüştür.

Kurallar:
- Çıktı SADECE bir JSON dizisi olsun, başka hiçbir şey yazma.
- Her parça için bir nesne: {"baslik": ..., "metin": ...}
- Her parça TEK bir konuyu kapsasın: bir süreç, bir başvuru türü, bir bölüm listesi.
  Sayfa tek konuysa tek parça da olabilir.
- "baslik": parçanın konusu (ör. "Ders Kayıt İşlem Adımları", "Kurum İçi Yatay
  Geçiş Başvurusu", "Çift Ana Dal Programı Olan Bölümler").
- "metin": parçanın tam içeriği düz metin olarak. İŞLEM ADIMLARINI SIRASIYLA ve
  numaralandırarak yaz — adım sırası cevabın kendisidir. Tabloları okunur
  cümlelere çevir; tarih, belge adı, birim adı gibi bilgileri atlama; hiçbir
  şey uydurma.
- "Akademik takvimde belirtilen tarihlerde" gibi ifadeleri aynen koru,
  kendi tarih ekleme.
- FİYAT VE ÜCRET TUTARLARINI METNE ALMA — bunlar her yıl değişir ve bayatlar.
  Bir işlemin ücretli/katkı paylı olduğunu söylemek serbest, tutar yazma.
- Menü, gezinme, buton metni gibi sayfa iskeletini alma.
- HTML'deki önemli bağlantıları (başvuru formu, kılavuz PDF'i, başka birimin
  sayfası) "(bağlantı: URL)" olarak metnin içinde bırak.

HTML:
"""


def call_llm(html):
    # transient TLS resets happen on long uploads — retry before giving up
    for attempt in range(3):
        try:
            response = requests.post(
                f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
                json={
                    "model": settings.OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": PROMPT + html}],
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


def render_chunk(chunk):
    return f"{chunk['baslik']}\n{chunk['metin'].strip()}\n"


def parse_page(html_path):
    """Send one page through the LLM and write the output.

    Writes nothing on failure, so an --all rerun retries the page
    instead of skipping past an empty output file.
    """
    html = Path(html_path).read_text(encoding="utf-8")
    print(f"{Path(html_path).stem}: {len(html)} chars, sending to {settings.OPENROUTER_MODEL}...")
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
            if entry.get("kind") == "link":  # authored link chunk, no page to parse
                continue
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
