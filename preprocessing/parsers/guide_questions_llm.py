"""Writes the questions students would ask into each parsed guide chunk.

A long page splits into many chunks that all start with the page title,
so to the search they all look alike: "erasmus başvuru tarihleri ne
zaman" found 21 sibling chunks of the KA131 call page before the one
with the dates. The questions give each chunk the words a student uses
for what only that chunk answers. The model sees a page's chunks
together, so it can ask what sets each one apart.

Run: uv run python -m preprocessing.parsers.guide_questions_llm --all
or:  uv run python -m preprocessing.parsers.guide_questions_llm <processed-json>
--all fills every page with a chunk that has no questions yet (the guides
parser rewrites a changed page without them), so it is safe to rerun.
The questions land in each chunk's "sorular" field, in place.
Requires OPENROUTER_API_KEY in the environment / .env.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import settings
from preprocessing.parsers.llm_call import json_array

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "rehber"
MANIFEST = Path(__file__).resolve().parent.parent / "data" / "raw" / "rehber" / "manifest.json"
QUESTIONS_PER_CHUNK = 5

PROMPT = """Aşağıda bir üniversitenin bilgilendirme sayfasından çıkarılmış, numaralı metin parçaları var.
Öğrenciler bir chatbot'a soru soruyor; doğru parçanın bulunabilmesi için her parçaya, öğrencilerin
o parçanın cevapladığı şeyi sorarken yazacağı soruları ekleyeceğiz.

Kurallar:
- Çıktı SADECE bir JSON dizisi olsun, başka hiçbir şey yazma. Her parça için bir nesne:
  {{"parca": <parçanın numarası>, "sorular": [<{n} soru>]}}
- Her parçaya tam {n} soru yaz.
- Sorular YALNIZCA o parçanın cevapladığı şeyi sorsun. Aynı sayfadaki başka bir parçanın
  cevapladığı soruyu yazma: parçayı diğerlerinden ayıran şeyi sor (tarih veren parça için
  "ne zaman", şart sayan parça için "kaç ortalama lazım", liste veren parça için "hangileri").
- Öğrenci gibi yaz: kısa ve uzun, resmî ve gündelik karışık. Bir kısmını Türkçe karakter
  kullanmadan ("basvuru", "ogrenci"), bir kısmını öğrenci kısaltmalarıyla yaz.
- Cevabı soruya koyma: parçadaki tarih, sayı ve kişi adlarını soruda kullanma ("başvuru ne
  zaman", "17 Mart'ta mı başvuru" değil).
- Öğrenci konuyu da söyler: sorularda sayfanın konusunu geçir ("erasmus için kaç ortalama
  lazım", yalnızca "kaç ortalama lazım" değil).
- Parçada cevabı olmayan bir şey sorma.

Sayfa: {title}

Parçalar:
{chunks}
"""


def render_chunks(chunks):
    return "\n\n".join(f"[{i}] {chunk['baslik']}\n{chunk['metin'].strip()}"
                       for i, chunk in enumerate(chunks, start=1))


def page_title(path):
    """The page's title from the scrape manifest; chunk headings alone
    often leave out what the page is about ("Asgari Şartlar" of what?)."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return next((entry["title"] for entry in manifest
                 if Path(entry.get("file", "")).stem == Path(path).stem), Path(path).stem)


def needs_questions(chunks):
    return any(not chunk.get("sorular") for chunk in chunks)


def add_questions(path):
    """Ask for every chunk of one page and write them in. A page the model
    answered only in part keeps what it got; --all asks again for the page."""
    path = Path(path)
    chunks = json.loads(path.read_text(encoding="utf-8"))
    print(f"{path.stem}: {len(chunks)} chunks, sending to {settings.PARSER_MODEL}...")
    answer = json_array(PROMPT.format(n=QUESTIONS_PER_CHUNK, title=page_title(path),
                                      chunks=render_chunks(chunks)))
    by_number = {item.get("parca"): item.get("sorular") for item in answer if isinstance(item, dict)}
    filled = 0
    for i, chunk in enumerate(chunks, start=1):
        questions = [q.strip() for q in by_number.get(i) or [] if isinstance(q, str) and q.strip()]
        if questions:
            chunk["sorular"] = questions
            filled += 1
    path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote questions for {filled}/{len(chunks)} chunks"
          + ("" if filled == len(chunks) else " (rerun --all for the rest)"))


def main():
    if not settings.OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set.")
    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        for path in sorted(PROCESSED_DIR.glob("*.json")):
            if not needs_questions(json.loads(path.read_text(encoding="utf-8"))):
                print(f"Skipping {path.stem} (questions already written)")
                continue
            add_questions(path)
    elif len(sys.argv) == 2:
        add_questions(sys.argv[1])
    else:
        sys.exit(f"Usage: {sys.argv[0]} <processed-json> | --all")


if __name__ == "__main__":
    main()
