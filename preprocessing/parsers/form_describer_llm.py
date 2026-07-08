"""Enrich the scraped form catalog with retrieval text.

Bare titles are thin for matching: a student writes "kimliğimi
kaybettim", the form is "Öğrenci Kimlik Kartı Formu". Same idea as the
FAQ corpus (embed the question, not the answer): one LLM call over all
titles generates a one-line use-case description and alias keywords per
form, and title + description + aliases become the embedded chunk text.

Run: uv run python preprocessing/parsers/form_describer_llm.py
Rereads the whole manifest and rewrites the output unconditionally —
it is one cheap call, so no skip logic. Output lands in
preprocessing/data/processed/formlar/forms.json.
Requires OPENROUTER_API_KEY in the environment / .env.
"""
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import settings

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "formlar" / "manifest.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "formlar" / "forms.json"

PROMPT = """Aşağıda İYTE Öğrenci İşleri'nin form/dilekçe başlıkları var. Her başlık için,
bir öğrencinin bu formu hangi durumda arayacağını anlatan kısa bir açıklama ve
arama eş anlamlıları üret.

Kurallar:
- Çıktı SADECE bir JSON dizisi olsun, başka hiçbir şey yazma.
- Her başlık için bir nesne: {"title": ..., "description": ..., "aliases": [...]}
- "title": başlığı BİREBİR kopyala, değiştirme.
- "description": tek cümle, formun ne için doldurulduğunu öğrenci diliyle anlat
  (ör. "Kaybolan veya çalınan öğrenci kimlik kartı yerine yenisini almak için doldurulur.").
- "aliases": öğrencilerin bu konuyu sorarken kullanabileceği 3-6 anahtar ifade
  (ör. ["kimlik kaybettim", "kimlik zayi", "yeni kimlik kartı"]).
- Hiçbir başlığı atlama, yeni başlık ekleme.

Başlıklar:
"""


def call_llm(titles):
    response = requests.post(
        f"{settings.OPENROUTER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
        json={
            "model": settings.OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": PROMPT + "\n".join(f"- {t}" for t in titles)}],
            "temperature": 0,
        },
        timeout=300,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array in LLM response:\n{content[:500]}")
    return json.loads(match.group(0))


def main():
    if not settings.OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set.")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    titles = [entry["title"] for entry in manifest]

    print(f"Describing {len(titles)} forms with {settings.OPENROUTER_MODEL}...")
    described = {item["title"]: item for item in call_llm(titles)}

    missing = [t for t in titles if t not in described]
    if missing:
        sys.exit(f"LLM output is missing {len(missing)} titles: {missing}")

    forms = []
    for entry in manifest:
        item = described[entry["title"]]
        forms.append({**entry,
                      "description": item["description"],
                      "aliases": item.get("aliases", [])})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(forms, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(forms)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
