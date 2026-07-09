"""Enrich the scraped program catalog with retrieval text.

Same idea as the forms describer: students don't ask with official
names — "ceng", "yazılım", "computer science okumak istiyorum" must
still find Bilgisayar Mühendisliği. One LLM call over the catalog
generates a one-line description and alias keywords per program
(English name, department abbreviation, colloquial terms).

Run: uv run python preprocessing/parsers/program_describer_llm.py
Rewrites the output unconditionally — one cheap call, no skip logic.
Output lands in preprocessing/data/processed/programlar/programs.json.
Requires OPENROUTER_API_KEY in the environment / .env.
"""
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import settings

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "programlar" / "manifest.json"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "programlar" / "programs.json"

PROMPT = """Aşağıda İYTE'nin (İzmir Yüksek Teknoloji Enstitüsü) diploma programları var.
Her program için kısa bir tanım ve arama eş anlamlıları üret.

Kurallar:
- Çıktı SADECE bir JSON dizisi olsun, başka hiçbir şey yazma.
- Her satır için bir nesne: {"title": ..., "level": ..., "description": ..., "aliases": [...]}
- "title" ve "level": girdiden BİREBİR kopyala, değiştirme.
- "description": tek cümle, programın ne çalıştığını anlat; İYTE'ye özgü bilgi uydurma.
- "aliases": öğrencilerin bu programı sorarken kullanabileceği 3-6 ifade:
  İngilizce adı, bölüm kısaltması (ör. "ceng", "eee"), gündelik adlar
  (ör. Bilgisayar Mühendisliği için ["computer engineering", "ceng", "yazılım", "bilgisayar bölümü"]).
- Hiçbir satırı atlama, yeni satır ekleme.

Programlar:
"""


def call_llm(lines):
    response = requests.post(
        f"{settings.OPENROUTER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
        json={
            "model": settings.OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": PROMPT + "\n".join(lines)}],
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


def main():
    if not settings.OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set.")

    catalog = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    lines = [f'- title: {e["title"]} | level: {e["level"]}' for e in catalog]

    print(f"Describing {len(catalog)} programs with {settings.OPENROUTER_MODEL}...")
    described = {(item["title"], item["level"]): item for item in call_llm(lines)}

    missing = [(e["title"], e["level"]) for e in catalog
               if (e["title"], e["level"]) not in described]
    if missing:
        sys.exit(f"LLM output is missing {len(missing)} programs: {missing}")

    programs = []
    for entry in catalog:
        item = described[(entry["title"], entry["level"])]
        programs.append({**entry,
                         "description": item["description"],
                         "aliases": item.get("aliases", [])})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(programs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(programs)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
