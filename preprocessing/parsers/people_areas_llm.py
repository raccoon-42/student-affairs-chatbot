"""Tag each academic with research areas from their bio.

"AI çalışan hangi hocalar var" is a filtered enumeration: the filter is
semantic, so top-k retrieval over individual bios can only ever surface
the nearest few people. The cure is the same as the roster and program
tam-liste chunks — precompute the enumeration. This pass reads every
department's people JSON, has the LLM assign canonical Turkish area
tags per person (bio-based, academics only), and writes the "areas"
field back into the same JSON. The vectorizer then emits one chunk per
area listing everyone tagged with it.

Run: uv run python preprocessing/parsers/people_areas_llm.py
One LLM call per department file; rewrites tags unconditionally.
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

PEOPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "people"

PROMPT = """Aşağıda bir üniversite bölümünün öğretim üyeleri ve biyografileri var.
Her kişiye, biyografisine dayanarak çalışma alanı etiketleri ata.

Kurallar:
- Çıktı SADECE bir JSON dizisi olsun, başka hiçbir şey yazma.
- Her kişi için bir nesne: {"name": ..., "areas": [...]}
- "name": girdiden BİREBİR kopyala.
- "areas": 1-4 Türkçe, küçük harf, kanonik alan etiketi (ör. "yapay zeka",
  "doğal dil işleme", "bilgisayar görüşü", "makine öğrenmesi", "yazılım mühendisliği",
  "siber güvenlik", "veri tabanları", "bilgisayar ağları", "yüksek başarımlı hesaplama").
- Aynı alanı hep aynı etiketle yaz; eş anlamlı türetme ("ai" değil "yapay zeka").
- ŞEMSİYE KURALI: doğal dil işleme, bilgisayar görüşü, makine öğrenmesi gibi alt
  alanlarda çalışan herkese "yapay zeka" etiketini DE ekle.
- Biyografi alan çıkarmaya yetmiyorsa "areas" boş dizi olsun; alan UYDURMA.
- Hiç kimseyi atlama.

Kişiler:
"""


def call_llm(people_lines):
    # transient timeouts happen on long OpenRouter calls — retry before giving up
    for attempt in range(3):
        try:
            response = requests.post(
                f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
                json={
                    "model": settings.OPENROUTER_MODEL,
                    "messages": [{"role": "user", "content": PROMPT + "\n\n".join(people_lines)}],
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


def tag_department(path):
    people = json.loads(path.read_text(encoding="utf-8"))
    # assistants get tagged too — "asistanlar ne çalışıyor" is a real
    # student question, and most have bios with thesis/research topics
    taggable = [p for p in people
                if p["role"] in ("akademik", "arastirma-gorevlisi") and p.get("bio")]
    if not taggable:
        print(f"{path.stem}: no academic bios to tag")
        return

    lines = [f"- {p['name']}: {p['bio']}" for p in taggable]
    print(f"{path.stem}: tagging {len(taggable)} academics with {settings.OPENROUTER_MODEL}...")
    tagged = {item["name"]: item.get("areas", []) for item in call_llm(lines)}

    missing = [p["name"] for p in taggable if p["name"] not in tagged]
    if missing:
        sys.exit(f"LLM output is missing {len(missing)} people: {missing}")

    for person in people:
        person["areas"] = tagged.get(person["name"], [])

    path.write_text(json.dumps(people, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = sum(1 for p in people if p.get("areas"))
    print(f"{path.stem}: wrote areas for {counts}/{len(people)} people")


def main():
    if not settings.OPENROUTER_API_KEY:
        sys.exit("OPENROUTER_API_KEY is not set.")
    files = sorted(PEOPLE_DIR.glob("*.json"))
    if not files:
        sys.exit(f"No people files in {PEOPLE_DIR} — run the people scraper first.")
    for path in files:
        tag_department(path)


if __name__ == "__main__":
    main()
