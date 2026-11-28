"""Builds the generated parts of the retrieval eval set and freezes rewrites.

  faq        tests/data/faq_paraphrases.json -> retrieval_eval/faq.json (free)
  synthetic  Sonnet writes student-style questions for sampled chunks, one
             headless `claude` call per collection -> retrieval_eval/synthetic.json
  rewrite    fills "rewritten" for every case that lacks it, with the
             production QueryRewriter (OpenRouter, a fraction of a cent per case)

core.json is written by hand. Rebuilding faq/synthetic keeps the frozen
rewrite of any case whose query is unchanged.

Run: uv run python scripts/build_retrieval_eval.py faq|synthetic|rewrite
"""
import argparse
import json
import random
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from scripts.eval_retrieval import COLLECTIONS, DATASET_DIR  # noqa: E402

PARAPHRASES = ROOT / "tests" / "data" / "faq_paraphrases.json"
SYNTH_MODEL = "claude-sonnet-5-5"
SYNTH_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"}, "question": {"type": "string"}},
        "required": ["index", "question"]}}},
    "required": ["questions"],
}
SYNTH_PROMPT = """Aşağıda İYTE (İzmir Yüksek Teknoloji Enstitüsü) öğrenci asistanının bilgi tabanından numaralı parçalar var. Her parça için, bir öğrencinin chatbot'a yazabileceği ve BU parçanın cevapladığı tek bir soru yaz.

- Öğrenci gibi yaz: kısa, gündelik dil; Türkçe karakterleri ve noktalamayı her zaman özenle kullanmak zorunda değilsin.
- Parçadaki resmi ve nadir ifadeleri kopyalama, öğrencinin kendi kelimeleriyle sor (örn. "yarıyıl sonu sınavları" yerine "finaller", "ilişik kesme" yerine "okuldan kaydımı sildirmek").
- Soru, bu parçayla cevaplanacak kadar belirli olsun. Takvim maddelerinde yarıyılı ve akademik yılı soruda geçir; kişi ya da ders parçalarında ismi veya ders kodunu kullan.
- Bir parçadan makul bir öğrenci sorusu çıkmıyorsa (örn. içeriksiz bir liste başlığı) o parçayı atla.

Parçalar:
{chunks}"""


def write(name, cases, as_of):
    """Writes a dataset file, keeping frozen rewrites for unchanged queries."""
    path = DATASET_DIR / f"{name}.json"
    frozen = {}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        frozen = {(c["id"], c["query"]): c["rewritten"] for c in old["cases"] if c.get("rewritten")}
    for case in cases:
        if (case["id"], case["query"]) in frozen:
            case["rewritten"] = frozen[(case["id"], case["query"])]
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"as_of": as_of, "cases": cases}, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    print(f"wrote {len(cases)} cases to {path.relative_to(ROOT)}")


def all_chunks(client, collection):
    chunks, offset = [], None
    while True:
        points, offset = client.scroll(collection, limit=1000, offset=offset, with_payload=True)
        chunks += [(p.payload["text"], p.payload.get("metadata", {})) for p in points]
        if offset is None:
            return chunks


def build_faq(client):
    data = json.loads(PARAPHRASES.read_text(encoding="utf-8"))
    indexed = {metadata.get("question") for _, metadata in all_chunks(client, settings.FAQ_COLLECTION)}
    cases, missing = [], 0
    for i, (question, paraphrase) in enumerate(zip(data["questions"], data["paraphrases"])):
        if question not in indexed:  # the FAQ changed since the paraphrases were made
            missing += 1
            continue
        cases.append({"id": f"faq-{i:03d}", "query": paraphrase, "category": "faq",
                      "relevant": [{"collection": "faq", "metadata": {"question": question}}]})
    print(f"{missing} paraphrases skipped: their question is no longer in the FAQ collection")
    write("faq", cases, date.today().isoformat())


def ask_claude(prompt):
    out = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json", "--json-schema", json.dumps(SYNTH_SCHEMA),
         "--model", SYNTH_MODEL, "--tools", ""],
        capture_output=True, text=True, timeout=600, check=True)
    events = json.loads(out.stdout)
    result = next(e for e in (events if isinstance(events, list) else [events]) if e.get("type") == "result")
    if result.get("is_error"):
        raise RuntimeError(result.get("result"))
    return result["structured_output"]["questions"], result.get("total_cost_usd") or 0.0


def build_synthetic(client, per_collection, seed):
    cases, cost = [], 0.0
    for name, collection in COLLECTIONS.items():
        chunks = [c for c in all_chunks(client, collection) if len(c[0]) >= 40]
        sample = random.Random(f"{seed}-{name}").sample(chunks, min(per_collection, len(chunks)))
        listing = "\n\n".join(f"[{i}] {text[:1200]}" for i, (text, _) in enumerate(sample))
        questions, call_cost = ask_claude(SYNTH_PROMPT.format(chunks=listing))
        cost += call_cost
        for item in questions:
            if not 0 <= item["index"] < len(sample) or not item["question"].strip():
                continue
            text = sample[item["index"]][0]
            cases.append({
                "id": f"syn-{name}-{item['index']:02d}", "query": item["question"].strip(),
                "category": name,
                # the start of the chunk, not a hash of all of it: production
                # appends the calendar's DURUM tag and wraps FAQ answers
                "relevant": [{"collection": name, "contains": [text[:160]]}],
            })
        print(f"{name}: {len(questions)} questions for {len(sample)} chunks")
    print(f"claude cost: ${cost:.2f}")
    write("synthetic", cases, date.today().isoformat())


def freeze_rewrites():
    from app.conversation import QueryRewriter
    from app.llm import OpenRouterLLM
    from app.retrieval import normalize_query

    rewriter = QueryRewriter(OpenRouterLLM(), settings.GUARD_MODEL)
    for path in sorted(DATASET_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        todo = [c for c in data["cases"] if not c.get("rewritten")]
        cost = 0.0
        for case in todo:
            # what Conversation._resolve_search_query does: the rewriter gets
            # department codes named, retrieval gets the normalized rewrite
            case["rewritten"] = normalize_query(
                rewriter.rewrite(normalize_query(case["query"]), case.get("history", [])))
            cost += (rewriter.last_usage or {}).get("cost") or 0.0
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{path.name}: {len(todo)} rewrites frozen (${cost:.3f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("faq", "synthetic", "rewrite"))
    parser.add_argument("--per-collection", type=int, default=15)
    parser.add_argument("--seed", default="2026-11-24")
    args = parser.parse_args()

    if args.step == "rewrite":
        freeze_rewrites()
        return
    from qdrant_client import QdrantClient
    client = QdrantClient(settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY,
                          check_compatibility=False)
    if args.step == "faq":
        build_faq(client)
    else:
        build_synthetic(client, args.per_collection, args.seed)


if __name__ == "__main__":
    main()
