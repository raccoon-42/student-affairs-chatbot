"""Compare the local embedding model against OpenRouter-hosted candidates.

Task: each FAQ question gets one LLM-generated paraphrase; a good embedding
model should retrieve the original question from the paraphrase. Ground
truth is exact (paraphrase i -> question i), so hit@1 / hit@3 / MRR need no
judging. Documents are pulled from the live Qdrant collections; ranking is
done in-memory — nothing is written to Qdrant.

Also prints top-3 calendar results for the golden test-case queries, and
per-query embedding latency (network round trip vs local inference).

Run: uv run python scripts/compare_embeddings_openrouter.py
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings

PARAPHRASE_CACHE = Path(__file__).resolve().parent.parent / "tests" / "test_cases" / "faq_paraphrases.json"
TEST_CASES_DIR = Path(__file__).resolve().parent.parent / "tests" / "test_cases"

# (model id, needs the e5/qwen3-style query instruction prefix)
CANDIDATES = [
    ("local", True),  # settings.EMBEDDING_MODEL via sentence-transformers
    ("qwen/qwen3-embedding-8b", True),
    ("google/gemini-embedding-2", False),
    ("openai/text-embedding-3-large", False),
    ("baai/bge-m3", False),
]


def scroll_collection(collection):
    """All points (text + payload) from a Qdrant collection."""
    from qdrant_client import QdrantClient
    client = QdrantClient(settings.QDRANT_URL)
    points, _ = client.scroll(collection_name=collection, limit=1000, with_payload=True)
    return [(p.payload["text"], p.payload) for p in points]


class LocalEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(settings.EMBEDDING_MODEL, trust_remote_code=True)

    def embed(self, texts):
        return np.asarray(self._model.encode(texts))


class OpenRouterEmbedder:
    def __init__(self, model):
        self._model = model

    def embed(self, texts, batch_size=32):
        vectors = []
        for i in range(0, len(texts), batch_size):
            data = self._embed_batch(texts[i:i + batch_size])
            vectors.extend(d["embedding"] for d in data)
            if len(texts) > batch_size:
                print(f"    embedded {len(vectors)}/{len(texts)}")
        return np.asarray(vectors)

    def _embed_batch(self, batch, attempts=3):
        for attempt in range(attempts):
            try:
                response = requests.post(
                    f"{settings.OPENROUTER_BASE_URL}/embeddings",
                    headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
                    json={"model": self._model, "input": batch},
                    timeout=60,
                )
            except requests.RequestException as error:
                print(f"    {self._model}: {type(error).__name__}, retrying ({attempt + 1}/{attempts})")
                continue
            if response.status_code == 200:
                return sorted(response.json()["data"], key=lambda d: d["index"])
            if response.status_code in (429, 500, 502, 503):
                print(f"    {self._model}: HTTP {response.status_code}, retrying ({attempt + 1}/{attempts})")
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{self._model} error {response.status_code}: {response.text[:300]}")
        raise RuntimeError(f"{self._model}: failed after {attempts} attempts")


def make_embedder(model_id):
    return LocalEmbedder() if model_id == "local" else OpenRouterEmbedder(model_id)


def with_query_instruction(query):
    """The production query prefix (retrieval.py) — also Qwen3's format."""
    return f"Instruct: {settings.EMBED_INSTRUCTION}\nQuery: {query}"


def generate_paraphrases(questions):
    """One Turkish paraphrase per FAQ question, cached across runs."""
    if PARAPHRASE_CACHE.exists():
        cached = json.loads(PARAPHRASE_CACHE.read_text(encoding="utf-8"))
        if cached.get("questions") == questions:
            return cached["paraphrases"]

    from app.llm import OpenRouterLLM
    llm = OpenRouterLLM()
    paraphrases = []
    for i in range(0, len(questions), 20):
        batch = questions[i:i + 20]
        prompt = (
            "Aşağıdaki üniversite SSS sorularının her birini, bir öğrencinin farklı "
            "kelimelerle sorabileceği şekilde yeniden yaz. Anlamı koru, kelimeleri değiştir. "
            "SADECE bir JSON dizisi döndür: her soru için bir yeniden yazım, aynı sırada.\n\n"
            + json.dumps(batch, ensure_ascii=False)
        )
        text = llm.chat(settings.GUARD_MODEL, [{"role": "user", "content": prompt}])
        match = re.search(r"\[.*\]", text, re.DOTALL)
        batch_out = json.loads(match.group(0))
        if len(batch_out) != len(batch):
            raise RuntimeError(f"Expected {len(batch)} paraphrases, got {len(batch_out)}")
        paraphrases.extend(batch_out)
        print(f"  paraphrased {len(paraphrases)}/{len(questions)}")

    PARAPHRASE_CACHE.write_text(
        json.dumps({"questions": questions, "paraphrases": paraphrases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paraphrases


def rank(doc_vectors, query_vectors):
    """Cosine-ranked doc indices for each query, best first."""
    docs = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    queries = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)
    return np.argsort(-(queries @ docs.T), axis=1)


def faq_metrics(rankings):
    """Query i's correct document is document i."""
    hit1 = hit3 = rr_sum = 0.0
    for i, order in enumerate(rankings):
        position = int(np.where(order == i)[0][0]) + 1
        hit1 += position == 1
        hit3 += position <= 3
        rr_sum += 1.0 / position
    n = len(rankings)
    return hit1 / n, hit3 / n, rr_sum / n


def golden_queries():
    queries = []
    for path in sorted(TEST_CASES_DIR.glob("*.json")):
        if path.name == PARAPHRASE_CACHE.name:
            continue
        for case in json.loads(path.read_text(encoding="utf-8")):
            queries.append(case["query"])
    return queries


def measure_latency(embedder, needs_instruction, sample_queries):
    times = []
    for query in sample_queries:
        text = with_query_instruction(query) if needs_instruction else query
        start = time.perf_counter()
        embedder.embed([text])
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model ids to run (default: all)")
    args = parser.parse_args()
    candidates = [(m, p) for m, p in CANDIDATES if args.models is None or m in args.models]

    print("Fetching corpora from Qdrant...")
    faq_docs = scroll_collection(settings.FAQ_COLLECTION)
    calendar_docs = scroll_collection(settings.CALENDAR_COLLECTION)
    questions = [text for text, _ in faq_docs]
    print(f"  faq: {len(faq_docs)} | calendar: {len(calendar_docs)}")

    print("Generating/loading FAQ paraphrases...")
    paraphrases = generate_paraphrases(questions)

    calendar_queries = golden_queries()
    latency_sample = calendar_queries[:5]
    results = {}

    for model_id, needs_instruction in candidates:
        label = settings.EMBEDDING_MODEL + " (local)" if model_id == "local" else model_id
        print(f"\n=== {label} ===")
        embedder = make_embedder(model_id)

        queries = [with_query_instruction(q) if needs_instruction else q for q in paraphrases]
        doc_vectors = embedder.embed(questions)
        query_vectors = embedder.embed(queries)
        hit1, hit3, mrr = faq_metrics(rank(doc_vectors, query_vectors))
        latency = measure_latency(embedder, needs_instruction, latency_sample)
        results[label] = (hit1, hit3, mrr, latency)
        print(f"  FAQ paraphrase retrieval: hit@1 {hit1:.3f} | hit@3 {hit3:.3f} | MRR {mrr:.3f}"
              f" | query embed {latency * 1000:.0f}ms")

        calendar_vectors = embedder.embed([text for text, _ in calendar_docs])
        query_texts = [with_query_instruction(q) if needs_instruction else q for q in calendar_queries]
        calendar_rankings = rank(calendar_vectors, embedder.embed(query_texts))
        for query, order in zip(calendar_queries, calendar_rankings):
            print(f"  Q: {query}")
            for doc_index in order[:3]:
                print(f"     {calendar_docs[doc_index][0][:100]}")

    print(f"\n{'model':50s} {'hit@1':>7s} {'hit@3':>7s} {'MRR':>7s} {'embed':>8s}")
    for label, (hit1, hit3, mrr, latency) in sorted(results.items(), key=lambda kv: -kv[1][2]):
        print(f"{label:50s} {hit1:7.3f} {hit3:7.3f} {mrr:7.3f} {latency * 1000:6.0f}ms")


if __name__ == "__main__":
    main()
