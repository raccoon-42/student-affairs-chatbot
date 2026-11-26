"""Do generated student questions help the guides search, and how to use them?

Each guides chunk carries ~5 questions Sonnet wrote for it
(preprocessing/parsers/guide_questions_llm.py). Variants, each in a
scratch "<guides>__qlab_<name>" collection (ids copied from the live
collection, every vector embedded here and disk-cached):
  now          the live collection: dense and BM25 on text + questions
               (ADR-0006's "append", in production since)
  text         dense and BM25 on the chunk text alone, the pre-ADR-0006 index
  bm25         BM25 on text + questions, dense on the text alone
  multivector  BM25 on text + questions; dense holds the text vector plus
               one vector per question, a chunk scores by its best match
  multi+append the same with the text + questions vector in place of the text one
  now+qvec     now, plus a third ranking from a separate multivector field
               holding only the question vectors: three rankings fused
Every variant fuses dense and BM25 rankings for each phrasing at RRF k=2,
as production does for guides (no filters there).

Run:    uv run python scripts/experiment_guide_questions.py
Clean:  uv run python scripts/experiment_guide_questions.py --drop
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from preprocessing.indexing.bm25 import FIELD, bm25_text, index_params  # noqa: E402
from scripts.eval_retrieval import CachedEmbedder, load_cases, rrf, score  # noqa: E402

GUIDES = settings.GUIDES_COLLECTION
LAB = "__qlab_"
VARIANTS = ("now", "text", "bm25", "multivector", "multi+append", "now+qvec")
DEPTH = settings.HYBRID_PREFETCH
CACHE = ROOT / "data" / "eval_cache" / "guide_questions.npz"


class BatchEmbedder(CachedEmbedder):
    """Disk-cached embeddings, fetched in batches."""

    def embed_many(self, texts):
        missing = list(dict.fromkeys(t for t in texts if self._key(t) not in self._vectors))
        if missing:
            print(f"embedding {len(missing)} texts...")
            for text, vector in zip(missing, self._inner.embed_documents(missing)):
                self._vectors[self._key(text)] = np.array(vector, dtype=np.float32)
            self.misses += len(missing)
        return [self._vectors[self._key(t)].tolist() for t in texts]


def appended(chunk):
    return "\n".join([chunk["text"], *chunk.get("questions", [])])


def build(client, embedder, chunks, live):
    """The scratch collections; live maps chunk text -> (id, vector). The
    live vectors embed text + questions, so the text alone is embedded here."""
    from qdrant_client.http import models as m

    missing = [c["text"] for c in chunks if c["text"] not in live]
    if missing:
        raise SystemExit(f"{len(missing)} parsed chunks are not in {GUIDES}; reindex guides first")
    without = sum(1 for c in chunks if not c.get("questions"))
    print(f"{len(chunks)} chunks, {without} without questions (link entries)")
    text_vectors = dict(zip((c["text"] for c in chunks),
                            embedder.embed_many([c["text"] for c in chunks])))
    appended_vectors = dict(zip((appended(c) for c in chunks),
                                embedder.embed_many([appended(c) for c in chunks])))
    questions = [q for c in chunks for q in c.get("questions", [])]
    question_vectors = dict(zip(questions, embedder.embed_many(questions)))
    embedder.save()

    for variant in VARIANTS[1:]:
        name = GUIDES + LAB + variant
        if client.collection_exists(name):
            client.delete_collection(name)
        multi = m.MultiVectorConfig(comparator=m.MultiVectorComparator.MAX_SIM)
        vectors_config = {"": m.VectorParams(
            size=3072, distance=m.Distance.COSINE,
            multivector_config=multi if variant.startswith("multi") else None)}
        if variant == "now+qvec":
            vectors_config["questions"] = m.VectorParams(
                size=3072, distance=m.Distance.COSINE, multivector_config=multi)
        client.create_collection(name, vectors_config=vectors_config)
        client.create_payload_index(name, FIELD, index_params(), wait=True)
        points = []
        for chunk in chunks:
            pid, text_vector = live[chunk["text"]][0], text_vectors[chunk["text"]]
            vector = {"text": text_vector,
                      "bm25": text_vector,
                      "multivector": [text_vector] + [question_vectors[q] for q in chunk.get("questions", [])],
                      "multi+append": [appended_vectors[appended(chunk)]]
                      + [question_vectors[q] for q in chunk.get("questions", [])],
                      "now+qvec": appended_vectors[appended(chunk)],
                      }[variant]
            vectors = {"": vector}
            if variant == "now+qvec":
                # link entries have no questions: their text stands in
                vectors["questions"] = ([question_vectors[q] for q in chunk.get("questions", [])]
                                        or [text_vector])
            points.append(m.PointStruct(id=pid, vector=vectors, payload={
                "text": chunk["text"], "metadata": chunk["metadata"],
                FIELD: bm25_text(chunk["text"] if variant == "text" else appended(chunk))}))
        for start in range(0, len(points), 50):
            client.upsert(name, points=points[start:start + 50])
        print(f"built {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--drop", action="store_true", help="delete the scratch collections")
    args = parser.parse_args()

    from qdrant_client import QdrantClient
    from qdrant_client.http import models as m

    from app.embeddings import default_embedder
    from app.retrieval import QdrantVectorStore, normalize_query
    from preprocessing.indexing.vectorizer import guides_chunks_from_dir

    client = QdrantClient(settings.QDRANT_URL, check_compatibility=False)
    if args.drop:
        for variant in VARIANTS[1:]:
            if client.collection_exists(GUIDES + LAB + variant):
                client.delete_collection(GUIDES + LAB + variant)
        print("scratch collections dropped")
        return

    chunks = guides_chunks_from_dir(ROOT / "preprocessing" / "data" / "processed" / "rehber")
    if not any(c.get("questions") for c in chunks):
        raise SystemExit("no questions yet: run preprocessing.parsers.guide_questions_llm --all first")
    live, offset = {}, None
    while True:
        records, offset = client.scroll(GUIDES, limit=200, offset=offset, with_payload=["text"],
                                        with_vectors=True)
        live.update({r.payload["text"]: (r.id, r.vector[""] if isinstance(r.vector, dict) else r.vector)
                     for r in records})
        if offset is None:
            break
    embedder = BatchEmbedder(default_embedder(), path=CACHE)
    build(client, embedder, chunks, live)

    query_cache = CachedEmbedder(default_embedder())
    read = QdrantVectorStore._read
    cases = [c for c in load_cases(("core", "synthetic"))
             if any(p["collection"] == "guides" for p in c["relevant"])]

    def ranking(points):
        return [(p.id, p.payload["text"], p.payload.get("metadata", {})) for p in points]

    def fused(variant, texts):
        name = GUIDES if variant == "now" else GUIDES + LAB + variant
        rankings = []
        for text in texts:
            vector = query_cache.embed_query(text)
            rankings.append(ranking(read(client.query_points, name,
                                         query=[vector] if variant.startswith("multi") else vector,
                                         limit=DEPTH, with_payload=True).points))
            if variant == "now+qvec":
                rankings.append(ranking(read(client.query_points, name, query=[vector],
                                             using="questions", limit=DEPTH,
                                             with_payload=True).points))
            if bm25_text(text).strip():
                rankings.append(ranking(read(client.query_points, name,
                                             query=m.TextQuery(text=bm25_text(text)), using=FIELD,
                                             limit=DEPTH, with_payload=True).points))
        return rrf(rankings, settings.RRF_K)

    forms = {"both": lambda c: [c["rewritten"], c["query"]] if c.get("rewritten") else [c["query"]],
             "raw": lambda c: [c["query"]]}
    rows, ranks = {}, {}
    for form, texts_of in forms.items():
        for case in cases:
            texts = list(dict.fromkeys(normalize_query(t) for t in texts_of(case)))
            relevant = {**case, "relevant": [p for p in case["relevant"] if p["collection"] == "guides"]}
            for variant in VARIANTS:
                result = score(relevant, {"guides": fused(variant, texts)})
                rows.setdefault((form, variant, case["source"]), []).append(result)
                ranks[(form, variant, case["id"])] = result["best"]
    query_cache.save()

    print(f"\n{len(cases)} guides cases; production top_k for guides = 4\n")
    for form in forms:
        print(f"form: {form}" + (" (what production searches with)" if form == "both" else ""))
        print(f"{'variant':<13}{'cases':<16}{'hit@1':>7}{'hit@3':>7}{'hit@4':>7}{'hit@5':>7}{'MRR':>7}")
        for variant in VARIANTS:
            for source in ("core", "synthetic", "all"):
                rs = [r for (f, v, s), rr in rows.items() if f == form and v == variant
                      and (source == "all" or s == source) for r in rr]
                hit = {k: sum(r["best"] is not None and r["best"] < k for r in rs) / len(rs) for k in (4, 5)}
                print(f"{variant if source == 'core' else '':<13}{source + f' ({len(rs)})':<16}"
                      f"{sum(r['hit@1'] for r in rs) / len(rs):>7.3f}{sum(r['hit@3'] for r in rs) / len(rs):>7.3f}"
                      f"{hit[4]:>7.3f}{hit[5]:>7.3f}{sum(r['rr'] for r in rs) / len(rs):>7.3f}")
        print()
    print("rank of the first relevant chunk, form both (1 = first; - = not in the top 50):")
    print(f"{'case':<28}" + "".join(f"{v:>13}" for v in VARIANTS))
    for case in cases:
        cells = [ranks[("both", v, case["id"])] for v in VARIANTS]
        if len(set(cells)) > 1 or cells[0] != 0:
            print(f"{case['id'][:27]:<28}" + "".join(f"{'-' if r is None else r + 1:>13}" for r in cells))
    print("(cases ranked 1st by every variant are left out)")


if __name__ == "__main__":
    main()
