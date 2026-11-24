"""Qdrant BM25 text-processing variants on the retrieval eval set.

Production scores BM25 with Qdrant's text index on a field holding our
tokenizer's output (folded, words cut to 5 letters). This checks that
choice against Qdrant's own text processing. Each collection is copied
into a scratch "<name>__bm25lab" collection with the same dense vectors
(no embedding calls) and three BM25-scored text fields:
  bm25_text  our tokenizer's output, as in production
  text       words, lowercase, ASCII folding
  text_stem  the same plus the Snowball Turkish stemmer
Every variant is scored alone and fused with the same dense ranking at
RRF k=2, like production (without production's filters and list rules).

The first run (2026-11-24, before the switch) also had our own sparse-vector
BM25 as a row; its numbers, and why we switched, are in docs/adr/0004.

Run:    uv run python scripts/experiment_bm25_qdrant.py
Clean:  uv run python scripts/experiment_bm25_qdrant.py --drop
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from preprocessing.indexing.bm25 import FIELD, bm25_text, index_params  # noqa: E402
from scripts.eval_retrieval import (COLLECTIONS, SOURCES, CachedEmbedder,  # noqa: E402
                                    load_cases, rrf, score)

LAB = "__bm25lab"
DEPTH = settings.HYBRID_PREFETCH


def build(client, name):
    from qdrant_client.http import models as m

    lab = COLLECTIONS[name] + LAB
    if client.collection_exists(lab):
        return lab
    client.create_collection(lab, vectors_config=m.VectorParams(size=3072, distance=m.Distance.COSINE))
    client.create_payload_index(lab, FIELD, index_params())
    for field, stem in (("text", False), ("text_stem", True)):
        client.create_payload_index(lab, field, m.TextIndexParams(
            type=m.TextIndexType.TEXT, tokenizer=m.TokenizerType.WORD, lowercase=True,
            ascii_folding=True, scoring=True,
            stemmer=m.SnowballParams(type=m.Snowball.SNOWBALL, language=m.SnowballLanguage.TURKISH)
            if stem else None))
    offset = None
    while True:
        points, offset = client.scroll(COLLECTIONS[name], limit=200, offset=offset,
                                       with_payload=True, with_vectors=True)
        client.upsert(lab, points=[m.PointStruct(
            id=p.id, vector={"": p.vector[""]},
            payload={**p.payload, "text_stem": p.payload["text"], FIELD: bm25_text(p.payload["text"])})
            for p in points])
        if offset is None:
            break
    print(f"built {lab}")
    return lab


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--drop", action="store_true", help="delete the scratch collections")
    args = parser.parse_args()

    from qdrant_client import QdrantClient
    from qdrant_client.http import models as m

    from app.embeddings import default_embedder
    from app.retrieval import QdrantVectorStore, normalize_query

    client = QdrantClient(settings.QDRANT_URL, check_compatibility=False)
    if args.drop:
        for collection in COLLECTIONS.values():
            if client.collection_exists(collection + LAB):
                client.delete_collection(collection + LAB)
        print("scratch collections dropped")
        return

    read = QdrantVectorStore._read
    cases = load_cases(SOURCES)
    names = sorted({p["collection"] for case in cases for p in case["relevant"]})
    labs = {name: build(client, name) for name in names}
    embedder = CachedEmbedder(default_embedder())

    def ranking(points):
        return [(p.id, p.payload["text"], p.payload.get("metadata", {})) for p in points]

    def lexical(variant, lab, text):
        field, query = {"5-letter cut field (now)": (FIELD, bm25_text(text)),
                        "qdrant: words + folding": ("text", text),
                        "qdrant: + Turkish stemmer": ("text_stem", text)}[variant]
        if not query.strip():
            return []
        return ranking(read(client.query_points, lab, query=m.TextQuery(text=query), using=field,
                            limit=DEPTH, with_payload=True).points)

    variants = ["5-letter cut field (now)", "qdrant: words + folding", "qdrant: + Turkish stemmer"]
    forms = {"both": lambda c: [c["rewritten"], c["query"]] if c.get("rewritten") else [c["query"]],
             "raw": lambda c: [c["query"]]}
    dense_cache = {}
    results = {}
    for form, texts_of in forms.items():
        rows = {(v, s): [] for v in variants for s in ("bm25", "hybrid")}
        for case in cases:
            texts = list(dict.fromkeys(normalize_query(t) for t in texts_of(case)))
            wanted = {p["collection"] for p in case["relevant"]}
            for variant in variants:
                lex, fused = {}, {}
                for name in wanted:
                    dense = []
                    for t in texts:
                        key = (name, t)
                        if key not in dense_cache:
                            dense_cache[key] = ranking(read(client.query_points, labs[name],
                                                            query=embedder.embed_query(t), limit=DEPTH,
                                                            with_payload=True).points)
                        dense.append(dense_cache[key])
                    sparse = [lexical(variant, labs[name], t) for t in texts]
                    lex[name] = rrf(sparse, settings.RRF_K)
                    fused[name] = rrf(dense + sparse, settings.RRF_K)
                rows[(variant, "bm25")].append(score(case, lex))
                rows[(variant, "hybrid")].append(score(case, fused))
        results[form] = {key: {metric: sum(r[metric] for r in rs) / len(rs) for metric in ("hit@1", "hit@k", "rr")}
                         for key, rs in rows.items()}
    embedder.save()

    print(f"\n{len(cases)} cases; hybrid = dense + BM25 fused at RRF k={settings.RRF_K}, "
          f"no production filters or list rules\n")
    for form in forms:
        print(f"form: {form}" + (" (what production searches with)" if form == "both" else ""))
        print(f"{'BM25 variant':<29}| {'BM25 alone':^20} | {'hybrid':^20}")
        print(f"{'':<29}| hit@1  hit@k   MRR  | hit@1  hit@k   MRR")
        for variant in variants:
            cells = [results[form][(variant, s)] for s in ("bm25", "hybrid")]
            print(f"{variant:<29}| " + " | ".join(f"{c['hit@1']:.3f}  {c['hit@k']:.3f}  {c['rr']:.3f}" for c in cells))
        print()


if __name__ == "__main__":
    main()
