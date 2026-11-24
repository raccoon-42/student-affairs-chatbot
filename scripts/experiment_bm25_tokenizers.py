"""BM25 tokenizer variants scored on the retrieval eval set, offline.

Production cuts every word to its first 5 letters so Turkish inflections
("sınav", "sınavı", "sınavlar") share a token. This compares that with
other prefix lengths, whole words, and a Turkish stemmer: BM25 is rebuilt
locally for each variant (same weights and IDF as Qdrant), then scored
alone and fused with the dense ranking the way production fuses them.
Nothing is re-indexed. Dense rankings come from Qdrant, query embeddings
from the eval cache.

Run (prefix variants only):
  uv run python scripts/experiment_bm25_tokenizers.py
With the Turkish stemmer (snowballstemmer, not a project dependency):
  uv run --with snowballstemmer python scripts/experiment_bm25_tokenizers.py
"""
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from preprocessing.indexing.bm25 import B, K1, _NON_WORD  # noqa: E402
from scripts.eval_retrieval import (COLLECTIONS, SOURCES, CachedEmbedder,  # noqa: E402
                                    load_cases, rrf, score)

DEPTH = settings.HYBRID_PREFETCH  # how deep each ranking goes before fusion, as in production


def fold(text):
    """bm25.tokenize's ASCII folding, without the cut."""
    text = unicodedata.normalize("NFKD", text.replace("ı", "i"))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def words(text):
    return [w for w in _NON_WORD.split(text) if w]


def prefix_tokenizer(length):
    def tokenize(text):
        return [w if any(c.isdigit() for c in w) or length is None else w[:length]
                for w in words(fold(text.lower()))]
    return tokenize


def stem_tokenizer(cut=None):
    import snowballstemmer
    stemmer = snowballstemmer.stemmer("turkish")

    def tokenize(text):
        # stem the Turkish spelling (suffix rules follow vowel harmony),
        # then fold, so "sınavlar" and a typed "sinavlar" still meet
        text = text.replace("I", "ı").replace("İ", "i").lower()
        out = []
        for w in words(text):
            if any(c.isdigit() for c in w):
                out.append(w)
                continue
            token = fold(stemmer.stemWord(w))
            out.append(token[:cut] if cut else token)
        return out
    return tokenize


class LocalBM25:
    """Qdrant's sparse search over one collection, for one tokenizer:
    stored per-word weights (saturation + length normalization), IDF over
    the collection at query time, query weight 1 per distinct word."""

    def __init__(self, chunks, tokenize):
        self.chunks = chunks
        self.tokenize = tokenize
        docs = [tokenize(text) for text, _ in chunks]
        avg = sum(map(len, docs)) / len(docs)
        self.weights = []
        for tokens in docs:
            norm = 1 - B + B * len(tokens) / avg
            self.weights.append({t: tf * (K1 + 1) / (tf + K1 * norm) for t, tf in Counter(tokens).items()})
        df = Counter(t for w in self.weights for t in w)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - k + 0.5) / (k + 0.5)) for t, k in df.items()}

    def rank(self, query):
        terms = set(self.tokenize(query))
        scored = [(sum(self.idf.get(t, 0.0) * w.get(t, 0.0) for t in terms), i)
                  for i, w in enumerate(self.weights)]
        scored = sorted((s, i) for s, i in scored if s > 0)[::-1][:DEPTH]
        return [(i, *self.chunks[i]) for _, i in scored]


def chunk_id(text, metadata):
    # both rankings must name a chunk the same way; text alone would merge
    # the FAQ questions that exist once per audience
    return text, json.dumps(metadata, sort_keys=True, ensure_ascii=False)


def main():
    from qdrant_client import QdrantClient

    from app.embeddings import default_embedder
    from app.retrieval import QdrantVectorStore, normalize_query

    variants = {f"prefix {n}" + (" (now)" if n == 5 else ""): prefix_tokenizer(n) for n in (4, 5, 6, 7)}
    variants["whole words"] = prefix_tokenizer(None)
    try:
        variants["stemmer"] = stem_tokenizer()
        variants["stemmer, cut to 6"] = stem_tokenizer(6)
    except ImportError:
        print("snowballstemmer not installed: stemmer variants skipped (see the docstring)\n")

    cases = load_cases(SOURCES)
    client = QdrantClient(settings.QDRANT_URL, check_compatibility=False)
    embedder = CachedEmbedder(default_embedder())
    names = sorted({p["collection"] for case in cases for p in case["relevant"]})

    chunks = {}
    for name in names:
        chunks[name], offset = [], None
        while True:
            points, offset = client.scroll(COLLECTIONS[name], limit=1000, offset=offset, with_payload=True)
            chunks[name] += [(p.payload["text"], p.payload.get("metadata", {})) for p in points]
            if offset is None:
                break

    dense_cache = {}

    def dense(text, name):
        key = (text, name)
        if key not in dense_cache:
            points = QdrantVectorStore._read(client.query_points, COLLECTIONS[name],
                                             query=embedder.embed_query(text), limit=DEPTH,
                                             with_payload=True).points
            dense_cache[key] = [(chunk_id(p.payload["text"], p.payload.get("metadata", {})),
                                 p.payload["text"], p.payload.get("metadata", {})) for p in points]
        return dense_cache[key]

    # production searches with the rewrite plus the student's own words
    forms = {"both": lambda c: [c["rewritten"], c["query"]] if c.get("rewritten") else [c["query"]],
             "raw": lambda c: [c["query"]]}
    results = {}
    for label, tokenize in variants.items():
        indexes = {name: LocalBM25(chunks[name], tokenize) for name in names}
        for form, texts_of in forms.items():
            rows = {"bm25": [], "hybrid": []}
            for case in cases:
                texts = list(dict.fromkeys(normalize_query(t) for t in texts_of(case)))
                lexical, fused = {}, {}
                for name in {p["collection"] for p in case["relevant"]}:
                    sparse = [[(chunk_id(text, md), text, md) for _, text, md in indexes[name].rank(t)]
                              for t in texts]
                    lexical[name] = rrf(sparse, settings.RRF_K)
                    fused[name] = rrf([dense(t, name) for t in texts] + sparse, settings.RRF_K)
                rows["bm25"].append(score(case, lexical))
                rows["hybrid"].append(score(case, fused))
            results[(label, form)] = {
                system: {m: sum(r[m] for r in rs) / len(rs) for m in ("hit@1", "hit@k", "rr")}
                for system, rs in rows.items()}
    embedder.save()

    print(f"{len(cases)} cases; hybrid = dense + BM25 fused with RRF k={settings.RRF_K} "
          f"(no production filters or list rules)\n")
    for form in forms:
        print(f"form: {form}" + (" (what production searches with)" if form == "both" else ""))
        print(f"{'tokenizer':<20}| {'BM25 alone':^20} | {'hybrid':^20}")
        print(f"{'':<20}| hit@1  hit@k   MRR  | hit@1  hit@k   MRR")
        for label in variants:
            r = results[(label, form)]
            print(f"{label:<20}| " + " | ".join(
                f"{r[s]['hit@1']:.3f}  {r[s]['hit@k']:.3f}  {r[s]['rr']:.3f}" for s in ("bm25", "hybrid")))
        print()


if __name__ == "__main__":
    main()
