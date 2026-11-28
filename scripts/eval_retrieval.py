"""Retrieval eval: does the chunk that answers a question come back?

Each case names the chunk(s) that answer it by collection plus text or
metadata to match, not by point id: ids are content hashes and change
whenever a chunk is re-indexed with new text. Two views:

- production: Retriever.retrieve_all, i.e. exactly what the model sees
  (per-collection top_k, calendar filters, people/courses list tiers,
  FAQ dedupe), with the date pinned to the dataset's as_of
- lab: dense and BM25 rankings fetched per collection and fused offline
  at several RRF k, i.e. the fusion knobs without the production extras

Every case is scored three ways: its raw query, its frozen rewrite, and
both together (what production searches with: the rewrite, plus the
student's own words as extra rankings). Query embeddings are cached in data/eval_cache,
so only new queries cost an embedding call.

Run: uv run python scripts/eval_retrieval.py [--source core faq] [--misses]
The dataset lives in tests/data/retrieval_eval/; build the generated
parts with scripts/build_retrieval_eval.py.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402

DATASET_DIR = ROOT / "tests" / "data" / "retrieval_eval"
CACHE = ROOT / "data" / "eval_cache" / "embeddings.npz"
SOURCES = ("core", "faq", "synthetic")
FORMS = ("raw", "rewritten", "both")
RRF_KS = (1, 2, 5, 10, 20, 60)
COLLECTIONS = {
    "calendar": settings.CALENDAR_COLLECTION, "regulations": settings.REGULATIONS_COLLECTION,
    "faq": settings.FAQ_COLLECTION, "forms": settings.FORMS_COLLECTION,
    "sks": settings.SKS_COLLECTION, "programs": settings.PROGRAMS_COLLECTION,
    "people": settings.PEOPLE_COLLECTION, "courses": settings.COURSES_COLLECTION,
    "guides": settings.GUIDES_COLLECTION,
}
# what Retriever.retrieve_all asks each collection for; the lab's hit@k
# uses the same cut so its numbers compare with production
TOP_K = {"calendar": 10, "regulations": 3, "faq": 4, "forms": 2, "sks": 2,
         "programs": 3, "people": 3, "courses": 3, "guides": 4}


def matches(predicate, text, metadata):
    """A chunk answers the case when every given condition holds."""
    low = text.casefold()
    if any(part.casefold() not in low for part in predicate.get("contains", [])):
        return False
    return all(metadata.get(key) == value for key, value in predicate.get("metadata", {}).items())


def first_rank(predicate, ranking):
    """0-based rank of the first matching chunk in [(text, metadata)], or None."""
    return next((rank for rank, (text, metadata) in enumerate(ranking)
                 if matches(predicate, text, metadata)), None)


def score(case, rankings, sent=False):
    """Ranks of the case's relevant chunks in rankings[collection]. sent:
    the rankings are what production sends (top_k plus anything added
    after it, like the teaching period), so all of each counts as kept."""
    ranks = [first_rank(p, rankings.get(p["collection"], [])) for p in case["relevant"]]
    found = [r for r in ranks if r is not None]
    within_k = [r is not None and (sent or r < TOP_K[p["collection"]])
                for p, r in zip(case["relevant"], ranks)]
    best = min(found) if found else None
    return {
        "best": best,
        "hit@1": best == 0,
        "hit@3": best is not None and best < 3,
        "hit@k": any(within_k),
        "recall@k": sum(within_k) / len(within_k),
        "rr": 1 / (best + 1) if best is not None else 0.0,
    }


def rrf(rankings, k):
    """Offline RRF, as Qdrant computes it: 1 / (rank + k), rank from 0."""
    scores, items = {}, {}
    for ranking in rankings:
        for rank, (pid, text, metadata) in enumerate(ranking):
            scores[pid] = scores.get(pid, 0.0) + 1 / (rank + k)
            items[pid] = (text, metadata)
    return [items[pid] for pid in sorted(scores, key=lambda pid: -scores[pid])]


class CachedEmbedder:
    """The production embedder with a disk cache keyed by model + text."""

    def __init__(self, inner, path=CACHE):
        self._inner = inner
        self._path = path
        self._vectors = dict(np.load(path)) if path.exists() else {}
        self.misses = 0

    def _key(self, text):
        return hashlib.sha1(f"{settings.OPENROUTER_EMBEDDING_MODEL}\0{text}".encode()).hexdigest()

    def embed_query(self, text):
        key = self._key(text)
        if key not in self._vectors:
            self._vectors[key] = np.array(self._inner.embed_query(text), dtype=np.float32)
            self.misses += 1
        return self._vectors[key].tolist()

    def save(self):
        if self.misses:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self._path, **self._vectors)


def load_cases(sources):
    cases = []
    for source in sources:
        path = DATASET_DIR / f"{source}.json"
        if not path.exists():
            print(f"skipping {source}: {path.relative_to(ROOT)} not built yet")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        as_of = date.fromisoformat(data["as_of"])
        cases += [{**case, "source": source, "as_of": as_of} for case in data["cases"]]
    return cases


def stale_labels(cases, client):
    """Labels that match no chunk at all: the corpus changed under them,
    so their case would count as a retrieval miss when it isn't one."""
    chunks = {}
    for name in {p["collection"] for case in cases for p in case["relevant"]}:
        chunks[name], offset = [], None
        while True:
            points, offset = client.scroll(COLLECTIONS[name], limit=1000, offset=offset, with_payload=True)
            chunks[name] += [(p.payload["text"], p.payload.get("metadata", {})) for p in points]
            if offset is None:
                break
    return [(case, p) for case in cases for p in case["relevant"]
            if first_rank(p, chunks[p["collection"]]) is None]


def run(cases, make_client, embedder):
    from qdrant_client.http.models import TextQuery

    from app.retrieval import QdrantVectorStore, Retriever, normalize_query
    from preprocessing.indexing.bm25 import FIELD, bm25_text

    # the lab's own queries get the store's retry too: a request
    # qdrant-client drops must not count as a retrieval miss
    retried = QdrantVectorStore._read
    client = make_client()

    def lab_rankings(text, names):
        """(dense, bm25) ranking of each named collection for one text."""
        query = normalize_query(text)
        vector = embedder.embed_query(query)
        words = bm25_text(query)
        rankings = {}
        for name in names:
            depth = settings.HYBRID_PREFETCH
            dense = retried(client.query_points, COLLECTIONS[name], query=vector, limit=depth,
                            with_payload=True).points
            lexical = retried(client.query_points, COLLECTIONS[name], query=TextQuery(text=words),
                              using=FIELD, limit=depth, with_payload=True).points if words else []
            rankings[name] = tuple([(p.id, p.payload["text"], p.payload.get("metadata", {}))
                                    for p in points] for points in (dense, lexical))
        return rankings

    def strip(ranking):
        return [(text, metadata) for _, text, metadata in ranking]

    records = []
    for case in cases:
        retriever = Retriever(QdrantVectorStore(client=client), embedder,
                              today=lambda d=case["as_of"]: d)
        names = {p["collection"] for p in case["relevant"]}
        texts = {"raw": case["query"], "rewritten": case.get("rewritten")}
        lab = {form: lab_rankings(text, names) for form, text in texts.items() if text}
        for form in FORMS:
            if form == "both":
                if not texts["rewritten"]:
                    continue
                text = texts["rewritten"]
                results = retriever.retrieve_all(text, also=[texts["raw"]])
                parts = ("rewritten", "raw")
            else:
                text = texts[form]
                if not text:
                    continue
                results = retriever.retrieve_all(text)
                parts = (form,)
            production = {name: [(r["text"], r["metadata"]) for r in results.get(name, [])]
                          for name in COLLECTIONS}
            record = {"case": case, "form": form, "text": text,
                      "production": production,
                      "scores": {"production": score(case, production, sent=True)}}
            # one phrasing: the plain rankings; two: their RRF, like production
            dense = {name: [lab[part][name][0] for part in parts] for name in names}
            lexical = {name: [lab[part][name][1] for part in parts] for name in names}
            for system, lists in (("dense", dense), ("bm25", lexical)):
                record["scores"][system] = score(case, {
                    name: strip(ranking[0]) if len(ranking) == 1 else rrf(ranking, settings.RRF_K)
                    for name, ranking in lists.items()})
            for k in RRF_KS:
                record["scores"][f"rrf k={k}"] = score(case, {
                    name: rrf(dense[name] + lexical[name], k) for name in names})
            records.append(record)
    return records


def summarize(records, system, form, subset=None):
    rows = [r["scores"][system] for r in records
            if r["form"] == form and (subset is None or subset(r))]
    if not rows:
        return None
    n = len(rows)
    return {"n": n, **{m: sum(row[m] for row in rows) / n
                       for m in ("hit@1", "hit@3", "hit@k", "recall@k", "rr")}}


def print_report(records, show_misses):
    systems = list(records[0]["scores"]) if records else []
    print("\nhit@k = a relevant chunk within the collection's production top_k "
          "(what the model would see); MRR over the collection's ranking\n")
    header = f"{'system':<12}" + "".join(
        f"| {form:<9} n  hit@1 hit@3 hit@k   MRR " for form in FORMS)
    print(header)
    print("-" * len(header))
    for system in systems:
        line = f"{system:<12}"
        for form in FORMS:
            s = summarize(records, system, form)
            line += (f"| {'':<9}{s['n']:>3}  {s['hit@1']:.2f}  {s['hit@3']:.2f}  {s['hit@k']:.2f}  {s['rr']:.2f} "
                     if s else f"| {'':<36}")
        print(line)

    def latest(r, form):
        # the case's form when it has one, else its raw query
        return r["form"] == form or (not r["case"].get("rewritten") and r["form"] == "raw")

    print("\nhit@k by category")
    categories = sorted({r["case"]["category"] for r in records})
    shown = (("production", "rewritten"), ("production", "both"), ("dense", "rewritten"),
             ("bm25", "rewritten"), ("rrf k=2", "rewritten"), ("rrf k=2", "both"))
    print(f"{'category':<14}{'n':>4}  " + "  ".join(
        f"{system[:10]:>10}" for system, _ in shown))
    print(f"{'':<18}  " + "  ".join(f"{form:>10}" for _, form in shown))
    for category in categories:
        cells, n = [], 0
        for system, form in shown:
            rows = [r["scores"][system]["hit@k"] for r in records
                    if r["case"]["category"] == category and latest(r, form)]
            n = max(n, len(rows))
            cells.append(f"{sum(rows) / len(rows):>10.2f}" if rows else f"{'-':>10}")
        print(f"{category:<14}{n:>4}  " + "  ".join(cells))

    if show_misses:
        print("\nproduction misses (rewrite + the student's words, else raw)")
        for r in records:
            if not latest(r, "both") or r["scores"]["production"]["hit@k"]:
                continue
            case = r["case"]
            print(f"\n- [{case['source']}/{case['id']}] {case['query']!r}"
                  + (f" -> {r['text']!r}" if r["form"] == "both" else ""))
            for p in case["relevant"]:
                want = {key: p[key] for key in ("contains", "metadata") if key in p}
                print(f"    want {p['collection']}: {json.dumps(want, ensure_ascii=False)[:150]}")
                for rank, (text, _) in enumerate(r["production"][p["collection"]][:3]):
                    print(f"    got  #{rank} {text[:110].replace(chr(10), ' | ')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--misses", action="store_true", help="list the production misses")
    parser.add_argument("--out", type=Path, help="write per-case ranks as JSON")
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    from app.embeddings import default_embedder

    cases = load_cases(args.source)
    if not cases:
        sys.exit("no cases")
    def make_client():
        return QdrantClient(settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY,
                            check_compatibility=False)

    stale = stale_labels(cases, make_client())
    for case, predicate in stale:
        print(f"stale label in {case['source']}/{case['id']}: {json.dumps(predicate, ensure_ascii=False)}")
    if stale:
        sys.exit(f"{len(stale)} labels match no chunk; fix them before scoring")
    embedder = CachedEmbedder(default_embedder())
    try:
        records = run(cases, make_client, embedder)
    finally:
        embedder.save()
    print(f"{len(cases)} cases, {embedder.misses} new embeddings")
    print_report(records, args.misses)
    if args.out:
        args.out.write_text(json.dumps([
            {"source": r["case"]["source"], "id": r["case"]["id"], "form": r["form"],
             "text": r["text"], "scores": r["scores"]} for r in records],
            ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
