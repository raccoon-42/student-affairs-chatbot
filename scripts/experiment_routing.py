"""Can a question skip the collections that clearly don't hold its answer?

Every answer call gets every collection's chunks. Fused (RRF) scores can't
say which collections matter: each search's #1 scores about the same. The
dense cosine of a collection's best chunk can, since every collection is
embedded by the same model. Per eval case, with the phrasings production
searches (the rewrite plus the student's words):

  top(c)  best cosine in collection c over the phrasings (and, where the
          chunks carry question vectors, over those too)
  rules   gap d: keep c if top(c) >= best collection's top - d
          top n: keep the n collections with the highest top(c)

A rule is safe when it keeps every collection a case needs. What it saves
is the chunk text of the dropped collections, measured on the chunks
retrieve_all returns. Free: query vectors come from the eval cache.

Run: uv run python scripts/experiment_routing.py [--source core faq synthetic]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from preprocessing.indexing import questions  # noqa: E402
from scripts.eval_retrieval import COLLECTIONS, SOURCES, CachedEmbedder, load_cases  # noqa: E402

GAPS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
TOPS = (1, 2, 3, 4, 5)


def phrasings(case):
    """The texts production searches, after retrieve_all's casing rule."""
    from app.retrieval import normalize_query
    query = normalize_query(case.get("rewritten") or case["query"])
    others = [normalize_query(case["query"])] if case.get("rewritten") else []
    query = next((t for t in others if t.casefold() == query.casefold()), query)
    return query, [t for t in others if t.casefold() != query.casefold()]


def measure(cases, client, embedder):
    from app.retrieval import QdrantVectorStore, Retriever

    settings.ROUTE_GAP = None  # the chunks sent today, before any routing
    store = QdrantVectorStore(client=client)
    read = QdrantVectorStore._read
    has_questions = {name: questions.has_field(client, c) for name, c in COLLECTIONS.items()}
    rows = []
    for case in cases:
        query, others = phrasings(case)
        vectors = [embedder.embed_query(t) for t in (query, *others)]
        top = {}
        for name, collection in COLLECTIONS.items():
            best = 0.0
            for vector in vectors:
                hit = read(client.query_points, collection, query=vector, limit=1).points
                best = max([best, *(p.score for p in hit)])
                if has_questions[name]:
                    hit = read(client.query_points, collection, query=[vector],
                               using=questions.VECTOR, limit=1).points
                    best = max([best, *(p.score for p in hit)])
            top[name] = best
        retriever = Retriever(store, embedder, today=lambda d=case["as_of"]: d)
        results = retriever.retrieve_all(query, also=others)
        chars = {name: sum(len(r["text"]) for r in results.get(name, [])) for name in COLLECTIONS}
        rows.append({"case": case, "top": top, "chars": chars,
                     "needed": {p["collection"] for p in case["relevant"]}})
    return rows


def kept_by_gap(top, gap):
    best = max(top.values())
    return {name for name, score in top.items() if score >= best - gap}


def kept_by_rank(top, n):
    return set(sorted(top, key=lambda name: -top[name])[:n])


def report(rows):
    n = len(rows)
    total_chars = sum(sum(r["chars"].values()) for r in rows)
    print(f"{n} cases, {total_chars / n:,.0f} chunk characters per question today\n")

    # how far below the best collection each needed collection scores
    gaps = sorted(max(r["top"].values()) - r["top"][c] for r in rows for c in r["needed"])
    ranks = defaultdict(int)
    for r in rows:
        order = sorted(r["top"], key=lambda name: -r["top"][name])
        for c in r["needed"]:
            ranks[order.index(c) + 1] += 1
    print("needed collection's gap below the best: "
          + ", ".join(f"p{q} {gaps[min(len(gaps) - 1, int(len(gaps) * q / 100))]:.3f}"
                      for q in (50, 90, 95, 99)) + f", max {gaps[-1]:.3f}")
    print("needed collection's rank among 9: "
          + ", ".join(f"#{k} {ranks[k]}" for k in sorted(ranks)) + "\n")

    print(f"{'rule':<14} {'safe cases':>11} {'kept/9':>7} {'chars kept':>11}")
    rules = [(f"gap {g:.2f}", lambda top, g=g: kept_by_gap(top, g)) for g in GAPS]
    rules += [(f"top {k}", lambda top, k=k: kept_by_rank(top, k)) for k in TOPS]
    # calendar chunks are one short event line each, so they score low
    # against any question; they are also the cheapest collection to keep
    rules += [(f"gap {g:.2f}+cal", lambda top, g=g: kept_by_gap(top, g) | {"calendar"})
              for g in (0.08, 0.10, 0.12, 0.15)]
    misses = {}
    for label, rule in rules:
        safe, kept_n, kept_chars, missed = 0, 0, 0, []
        for r in rows:
            kept = rule(r["top"])
            kept_n += len(kept)
            kept_chars += sum(r["chars"][c] for c in kept)
            if r["needed"] <= kept:
                safe += 1
            else:
                missed.append((r, r["needed"] - kept))
        misses[label] = missed
        print(f"{label:<14} {safe:>5}/{n:<5} {kept_n / n:>7.1f} {kept_chars / total_chars:>10.0%}")

    by_source = defaultdict(list)
    for r in rows:
        by_source[r["case"]["source"]].append(r)
    print("\nsafe cases by source, gap 0.10 | top 3 | gap 0.20")
    for source, part in sorted(by_source.items()):
        gap_safe = sum(r["needed"] <= kept_by_gap(r["top"], 0.10) for r in part)
        top_safe = sum(r["needed"] <= kept_by_rank(r["top"], 3) for r in part)
        wide_safe = sum(r["needed"] <= kept_by_gap(r["top"], 0.20) for r in part)
        print(f"  {source:<10} {len(part):>4}  {gap_safe:>4} | {top_safe:>4} | {wide_safe:>4}")

    for label in ("gap 0.10", "top 3"):
        print(f"\nmisses at {label}")
        for r, lost in misses[label]:
            best = max(r["top"].values())
            order = sorted(r["top"], key=lambda name: -r["top"][name])
            lost = ", ".join(f"{c} -{best - r['top'][c]:.3f}" for c in sorted(lost))
            print(f"  [{r['case']['source']}/{r['case']['id']}] {r['case']['query'][:70]!r}"
                  f"  best {order[0]} {best:.3f}, lost {lost}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", nargs="+", choices=SOURCES, default=list(SOURCES))
    args = parser.parse_args()

    from qdrant_client import QdrantClient

    from app.embeddings import default_embedder

    cases = load_cases(args.source)
    client = QdrantClient(settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY,
                          check_compatibility=False)
    embedder = CachedEmbedder(default_embedder())
    try:
        rows = measure(cases, client, embedder)
    finally:
        embedder.save()
    print(f"{embedder.misses} new embeddings")
    report(rows)


if __name__ == "__main__":
    main()
