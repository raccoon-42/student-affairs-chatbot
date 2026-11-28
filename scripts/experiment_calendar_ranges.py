"""Do calendar date ranges get found by "ne zaman bitiyor" with start/end words?

A range event reads "28 Eylül 2026 Pazartesi ile 6 Ocak 2027 Çarşamba
tarihleri arasında ... I. yarıyıl dersler": nothing in it says when the
period ends, so "dönem ne zaman bitiyor?" found the "son günü" events and
the lecture period ranked 15th or lower (top_k is 10). The variant appends
"Başlangıç: ..., bitiş (son gün): ..." to every range event and indexes the
calendar into a scratch collection with the production indexer. Single-day
events are unchanged.

Build:  uv run python scripts/experiment_calendar_ranges.py
Score:  CALENDAR_COLLECTION=calendar__lab_ranges uv run python scripts/eval_retrieval.py
Ranks:  uv run python scripts/experiment_calendar_ranges.py --ranks [--collection NAME]
Clean:  uv run python scripts/experiment_calendar_ranges.py --drop
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LAB = "calendar__lab_ranges"
TAKVIM = ROOT / "preprocessing" / "data" / "processed" / "takvim"
# the phrasings students use for "when does it end"; the lecture period
# ("28 Eylül 2026 ... I. yarıyıl dersler") answers all of them
QUERIES = [("dönem ne zaman bitiyor?", []),
           ("Yarıyıl ne zaman bitiyor?", ["dönem ne zaman bitiyor?"]),
           ("okul ne zaman kapanıyor?", []),
           ("dersler ne zaman bitiyor", []),
           ("dersler ne zaman başlıyor", [])]


def with_start_and_end(render_line):
    from preprocessing.parsers.academic_calendar_parser_llm import format_date_tr

    def render(event):
        line = render_line(event)
        start, end = event.get("start_date"), event.get("end_date")
        if start and end:
            line += f" Başlangıç: {format_date_tr(start)}, bitiş (son gün): {format_date_tr(end)}."
        return line
    return render


def build():
    from preprocessing.indexing import vectorizer
    from preprocessing.parsers import academic_calendar_parser_llm as parser

    parser.render_line = with_start_and_end(parser.render_line)
    chunks = vectorizer.calendar_chunks_from_dir(TAKVIM)
    print("example:", next(c["text"] for c in chunks if c["metadata"]["date2"]))
    vectorizer.store_embedding(chunks, LAB, vectorizer._calendar_payload)


def ranks(collection):
    os.environ["CALENDAR_COLLECTION"] = collection
    from qdrant_client import QdrantClient

    from app.embeddings import default_embedder
    from app.retrieval import QdrantVectorStore, Retriever
    from config import settings
    from scripts.eval_retrieval import CachedEmbedder

    embedder = CachedEmbedder(default_embedder())
    retriever = Retriever(QdrantVectorStore(client=QdrantClient(settings.QDRANT_URL,
                                                                api_key=settings.QDRANT_API_KEY,
                                                                check_compatibility=False)), embedder)
    print(f"collection {settings.CALENDAR_COLLECTION}: rank of the fall lecture period (top_k 10)")
    for query, also in QUERIES:
        hits = retriever.retrieve_calendar(query, top_k=40, vector=embedder.embed_query(query),
                                           also=[(t, embedder.embed_query(t)) for t in also])
        found = [i + 1 for i, h in enumerate(hits)
                 if "28 Eylül 2026" in h["text"] and "I. yarıyıl dersler" in h["text"]]
        print(f"  {query!r:<32} {found[0] if found else '>40'}")
    embedder.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ranks", action="store_true")
    parser.add_argument("--collection", default=LAB)
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()
    if args.drop:
        from qdrant_client import QdrantClient

        from config import settings
        QdrantClient(settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY).delete_collection(LAB)
        print(f"dropped {LAB}")
    elif args.ranks:
        ranks(args.collection)
    else:
        build()


if __name__ == "__main__":
    main()
