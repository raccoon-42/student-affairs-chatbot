"""Merge every collection into a single segment, so chunks with equal scores
always rank in the same order (see vectorizer.keep_one_segment). No
embedding calls, nothing re-uploaded: Qdrant merges in place. Run once for
collections indexed before this; the vectorizer keeps it so afterwards.

    python -m preprocessing.indexing.one_segment
"""
import time

from config import settings
from preprocessing.indexing.reindex_bm25 import COLLECTIONS
from preprocessing.indexing.vectorizer import keep_one_segment


def main():
    from qdrant_client import QdrantClient

    qdrant = QdrantClient(settings.QDRANT_URL)
    names = [name for name in COLLECTIONS if qdrant.collection_exists(collection_name=name)]
    for name in names:
        keep_one_segment(qdrant, name)
    for _ in range(120):  # the merge runs in the background, usually ~1s
        counts = {name: qdrant.get_collection(name).segments_count for name in names}
        if all(count == 1 for count in counts.values()):
            break
        time.sleep(1)
    for name, count in counts.items():
        print(f"{name}: {count} segment{'s' if count != 1 else ''}")


if __name__ == "__main__":
    main()
