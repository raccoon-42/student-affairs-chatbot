"""Write BM25 vectors for every collection from the stored chunk text — no
embedding calls. Run once to move existing collections to hybrid search,
and again after any change to bm25.tokenize().

    python -m preprocessing.indexing.reindex_bm25
"""
from config import settings
from preprocessing.indexing.vectorizer import rebuild_bm25

COLLECTIONS = (
    settings.CALENDAR_COLLECTION, settings.REGULATIONS_COLLECTION, settings.FAQ_COLLECTION,
    settings.FORMS_COLLECTION, settings.SKS_COLLECTION, settings.PROGRAMS_COLLECTION,
    settings.PEOPLE_COLLECTION, settings.COURSES_COLLECTION, settings.GUIDES_COLLECTION,
)


def main():
    from qdrant_client import QdrantClient

    qdrant = QdrantClient(settings.QDRANT_URL)
    for name in COLLECTIONS:
        if not qdrant.collection_exists(collection_name=name):
            print(f"{name}: not indexed yet, skipping")
            continue
        rebuild_bm25(qdrant, name)


if __name__ == "__main__":
    main()
