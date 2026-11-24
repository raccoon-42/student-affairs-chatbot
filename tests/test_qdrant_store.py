"""QdrantVectorStore and the BM25 migration against qdrant-client's local
mode: the real hybrid query (prefetch + RRF, filters, sparse vector)
without a server."""
from qdrant_client import QdrantClient
from qdrant_client.http.models import (Distance, Modifier, PointStruct, SparseVectorParams,
                                       VectorParams)

from app.retrieval import QdrantVectorStore
from preprocessing.indexing.bm25 import VECTOR_NAME, document_vectors, sparse_query
from preprocessing.indexing.vectorizer import _sparse_vector, rebuild_bm25

CHUNKS = [  # text, kind, dense vector
    ("Genel hükümler", "regulation", [1.0, 0.0, 0.0]),
    ("Amaç ve kapsam", "regulation", [0.9, 0.1, 0.0]),
    ("Madde 322: mazeret sınavı", "regulation", [0.0, 0.0, 1.0]),
    ("Madde 322 başka yönetmelik", "other", [0.0, 0.0, 1.0]),
]
QUERY = [1.0, 0.0, 0.0]  # dense favors the generic articles


def make_client(with_bm25):
    client = QdrantClient(":memory:")
    client.create_collection(
        "c", vectors_config=VectorParams(size=3, distance=Distance.COSINE),
        sparse_vectors_config=({VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF)}
                               if with_bm25 else None))
    sparse = document_vectors({i: text for i, (text, _, _) in enumerate(CHUNKS)})
    client.upsert("c", points=[
        PointStruct(id=i,
                    vector={"": dense, VECTOR_NAME: _sparse_vector(sparse[i])} if with_bm25 else dense,
                    payload={"text": text, "metadata": {"kind": kind}})
        for i, (text, kind, dense) in enumerate(CHUNKS)])
    return client


def test_bm25_lifts_an_exact_match_the_dense_search_ranks_low():
    store = QdrantVectorStore(client=make_client(with_bm25=True))
    hits = store.search("c", QUERY, limit=1, filters={"kind": "regulation"},
                        sparse=sparse_query("madde 322"))
    assert hits[0].text == "Madde 322: mazeret sınavı"


def test_filter_applies_to_both_rankings():
    store = QdrantVectorStore(client=make_client(with_bm25=True))
    hits = store.search("c", QUERY, limit=10, filters={"kind": "regulation"},
                        sparse=sparse_query("madde 322"))
    assert {hit.metadata["kind"] for hit in hits} == {"regulation"}


def test_collection_without_bm25_falls_back_to_dense():
    store = QdrantVectorStore(client=make_client(with_bm25=False))
    hits = store.search("c", QUERY, limit=1, sparse=sparse_query("madde 322"))
    assert hits[0].text == "Genel hükümler"


def test_rebuild_bm25_migrates_a_dense_only_collection():
    client = make_client(with_bm25=False)
    rebuild_bm25(client, "c")
    hits = QdrantVectorStore(client=client).search(
        "c", QUERY, limit=1, filters={"kind": "regulation"}, sparse=sparse_query("madde 322"))
    assert hits[0].text == "Madde 322: mazeret sınavı"


def test_a_dropped_request_is_sent_again():
    # qdrant-client loses ~1 in 600 requests under parallel load; one lost
    # search used to fail the student's whole message
    from qdrant_client.http.exceptions import ResponseHandlingException

    client = make_client(with_bm25=True)
    real_query = client.query_points
    dropped = []

    def flaky_query(*args, **kwargs):
        if not dropped:
            dropped.append(True)
            raise ResponseHandlingException(OSError(9, "Bad file descriptor"))
        return real_query(*args, **kwargs)

    client.query_points = flaky_query
    hits = QdrantVectorStore(client=client).search("c", QUERY, limit=1)
    assert dropped and hits[0].text == "Genel hükümler"
