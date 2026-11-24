"""QdrantVectorStore and the BM25 migration against qdrant-client's local
mode: the real hybrid query (prefetch + RRF, filters, Qdrant's BM25 text
index) without a server."""
from qdrant_client import QdrantClient
from qdrant_client.http.models import (Distance, Modifier, PointStruct, SparseVector,
                                       SparseVectorParams, VectorParams)

from app.retrieval import QdrantVectorStore
from preprocessing.indexing.bm25 import FIELD, bm25_text, index_params
from preprocessing.indexing.vectorizer import OLD_SPARSE_VECTOR, rebuild_bm25

CHUNKS = [  # text, kind, dense vector
    ("Genel hükümler", "regulation", [1.0, 0.0, 0.0]),
    ("Amaç ve kapsam", "regulation", [0.9, 0.1, 0.0]),
    ("Madde 322: mazeret sınavı", "regulation", [0.0, 0.0, 1.0]),
    ("Madde 322 başka yönetmelik", "other", [0.0, 0.0, 1.0]),
]
QUERY = [1.0, 0.0, 0.0]  # dense favors the generic articles


def make_client(with_bm25, old_sparse=False):
    """with_bm25: indexed the current way. old_sparse: the pre-ADR-0004 shape,
    dense plus a sparse BM25 vector and no BM25 field."""
    client = QdrantClient(":memory:")
    client.create_collection(
        "c", vectors_config=VectorParams(size=3, distance=Distance.COSINE),
        sparse_vectors_config=({OLD_SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF)}
                               if old_sparse else None))
    if with_bm25:
        client.create_payload_index("c", FIELD, index_params())
    client.upsert("c", points=[
        PointStruct(id=i,
                    vector={"": dense, OLD_SPARSE_VECTOR: SparseVector(indices=[1], values=[1.0])}
                    if old_sparse else {"": dense},
                    payload={"text": text, "metadata": {"kind": kind},
                             **({FIELD: bm25_text(text)} if with_bm25 else {})})
        for i, (text, kind, dense) in enumerate(CHUNKS)])
    return client


def test_bm25_lifts_an_exact_match_the_dense_search_ranks_low():
    store = QdrantVectorStore(client=make_client(with_bm25=True))
    hits = store.search("c", QUERY, limit=1, filters={"kind": "regulation"},
                        lexical=bm25_text("madde 322"))
    assert hits[0].text == "Madde 322: mazeret sınavı"


def test_filter_applies_to_both_rankings():
    store = QdrantVectorStore(client=make_client(with_bm25=True))
    hits = store.search("c", QUERY, limit=10, filters={"kind": "regulation"},
                        lexical=bm25_text("madde 322"))
    assert {hit.metadata["kind"] for hit in hits} == {"regulation"}


def test_collection_without_bm25_falls_back_to_dense():
    store = QdrantVectorStore(client=make_client(with_bm25=False))
    hits = store.search("c", QUERY, limit=1, lexical=bm25_text("madde 322"))
    assert hits[0].text == "Genel hükümler"


def test_rebuild_bm25_migrates_a_collection_from_sparse_vectors():
    client = make_client(with_bm25=False, old_sparse=True)
    rebuild_bm25(client, "c")
    assert OLD_SPARSE_VECTOR not in (client.get_collection("c").config.params.sparse_vectors or {})
    hits = QdrantVectorStore(client=client).search(
        "c", QUERY, limit=1, filters={"kind": "regulation"}, lexical=bm25_text("madde 322"))
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


def test_guide_questions_are_searched_but_never_returned(monkeypatch):
    """A chunk's student questions go into its embedding and its BM25 field;
    the hit the model gets is the chunk text alone."""
    import qdrant_client

    from app import embeddings
    from preprocessing.indexing import vectorizer

    client = QdrantClient(":memory:")
    embedded = []

    class FakeEmbedder:
        def embed_documents(self, texts):
            embedded.extend(texts)
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: client)
    monkeypatch.setattr(embeddings, "default_embedder", FakeEmbedder)
    dates = {"text": "Başvuru Tarihleri\n17 Mart - 2 Nisan arası online", "metadata": {"topic": "erasmus"},
             "questions": ["erasmus başvuruları ne zaman"]}
    rules = {"text": "Asgari Şartlar\nLisans için en az 2.20 ortalama", "metadata": {"topic": "erasmus"}}

    vectorizer.store_embedding([dates, rules], "g", vectorizer._sks_payload)

    assert sorted(embedded) == sorted([dates["text"] + "\nerasmus başvuruları ne zaman", rules["text"]])
    hits = QdrantVectorStore(client=client).search("g", [1.0, 0.0, 0.0], limit=1,
                                                   lexical=bm25_text("ne zaman"))
    assert hits[0].text == dates["text"]  # found by its question, returned without it

    # new questions change the point id: only that chunk is embedded again
    embedded.clear()
    vectorizer.store_embedding([{**dates, "questions": ["erasmus son gün"]}, rules], "g",
                               vectorizer._sks_payload)
    assert embedded == [dates["text"] + "\nerasmus son gün"]

    # the BM25 rebuild keeps the questions in the field
    rebuild_bm25(client, "g")
    hits = QdrantVectorStore(client=client).search("g", [1.0, 0.0, 0.0], limit=1,
                                                   lexical=bm25_text("son gün"))
    assert hits[0].text == dates["text"]


def test_collections_are_kept_in_one_segment(monkeypatch):
    """Segments searched in parallel returned equal-score chunks in a new
    order every run. Local mode has no segments, so this checks the setting
    is asked for; the order itself was checked against the server."""
    import qdrant_client

    from app import embeddings
    from preprocessing.indexing import vectorizer

    client = QdrantClient(":memory:")
    asked = []
    for method in ("create_collection", "update_collection"):
        real = getattr(client, method)
        monkeypatch.setattr(client, method, lambda *a, _real=real, _name=method, **k: (
            asked.append((_name, k["optimizers_config"].default_segment_number))
            if k.get("optimizers_config") else None) or _real(*a, **k))

    class FakeEmbedder:
        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: client)
    monkeypatch.setattr(embeddings, "default_embedder", FakeEmbedder)
    chunk = {"text": "Final sınavları", "metadata": {}}

    vectorizer.store_embedding([chunk], "c", vectorizer._sks_payload)  # a new collection
    vectorizer.store_embedding([chunk], "c", vectorizer._sks_payload)  # an existing one

    assert asked[0] == ("create_collection", 1)
    assert ("update_collection", 1) in asked[1:]


def test_equal_fused_scores_are_ordered_by_point_id():
    # dense ranks A over B, BM25 ranks B over A: the same RRF score. The
    # server returns such ties in no fixed order (local mode happens to sort
    # them, so the server's order is simulated by reversing its answer)
    a = ("00000000-0000-0000-0000-000000000001", "alpha beta", [1.0, 0.0, 0.0])
    b = ("00000000-0000-0000-0000-000000000002", "beta gamma", [0.6, 0.8, 0.0])
    client = QdrantClient(":memory:")
    client.create_collection("t", vectors_config=VectorParams(size=3, distance=Distance.COSINE))
    client.create_payload_index("t", FIELD, index_params())
    client.upsert("t", points=[PointStruct(id=pid, vector={"": vector},
                                           payload={"text": text, FIELD: bm25_text(text)})
                               for pid, text, vector in (a, b)])
    real_query = client.query_points

    def any_order(*args, **kwargs):
        response = real_query(*args, **kwargs)
        response.points.reverse()
        return response

    client.query_points = any_order
    hits = QdrantVectorStore(client=client).search("t", [1.0, 0.1, 0.0], limit=2,
                                                   lexical=bm25_text("beta gamma"))
    assert hits[0].score == hits[1].score
    assert [hit.text for hit in hits] == ["alpha beta", "beta gamma"]
