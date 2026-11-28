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


def test_a_list_filter_matches_any_of_its_values():
    store = QdrantVectorStore(client=make_client(with_bm25=True))
    hits = store.search("c", QUERY, limit=10, filters={"kind": ["regulation", "other"]})
    assert {hit.metadata["kind"] for hit in hits} == {"regulation", "other"}
    hits = store.search("c", QUERY, limit=10, filters={"kind": ["other"]})
    assert {hit.metadata["kind"] for hit in hits} == {"other"}


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

    # the text with its questions, and each question once more on its own (docs/adr/0011)
    assert sorted(embedded) == sorted([dates["text"] + "\nerasmus başvuruları ne zaman", rules["text"],
                                       "erasmus başvuruları ne zaman"])
    hits = QdrantVectorStore(client=client).search("g", [1.0, 0.0, 0.0], limit=1,
                                                   lexical=bm25_text("ne zaman"))
    assert hits[0].text == dates["text"]  # found by its question, returned without it

    # new questions change the point id: only that chunk is embedded again
    embedded.clear()
    vectorizer.store_embedding([{**dates, "questions": ["erasmus son gün"]}, rules], "g",
                               vectorizer._sks_payload)
    assert embedded == ["erasmus son gün", dates["text"] + "\nerasmus son gün"]

    # the BM25 rebuild keeps the questions in the field
    rebuild_bm25(client, "g")
    hits = QdrantVectorStore(client=client).search("g", [1.0, 0.0, 0.0], limit=1,
                                                   lexical=bm25_text("son gün"))
    assert hits[0].text == dates["text"]


def test_a_chunk_is_found_by_its_best_question():
    """docs/adr/0011: question vectors are a third ranking. B's text is only
    the second closest, but one of its questions matches the query exactly."""
    from preprocessing.indexing import questions

    def collection(with_questions):
        client = QdrantClient(":memory:")
        dense = VectorParams(size=3, distance=Distance.COSINE)
        client.create_collection("g", vectors_config=(
            {"": dense, questions.VECTOR: questions.params(3)} if with_questions else dense))
        chunks = [  # text, text vector, question vectors
            ("A", [1.0, 0.0, 0.0], [[0.0, 0.0, 1.0]]),
            ("B", [0.8, 0.6, 0.0], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            ("C, no questions", [0.0, 1.0, 0.0], None),
            ("D", [0.0, 0.0, 1.0], [[0.6, 0.8, 0.0]]),
        ]
        client.upsert("g", points=[
            PointStruct(id=i, vector={"": text_vector, **(
                {questions.VECTOR: question_vectors} if with_questions and question_vectors else {})},
                        payload={"text": text, "metadata": {}})
            for i, (text, text_vector, question_vectors) in enumerate(chunks)])
        return client

    query = [1.0, 0.0, 0.0]
    assert QdrantVectorStore(client=collection(False)).search("g", query, limit=1)[0].text == "A"
    assert QdrantVectorStore(client=collection(True)).search("g", query, limit=1)[0].text == "B"


def test_question_vectors_are_added_to_an_existing_collection(monkeypatch):
    """A guides collection indexed before docs/adr/0011 gets the field on its
    next sync: only the questions are embedded, the text vectors stay."""
    import qdrant_client

    from app import embeddings
    from preprocessing.indexing import questions, vectorizer

    client = QdrantClient(":memory:")
    embedded = []

    class FakeEmbedder:
        def embed_documents(self, texts):
            embedded.extend(texts)
            return [[0.0, 1.0, 0.0] if "\n" not in text else [1.0, 0.0, 0.0] for text in texts]

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: client)
    monkeypatch.setattr(embeddings, "default_embedder", FakeEmbedder)
    dates = {"text": "Başvuru Tarihleri\n17 Mart - 2 Nisan arası online", "metadata": {"topic": "erasmus"},
             "questions": ["erasmus başvurusu nereden yapılıyor"]}
    rules = {"text": "Asgari Şartlar\nLisans için en az 2.20 ortalama", "metadata": {"topic": "erasmus"}}
    vectorizer.store_embedding([dates, rules], "g", vectorizer._sks_payload)
    client.delete_vector_name("g", questions.VECTOR)  # the collection as it was before
    assert not questions.has_field(client, "g")

    embedded.clear()
    vectorizer.store_embedding([dates, rules], "g", vectorizer._sks_payload)

    assert embedded == ["erasmus başvurusu nereden yapılıyor"]
    assert questions.has_field(client, "g")
    # a point with only the text vector comes back as a plain list
    points = {p.payload["text"]: p.vector if isinstance(p.vector, dict) else {"": p.vector}
              for p in client.scroll("g", with_vectors=True)[0]}
    assert points[dates["text"]] == {"": [1.0, 0.0, 0.0], questions.VECTOR: [[0.0, 1.0, 0.0]]}
    assert points[rules["text"]] == {"": [1.0, 0.0, 0.0]}  # no questions, no field


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


def test_best_score_is_the_closest_chunk_or_question():
    """docs/adr/0013: routing compares collections by their closest chunk's
    cosine, counting question vectors where the chunks carry them."""
    from preprocessing.indexing import questions

    client = QdrantClient(":memory:")
    client.create_collection("g", vectors_config={
        "": VectorParams(size=3, distance=Distance.COSINE), questions.VECTOR: questions.params(3)})
    client.upsert("g", points=[
        PointStruct(id=0, vector={"": [0.6, 0.8, 0.0], questions.VECTOR: [[1.0, 0.0, 0.0]]},
                    payload={"text": "A", "metadata": {}}),
        PointStruct(id=1, vector={"": [0.0, 1.0, 0.0]}, payload={"text": "B", "metadata": {}})])
    store = QdrantVectorStore(client=client)

    assert abs(store.best_score("g", [[1.0, 0.0, 0.0]]) - 1.0) < 1e-6  # A's question
    assert abs(store.best_score("g", [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) - 1.0) < 1e-6  # B, 2nd phrasing
    assert store.best_score("missing", [[1.0, 0.0, 0.0]]) is None


def test_a_changed_chunk_never_shows_old_and_new_side_by_side(monkeypatch):
    """The stale delete went out after the upsert, so a live search in
    between got both versions of a changed article (#6). Every write is
    checked from outside, as a search between two requests would see it."""
    import qdrant_client

    from app import embeddings
    from preprocessing.indexing import vectorizer

    client = QdrantClient(":memory:")

    class FakeEmbedder:
        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: client)
    monkeypatch.setattr(embeddings, "default_embedder", FakeEmbedder)
    old = {"text": "MADDE 5 — eski metin", "metadata": {}}
    new = {"text": "MADDE 5 — yeni metin", "metadata": {}}
    other = {"text": "MADDE 6", "metadata": {}}
    vectorizer.store_embedding([old, other], "c", vectorizer._sks_payload)

    seen = []
    for method in ("upsert", "delete", "batch_update_points"):
        real = getattr(client, method)
        monkeypatch.setattr(client, method, lambda *a, _real=real, **k: (
            _real(*a, **k), seen.append({p.payload["text"] for p in client.scroll("c")[0]}))[0])

    vectorizer.store_embedding([new, other], "c", vectorizer._sks_payload)

    assert seen and seen[-1] == {new["text"], other["text"]}
    assert not any(old["text"] in texts and new["text"] in texts for texts in seen)


def test_recreate_keeps_the_old_collection_until_the_new_vectors_are_ready(monkeypatch):
    """--recreate dropped the collection before embedding, so searches got
    nothing for as long as the embedding calls took (#6)."""
    import qdrant_client

    from app import embeddings
    from preprocessing.indexing import vectorizer

    client = QdrantClient(":memory:")
    dim, during = [3], []

    class FakeEmbedder:
        def embed_documents(self, texts):
            if client.collection_exists("c"):
                during.append(client.count("c").count)
            return [[1.0] + [0.0] * (dim[0] - 1) for _ in texts]

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: client)
    monkeypatch.setattr(embeddings, "default_embedder", FakeEmbedder)
    chunk = {"text": "Final sınavları", "metadata": {}}
    vectorizer.store_embedding([chunk], "c", vectorizer._sks_payload)

    during.clear()
    dim[0] = 4  # a new embedding model: only --recreate can switch
    vectorizer.store_embedding([chunk], "c", vectorizer._sks_payload, recreate=True)

    assert during == [1]  # still searchable while the new vectors were made
    assert vectorizer._dense_size(client, "c") == 4
    assert client.count("c").count == 1


def test_a_collection_built_with_another_model_is_refused(monkeypatch):
    """Point ids are content hashes, so after a model change of the same
    dimension unchanged chunks kept the old model's vectors and the search
    compared them with the new model's queries, silently (#8)."""
    import pytest
    import qdrant_client

    from app import embeddings
    from config import settings
    from preprocessing.indexing import vectorizer

    client = QdrantClient(":memory:")
    embedded = []

    class FakeEmbedder:
        def embed_documents(self, texts):
            embedded.extend(texts)
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: client)
    monkeypatch.setattr(embeddings, "default_embedder", FakeEmbedder)
    monkeypatch.setattr(settings, "OPENROUTER_EMBEDDING_MODEL", "old/model")
    chunk = {"text": "Final sınavları", "metadata": {}}
    vectorizer.store_embedding([chunk], "c", vectorizer._sks_payload)

    monkeypatch.setattr(settings, "OPENROUTER_EMBEDDING_MODEL", "new/model")
    embedded.clear()
    with pytest.raises(SystemExit, match="old/model.*new/model.*--recreate"):
        vectorizer.store_embedding([chunk, {"text": "Bahar kayıtları", "metadata": {}}], "c",
                                   vectorizer._sks_payload)
    assert embedded == []  # refused before any embedding call
    with pytest.raises(RuntimeError, match="old/model.*new/model.*--recreate"):
        QdrantVectorStore(client=client).search("c", [1.0, 0.0, 0.0], limit=1)

    vectorizer.store_embedding([chunk], "c", vectorizer._sks_payload, recreate=True)
    assert embeddings.recorded_model(client, "c") == "new/model"
    assert QdrantVectorStore(client=client).search("c", [1.0, 0.0, 0.0], limit=1)[0].text == chunk["text"]


def test_a_collection_without_a_recorded_model_keeps_working(monkeypatch):
    """Collections indexed before the model was recorded are searched, with
    a warning, and their next sync records the model. The search checks
    each collection once, not on every query."""
    import qdrant_client

    from app import embeddings
    from config import settings
    from preprocessing.indexing import vectorizer

    client = make_client(with_bm25=True)  # no recorded model
    checked = []
    real = embeddings.recorded_model
    monkeypatch.setattr(embeddings, "recorded_model",
                        lambda client, collection: checked.append(collection) or real(client, collection))
    store = QdrantVectorStore(client=client)
    for _ in range(3):
        assert store.search("c", QUERY, limit=1)[0].text == "Genel hükümler"
    store.best_score("c", [QUERY])
    assert checked == ["c"]

    class FakeEmbedder:
        def embed_documents(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: client)
    monkeypatch.setattr(embeddings, "default_embedder", FakeEmbedder)
    vectorizer.store_embedding([{"text": "Genel hükümler", "metadata": {}}], "c", vectorizer._sks_payload)
    assert real(client, "c") == settings.OPENROUTER_EMBEDDING_MODEL


def test_store_sends_the_configured_api_key(monkeypatch):
    """docker-compose.yml's Qdrant refuses requests without the key."""
    import qdrant_client

    from config import settings

    client = make_client(with_bm25=True)
    seen = []
    monkeypatch.setattr(settings, "QDRANT_API_KEY", "k")
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda *a, **k: seen.append(k) or client)
    QdrantVectorStore().search("c", QUERY, limit=1)
    assert seen == [{"api_key": "k"}]
