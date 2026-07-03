"""Retriever tested through its real interface, offline: the in-memory
adapter stands in for Qdrant, a fake embedder stands in for the model."""
from app.retrieval import Retriever, InMemoryVectorStore
from config import settings


class FakeEmbedder:
    """Maps any text to a tiny vector based on keyword presence."""

    def embed(self, text):
        text = text.lower()
        return [
            1.0 if "sınav" in text else 0.0,
            1.0 if "kayıt" in text else 0.0,
            1.0 if "tatil" in text else 0.0,
            0.1,  # so no vector is all-zero
        ]


def make_retriever():
    store = InMemoryVectorStore()
    store.add(settings.CALENDAR_COLLECTION, "Final sınavları başlıyor",
              {"event_type": "exam", "academic_period": "fall", "date1": "05.Oca.25", "event": "Final sınavları"},
              [1.0, 0.0, 0.0, 0.1])
    store.add(settings.CALENDAR_COLLECTION, "Bahar kayıtları başlıyor",
              {"event_type": "registration", "academic_period": "spring"},
              [0.0, 1.0, 0.0, 0.1])
    store.add(settings.CALENDAR_COLLECTION, "Yılbaşı tatili",
              {"event_type": "holiday", "academic_period": None},
              [0.0, 0.0, 1.0, 0.1])
    store.add(settings.REGULATIONS_COLLECTION, "Sınav tekrarı kuralları",
              {"section": "2", "chapter": "1"},
              [1.0, 0.0, 0.0, 0.1])
    store.add(settings.REGULATIONS_COLLECTION, "Kayıt dondurma şartları",
              {"section": "3", "chapter": "2"},
              [0.0, 1.0, 0.0, 0.1])
    return Retriever(InMemoryVectorStoreWrapper(store), FakeEmbedder())


class InMemoryVectorStoreWrapper:
    """Records the filters the Retriever sends across the seam."""

    def __init__(self, store):
        self.store = store
        self.last_filters = "unset"

    def search(self, collection, vector, limit, filters=None):
        self.last_filters = filters
        return self.store.search(collection, vector, limit, filters)


def test_query_filter_extraction_reaches_the_store():
    retriever = make_retriever()
    retriever.retrieve_calendar("güz dönemi sınavları ne zaman")
    assert retriever._store.last_filters == {"event_type": "exam", "academic_period": "fall"}


def test_no_signals_means_no_filter():
    retriever = make_retriever()
    retriever.retrieve_calendar("okul hakkında bilgi")
    assert retriever._store.last_filters is None


def test_most_relevant_calendar_chunk_ranks_first():
    retriever = make_retriever()
    results = retriever.retrieve_calendar("sınav ne zaman")
    assert results, "expected results"
    assert "sınav" in results[0]["text"].lower()


def test_calendar_formatting_uses_event_type_icon():
    retriever = make_retriever()
    results = retriever.retrieve_calendar("sınav ne zaman")
    assert results[0]["text"].startswith("📝")


def test_regulations_formatting_includes_chapter_and_section():
    retriever = make_retriever()
    results = retriever.retrieve_regulations("sınav tekrarı")
    assert results[0]["text"].startswith("📖 Chapter 1, Section 2:")


def test_empty_collection_returns_empty_list():
    retriever = Retriever(InMemoryVectorStore(), FakeEmbedder())
    assert retriever.retrieve_calendar("sınav") == []


def test_hybrid_score_blends_semantic_and_bm25():
    store = InMemoryVectorStore()
    # Same vector: semantic scores tie, so BM25 on the text must break the tie
    store.add(settings.REGULATIONS_COLLECTION, "kayıt dondurma", {}, [1.0, 0.0, 0.0, 0.1])
    store.add(settings.REGULATIONS_COLLECTION, "başka bir madde", {}, [1.0, 0.0, 0.0, 0.1])
    retriever = Retriever(store, FakeEmbedder())
    results = retriever.retrieve_regulations("kayıt dondurma")
    assert "kayıt dondurma" in results[0]["text"]
