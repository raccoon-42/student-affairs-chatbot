from preprocessing.indexing.bm25 import BM25, preprocess_text


def test_scores_relevant_document_higher():
    docs = ["kayıt tarihleri açıklandı", "sınav takvimi yayınlandı", "yaz tatili başlıyor"]
    bm25 = BM25()
    bm25.fit(docs)
    scores = bm25.score("sınav takvimi", docs)
    assert scores[1] == max(scores)


def test_refit_resets_state():
    """Regression: fit() used to accumulate doc_freqs/idf across calls,
    so scores drifted the longer the process ran."""
    docs_a = ["elma armut", "elma kiraz"]
    docs_b = ["kayıt sınav", "sınav takvim"]

    fresh = BM25()
    fresh.fit(docs_b)
    expected = fresh.score("sınav", docs_b)

    reused = BM25()
    reused.fit(docs_a)
    reused.fit(docs_b)
    assert reused.score("sınav", docs_b) == expected
    assert "elma" not in reused.idf


def test_empty_corpus_does_not_crash():
    bm25 = BM25()
    bm25.fit([])
    assert bm25.score("anything", []) == []


def test_preprocess_text_normalizes():
    assert preprocess_text("  Merhaba,   dünya!  ") == "merhaba dünya"
