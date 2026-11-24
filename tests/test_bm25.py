from preprocessing.indexing.bm25 import (document_vectors, sparse_document, sparse_query,
                                         token_id, tokenize)


def test_tokenize_folds_turkish_and_cuts_to_prefix():
    assert tokenize("Sınavları İYTE'de!") == ["sinav", "iyte", "de"]


def test_ascii_typed_query_matches_turkish_text():
    assert tokenize("sinav ogrenci") == tokenize("Sınav Öğrenci")


def test_numbers_are_not_cut():
    assert tokenize("CENG 3221, 2025-2026") == ["ceng", "3221", "2025", "2026"]


def test_token_ids_are_stable_across_processes():
    # a fixed crc32, unlike hash(): the indexer and the API must agree
    assert token_id("harc") == 1601276529


def test_short_focused_chunk_outscores_long_one():
    focused = ["harc"] * 2 + [f"w{i}" for i in range(38)]   # 40 words
    general = ["harc"] * 6 + [f"w{i}" for i in range(394)]  # 400 words
    harc = token_id("harc")
    assert sparse_document(focused, 100)[harc] > sparse_document(general, 100)[harc]


def test_repeated_word_saturates():
    once = sparse_document(["harc", "a", "b", "c", "d", "e"], 6)[token_id("harc")]
    six_times = sparse_document(["harc"] * 6, 6)[token_id("harc")]
    assert once < six_times < 6 * once


def test_document_vectors_keep_the_input_keys():
    vectors = document_vectors({"a": "kayıt dondurma", "b": "sınav"})
    assert set(vectors) == {"a", "b"}
    assert set(vectors["b"]) == {token_id("sinav")}


def test_sparse_query_weights_each_word_once():
    assert sparse_query("harç harç ödeme") == {token_id("harc"): 1.0, token_id("odeme"): 1.0}


def test_text_without_words_gives_empty_vectors():
    assert tokenize("  !! ") == []
    assert sparse_query("?") == {}
