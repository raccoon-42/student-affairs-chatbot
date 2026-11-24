from preprocessing.indexing.bm25 import bm25_text, tokenize


def test_tokenize_folds_turkish_and_cuts_to_prefix():
    assert tokenize("Sınavları İYTE'de!") == ["sinav", "iyte", "de"]


def test_ascii_typed_query_matches_turkish_text():
    assert tokenize("sinav ogrenci") == tokenize("Sınav Öğrenci")


def test_numbers_are_not_cut():
    assert tokenize("CENG 3221, 2025-2026") == ["ceng", "3221", "2025", "2026"]


def test_the_field_qdrant_scores_is_the_tokens_joined():
    # Qdrant's word tokenizer splits this back into exactly our tokens
    assert bm25_text("Sınavları İYTE'de!") == "sinav iyte de"


def test_text_without_words_gives_an_empty_field():
    assert tokenize("  !! ") == []
    assert bm25_text("?") == ""
