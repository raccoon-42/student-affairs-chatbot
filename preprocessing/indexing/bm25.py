"""BM25 as sparse vectors, so Qdrant can search it over a whole collection.

Indexing stores each chunk's per-word BM25 weight (term frequency with
saturation and length normalization); Qdrant multiplies in IDF at search
time (Modifier.IDF), counted over the whole collection. The query sends
weight 1 per word, so the dot product is the BM25 score.

tokenize() is shared by both sides: a word only matches if indexing and
search turn it into the same id. After changing it, rebuild the stored
vectors: python -m preprocessing.indexing.reindex_bm25
"""
import re
import unicodedata
import zlib
from collections import Counter
from typing import Dict, Hashable, List

VECTOR_NAME = "bm25"  # the sparse vector next to each point's unnamed dense one
K1 = 1.5  # term frequency saturation
B = 0.75  # length normalization
# Turkish is agglutinative ("sınav", "sınavı", "sınavlar"); cutting words to
# their first 5 letters merges most inflections without a stemmer
PREFIX_LENGTH = 5

_NON_WORD = re.compile(r"[\W_]+")


def tokenize(text: str) -> List[str]:
    # fold to ASCII: students often type "sinav" for "sınav". NFKD splits
    # ö/ü/ş/ç/ğ (and "İ".lower()'s i + dot) into letter + mark; ı has no
    # decomposition, so it's mapped by hand
    text = unicodedata.normalize("NFKD", text.lower().replace("ı", "i"))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # splitting on apostrophes too: "İYTE'nin" -> "iyte", "nin"
    return [word if any(ch.isdigit() for ch in word) else word[:PREFIX_LENGTH]
            for word in _NON_WORD.split(text) if word]


def token_id(token: str) -> int:
    # crc32, not hash(): hash() is salted per process, and the ids the
    # indexer writes must match the ones the API searches with
    return zlib.crc32(token.encode("utf-8"))


def sparse_document(tokens: List[str], avg_length: float) -> Dict[int, float]:
    """BM25 weight per word of one chunk, without IDF (Qdrant adds it)."""
    length_norm = 1 - B + B * len(tokens) / avg_length if avg_length else 1.0
    return {token_id(token): tf * (K1 + 1) / (tf + K1 * length_norm)
            for token, tf in Counter(tokens).items()}


def document_vectors(texts: Dict[Hashable, str]) -> Dict[Hashable, Dict[int, float]]:
    """sparse_document for a whole collection, keyed like the input; the
    length normalization is against these chunks' average length."""
    tokens = {key: tokenize(text) for key, text in texts.items()}
    avg_length = sum(map(len, tokens.values())) / len(tokens) if tokens else 0.0
    return {key: sparse_document(words, avg_length) for key, words in tokens.items()}


def sparse_query(text: str) -> Dict[int, float]:
    return {token_id(token): 1.0 for token in tokenize(text)}
