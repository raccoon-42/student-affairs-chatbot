"""BM25 through Qdrant's full-text index (BM25 scoring, Qdrant 1.20+).

Qdrant tokenizes, keeps the statistics (IDF, average length) current as
chunks change, and scores BM25 itself. What it can't do is cut words to
their first 5 letters, which merges Turkish inflections ("sınav",
"sınavı", "sınavlar") and measured clearly better than whole words or a
Turkish stemmer (docs/adr/0004). So every chunk carries a second text
field, FIELD, holding our tokenizer's output; Qdrant indexes that field
with a plain word tokenizer, and queries go through the same tokenizer.

tokenize() is shared by both sides: a word only matches if indexing and
search cut it the same way. After changing it, rewrite the field:
python -m preprocessing.indexing.reindex_bm25
"""
import re
import unicodedata
from typing import List

FIELD = "bm25_text"  # payload field with the tokenized text Qdrant scores
# Qdrant's BM25 defaults; the in-memory test store mirrors them
K1 = 1.2  # term frequency saturation
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


def bm25_text(text: str) -> str:
    """The FIELD value for a chunk, and the text a query is searched with."""
    return " ".join(tokenize(text))


def bm25_ready(client, collection):
    """Whether the collection's points carry the indexed BM25 field. The
    server lists its payload indexes; qdrant-client's local mode (tests)
    lists none, so an empty list falls back to checking a point for the
    field. Ours always creates the index before writing the field."""
    schema = client.get_collection(collection).payload_schema or {}
    if FIELD in schema:
        return True
    if schema:
        return False
    points, _ = client.scroll(collection, limit=1, with_payload=[FIELD], with_vectors=False)
    return bool(points) and FIELD in (points[0].payload or {})


def index_params():
    """FIELD's index: the words are already folded and cut, so Qdrant's
    default text processing (word tokenizer, lowercase) and BM25 scoring
    are all it needs. Defaults also keep qdrant-client's local mode (tests)
    able to score it: it only scores default-tokenization indexes."""
    from qdrant_client.http import models

    return models.TextIndexParams(type=models.TextIndexType.TEXT, scoring=True)
