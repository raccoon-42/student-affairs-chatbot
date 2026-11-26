"""One vector per generated student question (guides, docs/adr/0011).

A chunk's questions are blended into its text vector (docs/adr/0006), where
a long text dilutes them. They also get their own vectors, in a separate
multivector field; a chunk matches by its best question, and the search
fuses that as a third ranking beside dense and BM25. Shared by the
vectorizer, which writes the field, and the search, which queries it.
"""
VECTOR = "questions"


def params(size, for_new_field=False):
    """The field's settings: cosine, one vector per question, best match
    counts. create_collection takes VectorParams; adding the field to an
    existing collection takes DenseVectorConfig."""
    from qdrant_client.http import models as m
    kind = m.DenseVectorConfig if for_new_field else m.VectorParams
    return kind(size=size, distance=m.Distance.COSINE,
                multivector_config=m.MultiVectorConfig(comparator=m.MultiVectorComparator.MAX_SIM))


def has_field(client, collection) -> bool:
    vectors = client.get_collection(collection).config.params.vectors
    return isinstance(vectors, dict) and VECTOR in vectors
