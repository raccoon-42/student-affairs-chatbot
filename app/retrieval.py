"""The Retriever: the one place retrieval happens.

Interface: retrieve_calendar(query) / retrieve_regulations(query) -> chunks.
Everything else — embedding, vector search, metadata filters, hybrid
scoring, result formatting — is implementation, hidden behind it.

The vector store sits behind a seam with two adapters: QdrantVectorStore
(production) and InMemoryVectorStore (tests). Both answer
search(collection, vector, limit, filters) -> [Hit].
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import settings
from preprocessing.extraction import extract_event_type, extract_academic_period, format_date_range
from preprocessing.indexing.bm25 import BM25, preprocess_text


@dataclass
class Hit:
    text: str
    metadata: dict
    score: float


class QdrantVectorStore:
    def __init__(self, url=None):
        self._url = url or settings.QDRANT_URL
        self._client = None  # created on first use, not at import

    def search(self, collection: str, vector: List[float], limit: int,
               filters: Optional[Dict[str, str]] = None) -> List[Hit]:
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(self._url)

        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        query_filter = None
        if filters:
            query_filter = Filter(must=[
                FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                for key, value in filters.items()
            ])

        if not self._client.collection_exists(collection):
            # a source that hasn't been indexed yet contributes nothing
            # instead of taking the whole conversation down
            print(f"Warning: collection '{collection}' does not exist in Qdrant — skipping")
            return []

        results = self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            query_filter=query_filter,
        ).points
        return [
            Hit(text=hit.payload["text"], metadata=hit.payload.get("metadata", {}), score=hit.score)
            for hit in results
        ]


class InMemoryVectorStore:
    """Second adapter for the same seam — used by tests, no server needed."""

    def __init__(self):
        self._collections: Dict[str, list] = {}

    def add(self, collection: str, text: str, metadata: dict, vector: List[float]):
        self._collections.setdefault(collection, []).append((text, metadata, np.array(vector, dtype=float)))

    def search(self, collection, vector, limit, filters=None):
        query = np.array(vector, dtype=float)
        hits = []
        for text, metadata, doc_vector in self._collections.get(collection, []):
            if filters and any(metadata.get(k) != v for k, v in filters.items()):
                continue
            denom = np.linalg.norm(query) * np.linalg.norm(doc_vector)
            score = float(np.dot(query, doc_vector) / denom) if denom else 0.0
            hits.append(Hit(text=text, metadata=metadata, score=score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]


class SentenceTransformerEmbedder:
    def __init__(self, model_name=None):
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None  # loaded on first use — importing this module stays cheap

    def embed(self, text: str) -> List[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model.encode(text).tolist()


class Retriever:
    def __init__(self, store, embedder,
                 semantic_weight=settings.SEMANTIC_WEIGHT,
                 bm25_weight=settings.BM25_WEIGHT):
        self._store = store
        self._embedder = embedder
        self._semantic_weight = semantic_weight
        self._bm25_weight = bm25_weight

    def retrieve_calendar(self, query: str, top_k: int = 10) -> List[Dict]:
        filters = {}
        event_type = extract_event_type(query)
        if event_type:
            filters["event_type"] = event_type
        academic_period = extract_academic_period(query)
        if academic_period:
            filters["academic_period"] = academic_period

        hits = self._hybrid_search(query, settings.CALENDAR_COLLECTION, top_k, filters or None)
        return [
            {"text": self._format_calendar(hit), "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_regulations(self, query: str, top_k: int = 3) -> List[Dict]:
        hits = self._hybrid_search(query, settings.REGULATIONS_COLLECTION, top_k)
        return [
            {"text": self._format_regulation(hit), "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_faq(self, query: str, top_k: int = 3) -> List[Dict]:
        # FAQ chunks embed only the question; the answer travels in metadata
        hits = self._hybrid_search(query, settings.FAQ_COLLECTION, top_k)
        return [
            {"text": self._format_faq(hit), "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def _hybrid_search(self, query, collection, top_k, filters=None):
        """Vector search for candidates, then re-rank with a blend of
        cosine similarity and BM25 over the candidate texts."""
        vector = self._embedder.embed(f"Instruct: {settings.EMBED_INSTRUCTION}\nQuery: {query}")
        candidates = self._store.search(collection, vector, limit=top_k * 2, filters=filters)
        if not candidates:
            return []

        documents = [preprocess_text(hit.text) for hit in candidates]
        bm25 = BM25()  # fresh per query: no state carried between requests
        bm25.fit(documents)
        bm25_scores = bm25.score(preprocess_text(query), documents)

        scored = [
            (hit, self._semantic_weight * hit.score + self._bm25_weight * bm25_score)
            for hit, bm25_score in zip(candidates, bm25_scores)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _format_calendar(hit: Hit) -> str:
        metadata = hit.metadata
        icons = {"holiday": "🎉", "exam": "📝", "deadline": "⏰"}
        icon = icons.get(metadata.get("event_type"))
        if icon:
            return f"{icon} {hit.text}"
        date_range = format_date_range(metadata.get("date1"), metadata.get("date2"))
        if date_range:
            return f"{date_range}: {metadata.get('event', hit.text)}"
        return hit.text

    @staticmethod
    def _format_faq(hit: Hit) -> str:
        answer = hit.metadata.get("answer", "")
        if answer:
            return f"Soru: {hit.text}\nCevap: {answer}"
        return hit.text

    @staticmethod
    def _format_regulation(hit: Hit) -> str:
        section = hit.metadata.get("section", "")
        chapter = hit.metadata.get("chapter", "")
        if section and chapter:
            return f"📖 Chapter {chapter}, Section {section}: {hit.text}"
        return f"📖 {hit.text}"


def default_retriever() -> Retriever:
    """The production wiring: Qdrant + sentence-transformers."""
    return Retriever(QdrantVectorStore(), SentenceTransformerEmbedder())
