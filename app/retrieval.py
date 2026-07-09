"""The Retriever: the one place retrieval happens.

Interface: retrieve_calendar(query) / retrieve_regulations(query) -> chunks.
Everything else — embedding, vector search, metadata filters, hybrid
scoring, result formatting — is implementation, hidden behind it.

The vector store sits behind a seam with two adapters: QdrantVectorStore
(production) and InMemoryVectorStore (tests). Both answer
search(collection, vector, limit, filters) -> [Hit].
"""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config import settings
from preprocessing.extraction import extract_academic_period
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


class Retriever:
    def __init__(self, store, embedder,
                 semantic_weight=settings.SEMANTIC_WEIGHT,
                 bm25_weight=settings.BM25_WEIGHT):
        self._store = store
        self._embedder = embedder
        self._semantic_weight = semantic_weight
        self._bm25_weight = bm25_weight

    def retrieve_all(self, query: str, audience: str = None) -> Dict[str, List[Dict]]:
        """Everything the conversation needs, in one call:
        the query is embedded once, the four collections are searched
        in parallel."""
        start = time.perf_counter()
        vector = self._embed_query(query)
        embed_seconds = time.perf_counter() - start

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=7) as pool:
            calendar = pool.submit(self.retrieve_calendar, query, 10, vector)
            regulations = pool.submit(self.retrieve_regulations, query, 5, vector)
            faq = pool.submit(self.retrieve_faq, query, 5, vector, audience)
            forms = pool.submit(self.retrieve_forms, query, 3, vector)
            sks = pool.submit(self.retrieve_sks, query, 4, vector)
            programs = pool.submit(self.retrieve_programs, query, 3, vector)
            people = pool.submit(self.retrieve_people, query, 3, vector)
            results = {
                "calendar": calendar.result(),
                "regulations": regulations.result(),
                "faq": faq.result(),
                "forms": forms.result(),
                "sks": sks.result(),
                "programs": programs.result(),
                "people": people.result(),
            }
        # the conversation logs these per turn (CLI + dev mode in the UI);
        # last-write-wins across sessions is fine for diagnostics
        self.last_timings = {"embed": embed_seconds, "search": time.perf_counter() - start}
        return results

    def retrieve_calendar(self, query: str, top_k: int = 10, vector=None) -> List[Dict]:
        # academic_period is a reliable hard filter; event_type is not — the
        # query's intent ("son tarih" -> deadline) and the event's class
        # ("Kayıt işlemleri" -> registration) often disagree, and a hard
        # filter then excludes the right answer. Hybrid scoring already
        # rewards matching terms, so event_type stays out of the filter.
        filters = {}
        academic_period = extract_academic_period(query)
        if academic_period:
            filters["academic_period"] = academic_period

        hits = self._hybrid_search(query, settings.CALENDAR_COLLECTION, top_k, filters or None, vector)
        return [
            {"text": self._format_calendar(hit), "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_regulations(self, query: str, top_k: int = 5, vector=None) -> List[Dict]:
        hits = self._hybrid_search(query, settings.REGULATIONS_COLLECTION, top_k, vector=vector)
        return [
            {"text": self._format_regulation(hit), "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_faq(self, query: str, top_k: int = 5, vector=None,
                     audience: str = None) -> List[Dict]:
        # FAQ chunks embed only the question; the answer travels in metadata.
        # 34 questions exist verbatim in both the lisans and lisansüstü FAQ
        # PDFs — the session's audience picks the right copy, and without a
        # profile the text dedupe below keeps the duplicate out of the top_k
        filters = {"audience": audience} if audience else None
        hits = self._hybrid_search(query, settings.FAQ_COLLECTION, top_k,
                                   filters=filters, vector=vector)
        results, seen = [], set()
        for hit, score in hits:
            if hit.text in seen:
                continue
            seen.add(hit.text)
            results.append(
                {"text": self._format_faq(hit), "score": score, "metadata": hit.metadata})
        return results

    def retrieve_forms(self, query: str, top_k: int = 3, vector=None) -> List[Dict]:
        # a link catalog, not documents: the chunk is title + use-case
        # description, the payoff is the source_url in metadata
        hits = self._hybrid_search(query, settings.FORMS_COLLECTION, top_k, vector=vector)
        return [
            {"text": hit.text, "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_sks(self, query: str, top_k: int = 4, vector=None) -> List[Dict]:
        # campus life (spor, topluluklar, yemekhane) — chunk text already
        # leads with the page title, so it renders as-is like mevzuat
        hits = self._hybrid_search(query, settings.SKS_COLLECTION, top_k, vector=vector)
        return [
            {"text": hit.text, "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_programs(self, query: str, top_k: int = 3, vector=None) -> List[Dict]:
        # degree-program catalog; per-level "tam liste" chunks ride along
        # so the model can ground "İYTE'de X bölümü yok" answers
        hits = self._hybrid_search(query, settings.PROGRAMS_COLLECTION, top_k, vector=vector)
        return [
            {"text": hit.text, "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_people(self, query: str, top_k: int = 3, vector=None) -> List[Dict]:
        """Instructors and staff, in three tiers: person chunks, the
        department roster, and per-area enumerations. List-type chunks
        are long and term-dense, so they'd gravitate into every people
        top-k; one is admitted only when it outscores the best individual
        — the scores themselves say whether the query is about the
        collective ("hocalar kimler") or one person ("maili ne"). And
        whenever ANY list chunk wins, the roster rides along: it is the
        data-complete table, so grouping/counting/"tüm hocalar" queries
        work even when an area chunk happens to win the scoring."""
        persons = self._hybrid_search(query, settings.PEOPLE_COLLECTION, top_k,
                                      filters={"kind": "person"}, vector=vector)
        roster = self._hybrid_search(query, settings.PEOPLE_COLLECTION, 1,
                                     filters={"kind": "roster"}, vector=vector)
        areas = self._hybrid_search(query, settings.PEOPLE_COLLECTION, 1,
                                    filters={"kind": "area"}, vector=vector)
        hits = persons
        lists = sorted(roster + areas, key=lambda pair: pair[1], reverse=True)
        if lists and (not persons or lists[0][1] >= persons[0][1]):
            chosen = [lists[0]]
            if roster and roster[0][0] is not lists[0][0]:
                chosen.append(roster[0])
            hits = chosen + persons[:max(0, top_k - len(chosen))]
        return [
            {"text": hit.text, "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def _embed_query(self, query):
        # any model-specific instruction prefix lives in the embedder
        return self._embedder.embed_query(query)

    def _hybrid_search(self, query, collection, top_k, filters=None, vector=None):
        """Vector search for candidates, then re-rank with a blend of
        cosine similarity and BM25 over the candidate texts."""
        if vector is None:
            vector = self._embed_query(query)
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
        # render_line already produced a readable line (term + dates +
        # description) at index time — reformatting from metadata loses it
        return hit.text

    @staticmethod
    def _format_faq(hit: Hit) -> str:
        answer = hit.metadata.get("answer", "")
        if answer:
            return f"Soru: {hit.text}\nCevap: {answer}"
        return hit.text

    @staticmethod
    def _format_regulation(hit: Hit) -> str:
        # render_article already leads with MADDE n (BÖLÜM ...)
        return hit.text


def default_retriever() -> Retriever:
    """The production wiring: Qdrant + the configured embedding backend."""
    from app.embeddings import default_embedder
    return Retriever(QdrantVectorStore(), default_embedder())
