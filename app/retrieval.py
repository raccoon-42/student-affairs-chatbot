"""The Retriever: the one place retrieval happens.

Interface: retrieve_calendar(query) / retrieve_regulations(query) -> chunks.
Everything else — embedding, vector search, metadata filters, hybrid
scoring, result formatting — is implementation, hidden behind it.

The vector store sits behind a seam with two adapters: QdrantVectorStore
(production) and InMemoryVectorStore (tests). Both answer
search(collection, vector, limit, filters, lexical) -> [Hit]: dense and BM25
each rank the collection, and RRF fuses the two rankings. BM25 is Qdrant's
own, on a pre-tokenized text field (see preprocessing/indexing/bm25.py).
"""
import math
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np

from config import settings
from preprocessing.extraction import extract_academic_period
from preprocessing.indexing.bm25 import B, FIELD, K1, bm25_ready, bm25_text, tokenize


# students glue course codes together ("ceng322"); the catalog says
# "CENG 322", so neither the embedding nor BM25 can match the glued form
_COURSE_CODE = re.compile(r'\b([a-zA-Z]{2,4})(\d{3})\b')


# student shorthand the corpus never uses; "dr" stays — it's also a title
_ABBREVIATIONS = {"yl": "yüksek lisans", "phd": "doktora"}
_ABBREVIATION = re.compile(r'\b(' + '|'.join(_ABBREVIATIONS) + r')\b', re.IGNORECASE)

# department codes students use; the corpus spells departments out, so
# "ceng'in erasmus anlaşmaları" matched no department (docs/adr/0008). The
# code stays and the name joins it; before a number it is a course code
# ("ceng 322") and stays one
_DEPARTMENTS = {"ceng": "bilgisayar mühendisliği", "mbg": "moleküler biyoloji ve genetik"}
_DEPARTMENT = re.compile(r'\b(' + '|'.join(_DEPARTMENTS) + r')\b(?![\s-]*\d)', re.IGNORECASE)


def _name_department(match):
    code, name = match.group(1), _DEPARTMENTS[match.group(1).lower()]
    # every entry point normalizes the same query, so a second pass must
    # not add the name again
    if match.string[match.end():].lstrip().lower().startswith(name):
        return code
    return f"{code} {name}"


# lowercase "iyte" read to the rewriter like a typo for a better-known
# university: "iyte mimarlık yüksek lisans" became "İTÜ mimarlık ..." 5 of 5
_IYTE = re.compile(r'\b[iıIİ]yte\b', re.IGNORECASE)


def normalize_query(query: str) -> str:
    query = _IYTE.sub("İYTE", query)
    query = _DEPARTMENT.sub(_name_department, query)
    query = _ABBREVIATION.sub(lambda m: _ABBREVIATIONS[m.group(1).lower()], query)
    return _COURSE_CODE.sub(r'\1 \2', query)


def _course_level(query: str) -> Optional[str]:
    """Program level a (normalized) query names, as course metadata spells it."""
    text = query.lower()
    if re.search(r'y[üu]ksek\s*lisans|master', text):
        return "yukseklisans"
    if re.search(r'doktora|phd', text):
        return "doktora"
    if re.search(r'\blisans\b', text):  # not "lisansüstü": that is both of the above
        return "lisans"
    return None


_YEAR = re.compile(r'\b20\d{2}\b')
CALENDAR_POOL = 50  # fetched before ended events are dropped, then cut to top_k
CALENDAR_GRACE_DAYS = 14  # "notlar açıklandı mı" right after an event still finds it


def _event_dates(metadata: Dict):
    """(start, end) of a calendar event, or None when its dates don't parse."""
    start = metadata.get("date1") or metadata.get("parsed_date1")
    end = metadata.get("date2") or metadata.get("parsed_date2") or start
    try:
        return date.fromisoformat(start), date.fromisoformat(end)
    except (TypeError, ValueError):
        return None


def _ended_before(metadata: Dict, cutoff: date) -> bool:
    dates = _event_dates(metadata)
    return dates is not None and dates[1] < cutoff  # undated events stay


def _calendar_status(metadata: Dict, today: date) -> str:
    """Deterministic past/upcoming tag for a calendar event, computed
    against today's date — date comparison never reaches the model."""
    dates = _event_dates(metadata)
    if dates is None:
        return ""
    start_day, end_day = dates
    if end_day < today:
        return "(DURUM: geçti)"
    if start_day > today:
        return "(DURUM: gelecekte, henüz geçmedi)"
    return "(DURUM: şu an devam ediyor)"


@dataclass
class Hit:
    text: str
    metadata: dict
    score: float


# fused results fetched past the limit, so equal scores straddling it can be
# ordered before the cut (tie groups are a few chunks at most)
TIE_MARGIN = 10


class QdrantVectorStore:
    def __init__(self, url=None, client=None):
        self._url = url or settings.QDRANT_URL
        self._client = client  # created on first use, not at import
        self._bm25_ready = set()  # collections known to carry BM25 vectors

    def search(self, collection: str, vector: List[float], limit: int,
               filters: Optional[Dict[str, str]] = None,
               lexical: Optional[str] = None,
               rrf_k: Optional[int] = None, also=()) -> List[Hit]:
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(self._url)

        from qdrant_client.http.models import (Filter, FieldCondition, MatchValue, Prefetch,
                                               Rrf, RrfQuery, TextQuery)
        query_filter = None
        if filters:
            query_filter = Filter(must=[
                FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                for key, value in filters.items()
            ])

        if not self._read(self._client.collection_exists, collection):
            # a source that hasn't been indexed yet contributes nothing
            # instead of taking the whole conversation down
            print(f"Warning: collection '{collection}' does not exist in Qdrant — skipping")
            return []

        # the filter goes on each ranking: the server would push a top-level
        # one down, but qdrant-client's local mode (tests) doesn't
        depth = max(settings.HYBRID_PREFETCH, limit)
        prefetch = []
        # `also`: (vector, lexical) of other phrasings, each two more rankings
        for dense, words in [(vector, lexical), *also]:
            prefetch.append(Prefetch(query=dense, filter=query_filter, limit=depth))
            if words and self._has_bm25(collection):
                prefetch.append(Prefetch(query=TextQuery(text=words), using=FIELD,
                                         filter=query_filter, limit=depth))
        results = self._read(
            self._client.query_points,
            collection_name=collection,
            prefetch=prefetch,
            query=RrfQuery(rrf=Rrf(k=rrf_k or settings.RRF_K)),
            limit=limit + TIE_MARGIN,
        ).points
        # equal fused scores are common (ranks 1 and 3 score like ranks 3
        # and 1) and Qdrant returns them in no fixed order: break ties by
        # point id, a content hash, so a question always gets the same
        # chunks (docs/adr/0007)
        results = sorted(results, key=lambda hit: (-hit.score, str(hit.id)))[:limit]
        return [
            Hit(text=hit.payload["text"], metadata=hit.payload.get("metadata", {}), score=hit.score)
            for hit in results
        ]

    def _has_bm25(self, collection: str) -> bool:
        # only a yes is cached: a collection indexed before the BM25 field is
        # searched dense-only, and re-checked, until it gets the field
        if collection not in self._bm25_ready:
            if not self._read(bm25_ready, self._client, collection):
                print(f"Warning: collection '{collection}' has no BM25 field — dense only. "
                      f"Run: python -m preprocessing.indexing.reindex_bm25")
                return False
            self._bm25_ready.add(collection)
        return True

    @staticmethod
    def _read(request, *args, **kwargs):
        """One Qdrant read, sent again on a transport error. Under parallel
        load qdrant-client loses about 1 in 600 requests ("Bad file
        descriptor", timeouts) while the server itself is fine: 16 of 10,000
        searches in a stress test, on client 1.18 and 1.20 alike. A message
        runs nine searches at once and one failure fails the message, so
        each read gets two more tries; reads are safe to repeat."""
        from qdrant_client.http.exceptions import ResponseHandlingException

        for attempt in range(3):
            try:
                return request(*args, **kwargs)
            except ResponseHandlingException:
                if attempt == 2:
                    raise


class InMemoryVectorStore:
    """Second adapter for the same seam — used by tests, no server needed.
    Mirrors Qdrant: BM25 with IDF over the whole collection, filters before
    ranking, RRF over the rankings."""

    def __init__(self):
        self._collections: Dict[str, list] = {}

    def add(self, collection: str, text: str, metadata: dict, vector: List[float]):
        self._collections.setdefault(collection, []).append((text, metadata, np.array(vector, dtype=float)))

    def search(self, collection, vector, limit, filters=None, lexical=None, rrf_k=None, also=()):
        entries = self._collections.get(collection, [])
        docs = [Counter(tokenize(text)) for text, _, _ in entries]
        avg_length = sum(sum(doc.values()) for doc in docs) / len(docs) if docs else 0.0
        document_freq = Counter(token for doc in docs for token in doc)
        idf = {token: math.log((len(entries) - n + 0.5) / (n + 0.5) + 1)
               for token, n in document_freq.items()}

        def bm25(words, i):
            doc, length = docs[i], sum(docs[i].values())
            norm = 1 - B + B * length / avg_length if avg_length else 1.0
            return sum(idf[t] * doc[t] * (K1 + 1) / (doc[t] + K1 * norm)
                       for t in set((words or "").split()) if t in doc)

        rankings = []
        for query_vector, query_words in [(vector, lexical), *also]:
            query = np.array(query_vector, dtype=float)
            dense, lexical_hits = [], []
            for i, (text, metadata, doc_vector) in enumerate(entries):
                if filters and any(metadata.get(k) != v for k, v in filters.items()):
                    continue
                denom = np.linalg.norm(query) * np.linalg.norm(doc_vector)
                dense.append((float(np.dot(query, doc_vector) / denom) if denom else 0.0, i))
                score = bm25(query_words, i)
                if score > 0:  # BM25 only returns chunks sharing a word
                    lexical_hits.append((score, i))
            rankings += [dense, lexical_hits]

        fused = {}
        for ranking in rankings:
            ranking.sort(key=lambda pair: pair[0], reverse=True)
            for rank, (_, i) in enumerate(ranking[:max(settings.HYBRID_PREFETCH, limit)]):
                fused[i] = fused.get(i, 0.0) + 1 / (rank + (rrf_k or settings.RRF_K))
        ranked = sorted(fused, key=fused.get, reverse=True)[:limit]
        return [Hit(text=entries[i][0], metadata=entries[i][1], score=fused[i]) for i in ranked]


# people and courses mix single chunks with long "list" chunks. Both tiers
# come from ONE search so their fused scores can be compared — RRF scores
# from separate searches can't (each search's #1 scores about the same).
# Deep enough for top_k singles plus every list chunk ranked above them.
TIERED_POOL = 30


class Retriever:
    def __init__(self, store, embedder, today=date.today):
        self._store = store
        self._embedder = embedder
        self._today = today  # a clock, so tests can pin the date

    def retrieve_all(self, query: str, audience: str = None, also=()) -> Dict[str, List[Dict]]:
        """Everything the conversation needs, in one call: the query is
        embedded once and the collections are searched in parallel.

        `also` takes other phrasings of the same question: the student's own
        words when `query` is a rewrite. Each adds its rankings to every
        search, so a rewrite that misreads the question ("mbg" expanded
        wrong) can't hide the chunks the original words find."""
        query = normalize_query(query)
        others = [text for text in dict.fromkeys(normalize_query(t) for t in also if t)
                  if text.casefold() != query.casefold()]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1 + len(others)) as pool:
            vector, *other_vectors = pool.map(self._embed_query, [query, *others])
        also = list(zip(others, other_vectors))
        embed_seconds = time.perf_counter() - start

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=9) as pool:
            calendar = pool.submit(self.retrieve_calendar, query, 10, vector, also=also)
            regulations = pool.submit(self.retrieve_regulations, query, 5, vector, also=also)
            faq = pool.submit(self.retrieve_faq, query, 5, vector, audience, also=also)
            forms = pool.submit(self.retrieve_forms, query, 3, vector, also=also)
            sks = pool.submit(self.retrieve_sks, query, 4, vector, also=also)
            programs = pool.submit(self.retrieve_programs, query, 3, vector, also=also)
            people = pool.submit(self.retrieve_people, query, 3, vector, also=also)
            courses = pool.submit(self.retrieve_courses, query, 4, vector, also=also)
            # 4, not 3: a long page splits into many look-alike chunks
            # (docs/adr/0006)
            guides = pool.submit(self.retrieve_guides, query, 4, vector, also=also)
            course_departments = pool.submit(self.retrieve_course_departments, vector)
            results = {
                "calendar": calendar.result(),
                "regulations": regulations.result(),
                "faq": faq.result(),
                "forms": forms.result(),
                "sks": sks.result(),
                "programs": programs.result(),
                "people": people.result(),
                "courses": courses.result(),
                "guides": guides.result(),
                "course_departments": course_departments.result(),
            }
        # the conversation logs these per turn (CLI + dev mode in the UI);
        # last-write-wins across sessions is fine for diagnostics
        self.last_timings = {"embed": embed_seconds, "search": time.perf_counter() - start}
        return results

    def retrieve_calendar(self, query: str, top_k: int = 10, vector=None, also=()) -> List[Dict]:
        # academic_period is a reliable hard filter; event_type is not — the
        # query's intent ("son tarih" -> deadline) and the event's class
        # ("Kayıt işlemleri" -> registration) often disagree, and a hard
        # filter then excludes the right answer. Hybrid scoring already
        # rewards matching terms, so event_type stays out of the filter.
        filters = {}
        academic_period = extract_academic_period(query)
        if academic_period:
            filters["academic_period"] = academic_period

        hits = self._hybrid_search(query, settings.CALENDAR_COLLECTION, CALENDAR_POOL,
                                   filters or None, vector, rrf_k=settings.CALENDAR_RRF_K,
                                   also=also)
        # two academic years of near-identical events: last year's exams would
        # take the slots of the upcoming ones a student means. A year in the
        # query ("2025-2026") asks for the past on purpose; and when nothing
        # is left, the past is still the best answer there is.
        today = self._today()
        if not _YEAR.search(query):
            cutoff = today - timedelta(days=CALENDAR_GRACE_DAYS)
            current = [(hit, score) for hit, score in hits
                       if not _ended_before(hit.metadata, cutoff)]
            hits = current or hits
        return [
            {"text": self._format_calendar(hit, today), "score": score, "metadata": hit.metadata}
            for hit, score in hits[:top_k]
        ]

    def retrieve_regulations(self, query: str, top_k: int = 5, vector=None, also=()) -> List[Dict]:
        hits = self._hybrid_search(query, settings.REGULATIONS_COLLECTION, top_k, vector=vector,
                                   also=also)
        return [
            {"text": self._format_regulation(hit), "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_faq(self, query: str, top_k: int = 5, vector=None,
                     audience: str = None, also=()) -> List[Dict]:
        # FAQ chunks embed only the question; the answer travels in metadata.
        # 34 questions exist verbatim in both the lisans and lisansüstü FAQ
        # PDFs — the session's audience picks the right copy, and without a
        # profile the text dedupe below keeps the duplicate out of the top_k
        filters = {"audience": audience} if audience else None
        hits = self._hybrid_search(query, settings.FAQ_COLLECTION, top_k,
                                   filters=filters, vector=vector, also=also)
        results, seen = [], set()
        for hit, score in hits:
            if hit.text in seen:
                continue
            seen.add(hit.text)
            results.append(
                {"text": self._format_faq(hit), "score": score, "metadata": hit.metadata})
        return results

    def retrieve_forms(self, query: str, top_k: int = 3, vector=None, also=()) -> List[Dict]:
        # a link catalog, not documents: the chunk is title + use-case
        # description, the payoff is the source_url in metadata
        hits = self._hybrid_search(query, settings.FORMS_COLLECTION, top_k, vector=vector,
                                   also=also)
        return [
            {"text": hit.text, "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_sks(self, query: str, top_k: int = 4, vector=None, also=()) -> List[Dict]:
        # campus life (spor, topluluklar, yemekhane) — chunk text already
        # leads with the page title, so it renders as-is like mevzuat
        hits = self._hybrid_search(query, settings.SKS_COLLECTION, top_k, vector=vector,
                                   also=also)
        return [
            {"text": hit.text, "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_programs(self, query: str, top_k: int = 3, vector=None, also=()) -> List[Dict]:
        # degree-program catalog; per-level "tam liste" chunks ride along
        # so the model can ground "İYTE'de X bölümü yok" answers
        hits = self._hybrid_search(query, settings.PROGRAMS_COLLECTION, top_k, vector=vector,
                                   also=also)
        return [
            {"text": hit.text, "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_people(self, query: str, top_k: int = 3, vector=None, also=()) -> List[Dict]:
        """Instructors and staff, in three tiers: person chunks, the
        department roster, and per-area enumerations. List-type chunks
        are long and term-dense, so they'd gravitate into every people
        top-k; one is admitted only when it outranks the best individual
        — the ranking itself says whether the query is about the
        collective ("hocalar kimler") or one person ("maili ne"). And
        whenever ANY list chunk wins, the roster rides along: it is the
        data-complete table, so grouping/counting/"tüm hocalar" queries
        work even when an area chunk happens to win the scoring."""
        ranked = self._hybrid_search(query, settings.PEOPLE_COLLECTION, TIERED_POOL,
                                     vector=vector, also=also)
        persons = [pair for pair in ranked if pair[0].metadata.get("kind") == "person"][:top_k]
        # role-scoped enumerations ("asistanların alanları") — see the
        # vectorizer's role chunk for why these need their own tier
        lists = [pair for pair in ranked
                 if pair[0].metadata.get("kind") in ("roster", "area", "role")]
        hits = persons
        if lists and (not persons or lists[0][1] >= persons[0][1]):
            # every list tied for first: RRF ties when two chunks swap ranks
            # between dense and BM25, and the server's tie order is arbitrary
            chosen = [pair for pair in lists if math.isclose(pair[1], lists[0][1])]
            if all(hit.metadata.get("kind") != "roster" for hit, _ in chosen):
                chosen += ([pair for pair in lists if pair[0].metadata.get("kind") == "roster"][:1]
                           or self._hybrid_search(query, settings.PEOPLE_COLLECTION, 1,
                                                  filters={"kind": "roster"}, vector=vector,
                                                  also=also))
            hits = chosen + persons[:max(0, top_k - len(chosen))]
        return [
            {"text": hit.text, "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_courses(self, query: str, top_k: int = 4, vector=None, also=()) -> List[Dict]:
        # course catalog, two tiers like people: individual course chunks
        # plus per-level "tam liste" chunks. The list is term-dense (every
        # course code and name), so it's admitted only when it outscores
        # the best individual course — "X dersi var mı / hangi dersler
        # var" wins the list, "CENG 311'in önkoşulu ne" wins the course.
        ranked = self._hybrid_search(query, settings.COURSES_COLLECTION, TIERED_POOL,
                                     vector=vector, also=also)
        singles = [pair for pair in ranked if pair[0].metadata.get("kind") == "course"][:top_k]
        lists = [pair for pair in ranked if pair[0].metadata.get("kind") == "list"]
        hits = singles
        if lists and (not singles or lists[0][1] >= singles[0][1]):
            # the ranking decides THAT a list is wanted; a level named in the
            # query decides WHICH — "yl dersleri" must not get the PhD list
            level = _course_level(query)
            chosen = next((pair for pair in lists
                           if level in pair[0].metadata.get("levels", [])), lists[0])
            hits = [chosen] + singles[:max(0, top_k - 1)]
        return [
            {"text": hit.text, "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def retrieve_course_departments(self, vector) -> List[str]:
        """Departments whose course catalog is indexed, read off the per-level
        list chunks. The prompt states them outright: left to infer coverage,
        the model took PROGRAMS' PhD programs for catalogs it has."""
        lists = self._hybrid_search("ders listesi", settings.COURSES_COLLECTION, 100,
                                    filters={"kind": "list"}, vector=vector)
        return sorted({hit.metadata["department"] for hit, _ in lists
                       if hit.metadata.get("department")})

    def retrieve_guides(self, query: str, top_k: int = 4, vector=None, also=()) -> List[Dict]:
        # ÖİDB process guides (ders seçimi, yatay geçiş, ÇAP/yan dal,
        # başvuru, askerlik) — chunk text leads with the page title
        hits = self._hybrid_search(query, settings.GUIDES_COLLECTION, top_k, vector=vector,
                                   also=also)
        return [
            {"text": hit.text, "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def _embed_query(self, query):
        # any model-specific instruction prefix lives in the embedder
        return self._embedder.embed_query(query)

    def _hybrid_search(self, query, collection, top_k, filters=None, vector=None, rrf_k=None,
                       also=()):
        """Dense and BM25 each rank the whole collection, for the query and
        for each (text, vector) phrasing in `also`; the store fuses all the
        rankings (RRF). Scores only compare within one call."""
        query = normalize_query(query)  # idempotent; covers direct calls too
        if vector is None:
            vector = self._embed_query(query)
        hits = self._store.search(collection, vector, limit=top_k, filters=filters,
                                  lexical=bm25_text(query), rrf_k=rrf_k,
                                  also=[(v, bm25_text(text)) for text, v in also])
        return [(hit, hit.score) for hit in hits]

    @staticmethod
    def _format_calendar(hit: Hit, today: date) -> str:
        # render_line already produced a readable line (term + dates +
        # description) at index time — reformatting from metadata loses it.
        # The DURUM tag is computed here so the model never does date math
        # (Haiku called an upcoming date "geçmiş" when left to compare).
        status = _calendar_status(hit.metadata, today)
        return f"{hit.text} {status}" if status else hit.text

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
