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
from preprocessing.indexing import questions
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
# the term's lines (see _current_term): its teaching periods ("... I.
# yarıyıl dersler", "Yaz Öğretimi dersleri") and its final exams ("...
# yarıyıl sonu sınavları", "Yaz Öğretimi sınavları"; not the grade
# deadlines after them)
TEACHING_PERIOD = re.compile(r"yarıyıl ders|Yaz Öğretimi dersleri")
FINAL_EXAMS = re.compile(r"yarıyıl sonu sınavları\b|Yaz Öğretimi sınavları")
TERM_POOL = 200  # every event of a type: 58 exams, 44 courses
# render_line puts the term first: "[2026-2027 Güz Yarıyılı] 28 Eylül ..."
_TERM_NAME = re.compile(r"^\[([^\]]+)\]")


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


_MONTHS = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos",
           "Eylül", "Ekim", "Kasım", "Aralık"]
_TEXT_DATE = r"(\d{1,2}) (" + "|".join(_MONTHS) + r") (\d{4})"
# "02 Nisan 2026", "17 Mart 2026 - 02 Nisan 2026", with a suffix ("2026’dan")
_DATE_SPAN = re.compile(_TEXT_DATE + r"(?:\s*[-–]\s*" + _TEXT_DATE + r")?(?:['’]\w+)?")
# older dates in prose are references ("15 Temmuz 2016", a 2019 law), not
# deadlines a student could still meet or miss
DATED_REFERENCE = timedelta(days=365)


def _mark_dates(text: str, today: date) -> str:
    """The calendar's DURUM label after each recent date written out in
    prose: "17 Mart 2026 - 02 Nisan 2026 (DURUM: geçti)". Guide pages keep
    their deadlines in sentences, and the model, left to compare them with
    today, presented a call that closed in April as open in November."""
    def label(match):
        day, month, year, end_day, end_month, end_year = match.groups()
        try:
            start = date(int(year), _MONTHS.index(month) + 1, int(day))
            end = date(int(end_year), _MONTHS.index(end_month) + 1, int(end_day)) if end_day else start
        except ValueError:  # "31 Şubat"
            return match.group(0)
        if end < today - DATED_REFERENCE:
            return match.group(0)
        status = _calendar_status({"date1": start.isoformat(), "date2": end.isoformat()}, today)
        return f"{match.group(0)} {status}"
    return _DATE_SPAN.sub(label, text)


@dataclass
class Hit:
    text: str
    metadata: dict
    score: float


# fused results fetched past the limit, so equal scores straddling it can be
# ordered before the cut (tie groups are a few chunks at most)
TIE_MARGIN = 10


class QdrantVectorStore:
    def __init__(self, url=None, client=None, embedding_model=None):
        self._url = url or settings.QDRANT_URL
        self._client = client  # created on first use, not at import
        self._model = embedding_model or settings.OPENROUTER_EMBEDDING_MODEL  # the queries'
        self._bm25_ready = set()  # collections known to carry BM25 vectors
        self._questions = {}  # collection -> (has question vectors, checked at)
        self._model_ok = set()  # collections not embedded with another model

    def search(self, collection: str, vector: List[float], limit: int,
               filters: Optional[Dict[str, str]] = None,
               lexical: Optional[str] = None,
               rrf_k: Optional[int] = None, also=()) -> List[Hit]:
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(self._url)

        from qdrant_client.http.models import (Filter, FieldCondition, MatchAny, MatchValue,
                                               Prefetch, Rrf, RrfQuery, TextQuery)
        query_filter = None
        if filters:
            # a list matches any of its values
            query_filter = Filter(must=[
                FieldCondition(key=f"metadata.{key}",
                               match=MatchAny(any=value) if isinstance(value, list)
                               else MatchValue(value=value))
                for key, value in filters.items()
            ])

        if not self._read(self._client.collection_exists, collection):
            # a source that hasn't been indexed yet contributes nothing
            # instead of taking the whole conversation down
            print(f"Warning: collection '{collection}' does not exist in Qdrant — skipping")
            return []
        self._check_model(collection)

        # the filter goes on each ranking: the server would push a top-level
        # one down, but qdrant-client's local mode (tests) doesn't
        depth = max(settings.HYBRID_PREFETCH, limit)
        prefetch = []
        # `also`: (vector, lexical) of other phrasings, each two more rankings
        # (three where the chunks carry question vectors, docs/adr/0011)
        by_question = self._has_questions(collection)
        for dense, words in [(vector, lexical), *also]:
            prefetch.append(Prefetch(query=dense, filter=query_filter, limit=depth))
            if by_question:
                prefetch.append(Prefetch(query=[dense], using=questions.VECTOR,
                                         filter=query_filter, limit=depth))
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

    def best_score(self, collection: str, vectors: List[List[float]]) -> Optional[float]:
        """Cosine of the chunk closest to any of the vectors, or of its
        closest question where the chunks carry question vectors. Unlike
        search()'s fused scores these compare across collections: all are
        embedded by one model. None when the collection doesn't exist."""
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(self._url)
        from qdrant_client.http.models import QueryRequest

        if not self._read(self._client.collection_exists, collection):
            return None
        self._check_model(collection)
        requests = [QueryRequest(query=vector, limit=1) for vector in vectors]
        if self._has_questions(collection):
            requests += [QueryRequest(query=[vector], using=questions.VECTOR, limit=1)
                         for vector in vectors]
        responses = self._read(self._client.query_batch_points, collection, requests=requests)
        return max((point.score for response in responses for point in response.points),
                   default=None)

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

    def _has_questions(self, collection: str) -> bool:
        # most collections never get the field, so a no is cached too, but
        # only for a minute: the daily update can add it to a running server
        known = self._questions.get(collection)
        if known is None or (not known[0] and time.monotonic() - known[1] > 60):
            known = (self._read(questions.has_field, self._client, collection), time.monotonic())
            self._questions[collection] = known
        return known[0]

    def _check_model(self, collection: str):
        """Vectors from another embedding model live in another space: with
        the same dimension the search returns noise and no error, so it
        fails instead. Only a pass is cached: after a --recreate a mismatch
        clears without a restart. A collection indexed before the model was
        recorded is searched, with a warning."""
        from app.embeddings import recorded_model

        if collection in self._model_ok:
            return
        recorded = self._read(recorded_model, self._client, collection)
        if recorded is None:
            print(f"Warning: collection '{collection}' records no embedding model — assuming "
                  f"{self._model}. Its next vectorizer run records it")
        elif recorded != self._model:
            raise RuntimeError(
                f"collection '{collection}' was embedded with {recorded}, but queries are "
                f"embedded with {self._model} (OPENROUTER_EMBEDDING_MODEL). Rerun its "
                f"vectorizer with --recreate, or set the model back")
        self._model_ok.add(collection)

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
                if filters and any(metadata.get(k) not in (v if isinstance(v, list) else [v])
                                   for k, v in filters.items()):
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

    def best_score(self, collection, vectors):
        scores = []
        for vector in vectors:
            query = np.array(vector, dtype=float)
            for _, _, doc_vector in self._collections.get(collection, []):
                denom = np.linalg.norm(query) * np.linalg.norm(doc_vector)
                scores.append(float(np.dot(query, doc_vector) / denom) if denom else 0.0)
        return max(scores, default=None)


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
        others = list(dict.fromkeys(normalize_query(t) for t in also if t))
        # a rewrite that only changes capitals is the student's question as
        # they wrote it: search their words, once. The embedding isn't
        # case-blind, and the student's casing did better on the eval
        # ("erasmus başvuru formunu ..." found the online-application chunk,
        # "Erasmus başvuru formunu ..." didn't; docs/adr/0011)
        query = next((text for text in others if text.casefold() == query.casefold()), query)
        others = [text for text in others if text.casefold() != query.casefold()]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1 + len(others)) as pool:
            vector, *other_vectors = pool.map(self._embed_query, [query, *others])
        also = list(zip(others, other_vectors))
        embed_seconds = time.perf_counter() - start

        start = time.perf_counter()
        collections = {"calendar": settings.CALENDAR_COLLECTION,
                       "regulations": settings.REGULATIONS_COLLECTION,
                       "faq": settings.FAQ_COLLECTION, "forms": settings.FORMS_COLLECTION,
                       "sks": settings.SKS_COLLECTION, "programs": settings.PROGRAMS_COLLECTION,
                       "people": settings.PEOPLE_COLLECTION,
                       "courses": settings.COURSES_COLLECTION,
                       "guides": settings.GUIDES_COLLECTION}
        # nine searches, the course catalog list, the term and nine closeness checks
        with ThreadPoolExecutor(max_workers=20) as pool:
            closest = {name: pool.submit(self._store.best_score, collection,
                                         [vector, *other_vectors])
                       for name, collection in collections.items()}
            # every collection's chunks go to every answer, so each cut is
            # what the retrieval eval showed still holds every needed chunk
            # (docs/adr/0012); regulations keeps one above that
            calendar = pool.submit(self.retrieve_calendar, query, 10, vector, also=also)
            regulations = pool.submit(self.retrieve_regulations, query, 3, vector, also=also)
            faq = pool.submit(self.retrieve_faq, query, 4, vector, audience, also=also)
            forms = pool.submit(self.retrieve_forms, query, 2, vector, also=also)
            sks = pool.submit(self.retrieve_sks, query, 2, vector, also=also)
            programs = pool.submit(self.retrieve_programs, query, 3, vector, also=also)
            people = pool.submit(self.retrieve_people, query, 3, vector, also=also)
            courses = pool.submit(self.retrieve_courses, query, 3, vector, also=also)
            # 4, not 3: a long page splits into many look-alike chunks
            # (docs/adr/0006)
            guides = pool.submit(self.retrieve_guides, query, 4, vector, also=also)
            course_departments = pool.submit(self.retrieve_course_departments, vector)
            term = pool.submit(self._term, vector, self._today())
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
                "term": term.result(),
            }
            self.last_routing = self._route(results, {name: future.result()
                                                      for name, future in closest.items()})
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
        if vector is None:  # embedded once, for the search and the current term
            vector = self._embed_query(normalize_query(query))
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
        hits = hits[:top_k]
        if not _YEAR.search(query):
            shown = {hit.text for hit, _ in hits}
            hits += [(hit, hit.score) for hit in self._current_term(vector, today)
                     if hit.text not in shown]
        return [
            {"text": self._format_calendar(hit, today), "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    def _current_term(self, vector, today) -> List[Hit]:
        """The term under way, or between terms the next one: its teaching
        periods and its final exams (docs/adr/0015). "dönem ne zaman
        bitiyor?" means that term, but the lecture period's line ("28 Eylül
        ile 6 Ocak tarihleri arasında ... I. yarıyıl dersler") shares no
        words with "bitiyor" and ranked past 40, so the model guessed the end
        date; the rewrite of "okul ne zaman kapanıyor?" kept the lectures and
        dropped the exams. Two or three short lines, every time."""
        return (self._under_way_or_next(vector, today, "course", TEACHING_PERIOD)
                + self._under_way_or_next(vector, today, "exam", FINAL_EXAMS))

    def _term(self, vector, today) -> str:
        """Which term today falls in, for the prompt (docs/adr/0015): the
        model then needs no date math for "bu dönem", and knows it even when
        routing leaves the calendar out. Its dates still come from the
        calendar lines, with markers."""
        hits = self._current_term(vector, today)

        def name(hit):
            match = _TERM_NAME.match(hit.text)
            return match.group(1) if match else None

        under_way = [hit for hit in hits if (dates := _event_dates(hit.metadata))
                     and dates[0] <= today <= dates[1] and name(hit)]
        if under_way:
            return f"{name(under_way[0])} (sürüyor)"
        upcoming = [name(hit) for hit in hits
                    if hit.metadata.get("event_type") == "course" and name(hit)]
        if upcoming:
            return f"dönem arası; sıradaki: {upcoming[0]}"
        return "takvimde sürmekte olan ya da sıradaki yarıyıl yok"

    def _under_way_or_next(self, vector, today, event_type, period) -> List[Hit]:
        hits = self._store.search(settings.CALENDAR_COLLECTION, vector, limit=TERM_POOL,
                                  filters={"event_type": event_type})
        periods = [(hit, dates) for hit in hits
                   if period.search(hit.metadata.get("event", ""))
                   and (dates := _event_dates(hit.metadata))]
        under_way = [hit for hit, (start, end) in periods if start <= today <= end]
        if under_way:
            return under_way
        upcoming = [(hit, start) for hit, (start, _) in periods if start > today]
        first = min((start for _, start in upcoming), default=None)
        return [hit for hit, start in upcoming if start == first]

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
        # profile the text dedupe below keeps the duplicate out of the top_k.
        # The hazırlık (English prep) FAQ has no profile of its own and
        # shares no question with either copy, so every audience gets it too
        filters = {"audience": [audience, "hazirlik"]} if audience else None
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
        today = self._today()
        return [
            {"text": _mark_dates(hit.text, today), "score": score, "metadata": hit.metadata}
            for hit, score in hits
        ]

    @staticmethod
    def _route(results, closest):
        """Empty the collections whose closest chunk is far less close than
        the best collection's: the answer call would pay for chunks that
        don't hold the answer (docs/adr/0013). Returns what it decided, for
        the dev log."""
        scores = {name: score for name, score in closest.items() if score is not None}
        dropped = []
        if settings.ROUTE_GAP is not None and scores:
            floor = max(scores.values()) - settings.ROUTE_GAP
            dropped = [name for name, score in scores.items() if score < floor]
            for name in dropped:
                results[name] = []
        return {"scores": scores, "dropped": dropped}

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
