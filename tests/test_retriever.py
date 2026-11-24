"""Retriever tested through its real interface, offline: the in-memory
adapter stands in for Qdrant, a fake embedder stands in for the model."""
from datetime import date

from app.retrieval import Retriever, InMemoryVectorStore, normalize_query
from config import settings


class FakeEmbedder:
    """Maps any text to a tiny vector based on keyword presence."""

    def embed_query(self, text):
        text = text.lower()
        return [
            1.0 if "sınav" in text else 0.0,
            1.0 if "kayıt" in text else 0.0,
            1.0 if "tatil" in text else 0.0,
            0.1,  # so no vector is all-zero
        ]


def make_retriever():
    store = InMemoryVectorStore()
    store.add(settings.CALENDAR_COLLECTION, "Final sınavları başlıyor",
              {"event_type": "exam", "academic_period": "fall", "date1": "05.Oca.25", "event": "Final sınavları"},
              [1.0, 0.0, 0.0, 0.1])
    store.add(settings.CALENDAR_COLLECTION, "Bahar kayıtları başlıyor",
              {"event_type": "registration", "academic_period": "spring"},
              [0.0, 1.0, 0.0, 0.1])
    store.add(settings.CALENDAR_COLLECTION, "Yılbaşı tatili",
              {"event_type": "holiday", "academic_period": None},
              [0.0, 0.0, 1.0, 0.1])
    store.add(settings.REGULATIONS_COLLECTION, "Sınav tekrarı kuralları",
              {"section": "2", "chapter": "1"},
              [1.0, 0.0, 0.0, 0.1])
    store.add(settings.REGULATIONS_COLLECTION, "Kayıt dondurma şartları",
              {"section": "3", "chapter": "2"},
              [0.0, 1.0, 0.0, 0.1])
    return Retriever(InMemoryVectorStoreWrapper(store), FakeEmbedder())


class InMemoryVectorStoreWrapper:
    """Records the filters the Retriever sends across the seam."""

    def __init__(self, store):
        self.store = store
        self.last_filters = "unset"

    def search(self, collection, vector, limit, filters=None, sparse=None, rrf_k=None):
        self.last_filters = filters
        return self.store.search(collection, vector, limit, filters, sparse, rrf_k)


def test_normalize_query_expands_student_abbreviations():
    # regression: "yl dersleri ne bilgiayarda" got "YL ile neyi kastediyorsunuz?"
    assert normalize_query("yl dersleri ne") == "yüksek lisans dersleri ne"
    assert normalize_query("YL'de kaç ders") == "yüksek lisans'de kaç ders"
    assert normalize_query("phd courses ceng322") == "doktora courses ceng 322"
    assert normalize_query("yıldız hoca, dr ali") == "yıldız hoca, dr ali"  # "dr" is a title too


def test_query_filter_extraction_reaches_the_store():
    retriever = make_retriever()
    retriever.retrieve_calendar("güz dönemi sınavları ne zaman")
    assert retriever._store.last_filters == {"academic_period": "fall"}


def test_no_signals_means_no_filter():
    retriever = make_retriever()
    retriever.retrieve_calendar("okul hakkında bilgi")
    assert retriever._store.last_filters is None


def test_most_relevant_calendar_chunk_ranks_first():
    retriever = make_retriever()
    results = retriever.retrieve_calendar("sınav ne zaman")
    assert results, "expected results"
    assert "sınav" in results[0]["text"].lower()


def test_calendar_returns_stored_text_verbatim():
    retriever = make_retriever()
    results = retriever.retrieve_calendar("sınav ne zaman")
    assert results[0]["text"] == "Final sınavları başlıyor"


def test_regulations_return_stored_text_verbatim():
    retriever = make_retriever()
    results = retriever.retrieve_regulations("sınav tekrarı")
    assert results[0]["text"] == "Sınav tekrarı kuralları"


def test_empty_collection_returns_empty_list():
    retriever = Retriever(InMemoryVectorStore(), FakeEmbedder())
    assert retriever.retrieve_calendar("sınav") == []


def test_retrieve_all_embeds_the_query_exactly_once():
    class CountingEmbedder(FakeEmbedder):
        def __init__(self):
            self.calls = 0

        def embed_query(self, text):
            self.calls += 1
            return super().embed_query(text)

    embedder = CountingEmbedder()
    retriever = Retriever(InMemoryVectorStore(), embedder)

    results = retriever.retrieve_all("sınav ne zaman")

    assert embedder.calls == 1  # one embed shared by all nine searches
    assert set(results) == {"calendar", "regulations", "faq", "forms", "sks", "programs",
                            "people", "courses", "guides", "course_departments"}


def test_retrieve_all_names_the_departments_with_a_course_catalog():
    # regression: with no coverage stated, the model listed İnşaat/Matematik
    # (seen in PROGRAMS) as "departments whose course catalog I have"
    store = InMemoryVectorStore()
    for department, level in (("Bilgisayar Mühendisliği", "lisans"),
                              ("Bilgisayar Mühendisliği", "doktora"), ("Fizik", "lisans")):
        store.add(settings.COURSES_COLLECTION, f"{department} {level} derslerinin tam listesi",
                  {"kind": "list", "department": department, "levels": [level]}, [0.0, 1.0, 0.0, 0.1])
    store.add(settings.COURSES_COLLECTION, "CENG 311 — Computer Architecture",
              {"kind": "course", "department": "Bilgisayar Mühendisliği"}, [1.0, 0.0, 0.0, 0.1])
    results = Retriever(store, FakeEmbedder()).retrieve_all("doktora dersleri ne")
    assert results["course_departments"] == ["Bilgisayar Mühendisliği", "Fizik"]


def test_faq_returns_question_and_answer():
    store = InMemoryVectorStore()
    store.add(settings.FAQ_COLLECTION, "Kimlik kartım kaybolursa ne yapmalıyım?",
              {"question": "Kimlik kartım kaybolursa ne yapmalıyım?", "answer": "Zayi dilekçesi verin.",
               "audience": "lisans", "category": "ÖĞRENCİ KİMLİK KARTI"},
              [0.0, 1.0, 0.0, 0.1])
    retriever = Retriever(store, FakeEmbedder())

    results = retriever.retrieve_faq("kayıt kartı")

    assert results[0]["text"] == "Soru: Kimlik kartım kaybolursa ne yapmalıyım?\nCevap: Zayi dilekçesi verin."
    assert results[0]["metadata"]["audience"] == "lisans"


def test_forms_return_stored_text_with_link_metadata():
    store = InMemoryVectorStore()
    store.add(settings.FORMS_COLLECTION,
              "Öğrenci Kimlik Kartı Formu\nKayıp kimlik yerine yenisini almak için doldurulur.\n"
              "Anahtar kelimeler: kimlik kaybettim, kimlik zayi",
              {"document_title": "Öğrenci Kimlik Kartı Formu",
               "source_url": "https://example.test/kimlik-formu.pdf",
               "form_code": "İYTE-ÖİDB-0010", "category": "form"},
              [0.0, 1.0, 0.0, 0.1])
    retriever = Retriever(store, FakeEmbedder())

    results = retriever.retrieve_forms("kayıt kartı")

    assert "Öğrenci Kimlik Kartı Formu" in results[0]["text"]
    assert results[0]["metadata"]["source_url"] == "https://example.test/kimlik-formu.pdf"


def test_sks_returns_stored_text_with_page_metadata():
    store = InMemoryVectorStore()
    store.add(settings.SKS_COLLECTION,
              "YEMEKHANE HİZMETİ\nYemekhane Kart Sistemi\nKimlik kartlarına para yükleme "
              "cihazları Merkezi Kafeterya binasındadır.",
              {"document_title": "YEMEKHANE HİZMETİ", "topic": "yemekhane",
               "source_url": "https://sks.iyte.edu.tr/beslenme-hizmetleri/yemekhane-hizmeti/",
               "section": "Yemekhane Kart Sistemi"},
              [0.0, 1.0, 0.0, 0.1])
    retriever = Retriever(store, FakeEmbedder())

    results = retriever.retrieve_sks("kayıt kartı")

    assert "Yemekhane Kart Sistemi" in results[0]["text"]
    assert results[0]["metadata"]["topic"] == "yemekhane"


def make_people_retriever():
    store = InMemoryVectorStore()
    # FakeEmbedder axes (sınav/kayıt/tatil) let each query pick its
    # winner: a person chunk, the roster, and one area enumeration
    store.add(settings.PEOPLE_COLLECTION, "Ali Hoca — sınav sorumlusu",
              {"kind": "person", "document_title": "Ali Hoca"},
              [1.0, 0.0, 0.0, 0.1])
    store.add(settings.PEOPLE_COLLECTION, "Kadro tam listesi: kayıt dönemi görevlileri",
              {"kind": "roster", "document_title": "Kadro (tam liste)"},
              [0.0, 1.0, 0.0, 0.1])
    store.add(settings.PEOPLE_COLLECTION, "tatil planlaması alanında çalışan öğretim üyeleri",
              {"kind": "area", "document_title": "tatil planlaması"},
              [0.0, 0.0, 1.0, 0.1])
    return Retriever(store, FakeEmbedder())


def test_people_list_chunk_admitted_only_when_it_wins():
    retriever = make_people_retriever()
    # collective-style query: the roster outscores -> leads the results
    results = retriever.retrieve_people("kayıt görevlileri kimler")
    assert results[0]["metadata"]["kind"] == "roster"
    # person query: the individual wins -> no list chunk crowds the top-k
    results = retriever.retrieve_people("sınav sorumlusu hoca")
    assert all(r["metadata"]["kind"] == "person" for r in results)


def test_people_roster_rides_along_when_an_area_chunk_wins():
    retriever = make_people_retriever()
    # area-flavored collective query: the area chunk wins the gate, and
    # the roster joins it — grouping/counting needs the complete table
    results = retriever.retrieve_people("tatil planlaması çalışan hocalar")
    kinds = [r["metadata"]["kind"] for r in results]
    assert kinds[0] == "area" and kinds[1] == "roster"


TODAY = date(2026, 11, 24)  # 2026-2027 güz yarıyılı
PAST_FALL = ("[2025-2026 Güz Yarıyılı] 12 Ocak 2026 ile 23 Ocak 2026 arasında yarıyıl sonu sınavları",
             "fall", "2026-01-12", "2026-01-23", [1.0, 0.0, 0.0, 0.1])
PAST_SPRING = ("[2025-2026 Bahar Yarıyılı] 17 Haziran 2026 ile 30 Haziran 2026 arasında yarıyıl sonu sınavları",
               "spring", "2026-06-17", "2026-06-30", [1.0, 0.0, 0.0, 0.1])
UPCOMING_FALL = ("[2026-2027 Güz Yarıyılı] 11 Ocak 2027 ile 22 Ocak 2027 arasında yarıyıl sonu sınavları",
                 "fall", "2027-01-11", "2027-01-22", [0.9, 0.0, 0.4, 0.1])
JUST_ENDED = ("[2026-2027 Güz Yarıyılı] 2 Kasım 2026 ile 20 Kasım 2026 arasında doktora yeterlik sınavları",
              "fall", "2026-11-02", "2026-11-20", [0.9, 0.0, 0.4, 0.1])


def make_calendar_retriever(*events):
    store = InMemoryVectorStore()
    for text, period, start, end, vector in events:
        store.add(settings.CALENDAR_COLLECTION, text,
                  {"academic_period": period, "date1": start, "date2": end}, vector)
    return Retriever(store, FakeEmbedder(), today=lambda: TODAY)


def test_calendar_drops_ended_events_that_crowd_out_upcoming_ones():
    # regression: "sinavlar ne zaman" in Nov 2026 answered with spring dates
    # because last year's exam events, ranked higher, filled every slot
    retriever = make_calendar_retriever(PAST_FALL, PAST_SPRING, UPCOMING_FALL)
    texts = [r["text"] for r in retriever.retrieve_calendar("sınavlar ne zaman", top_k=2)]
    assert any("11 Ocak 2027" in text for text in texts)
    assert not any("2026 arasında yarıyıl sonu" in text for text in texts)


def test_calendar_keeps_events_inside_the_grace_window():
    retriever = make_calendar_retriever(PAST_FALL, JUST_ENDED, UPCOMING_FALL)
    texts = [r["text"] for r in retriever.retrieve_calendar("sınavlar ne zaman")]
    assert any("20 Kasım 2026" in text for text in texts)
    assert not any("23 Ocak 2026" in text for text in texts)


def test_calendar_keeps_past_events_when_the_query_names_a_year():
    retriever = make_calendar_retriever(PAST_FALL, UPCOMING_FALL)
    texts = [r["text"] for r in retriever.retrieve_calendar("2025-2026 güz sınavları ne zaman")]
    assert any("23 Ocak 2026" in text for text in texts)


def test_calendar_falls_back_to_past_events_when_nothing_is_upcoming():
    retriever = make_calendar_retriever(PAST_FALL, PAST_SPRING)
    assert retriever.retrieve_calendar("sınavlar ne zaman")


def test_courses_list_matches_the_level_in_the_query():
    # regression: the first-ranked list won whatever its level, so a
    # yüksek lisans question could get the PhD list
    store = InMemoryVectorStore()
    store.add(settings.COURSES_COLLECTION, "Doktora derslerinin tam listesi: kayıt",
              {"kind": "list", "levels": ["doktora"]}, [0.0, 1.0, 0.0, 0.1])
    store.add(settings.COURSES_COLLECTION, "Graduate course list",
              {"kind": "list", "levels": ["yukseklisans"]}, [0.0, 0.5, 0.5, 0.1])
    store.add(settings.COURSES_COLLECTION, "CENG 311 sınav",
              {"kind": "course", "levels": ["lisans"]}, [1.0, 0.0, 0.0, 0.1])
    retriever = Retriever(store, FakeEmbedder())
    top = retriever.retrieve_courses("yüksek lisans dersleri kayıt listesi")[0]
    assert top["metadata"] == {"kind": "list", "levels": ["yukseklisans"]}
    # no level named: the ranking alone picks
    top = retriever.retrieve_courses("dersleri kayıt listesi")[0]
    assert top["metadata"]["levels"] == ["doktora"]


def test_bm25_breaks_a_dense_tie():
    store = InMemoryVectorStore()
    # Same vector: semantic scores tie, so BM25 on the text must break the tie
    store.add(settings.REGULATIONS_COLLECTION, "başka bir madde", {}, [1.0, 0.0, 0.0, 0.1])
    store.add(settings.REGULATIONS_COLLECTION, "kayıt dondurma", {}, [1.0, 0.0, 0.0, 0.1])
    retriever = Retriever(store, FakeEmbedder())
    results = retriever.retrieve_regulations("kayıt dondurma")
    assert "kayıt dondurma" in results[0]["text"]


def test_bm25_finds_a_chunk_the_dense_search_ranks_low():
    store = InMemoryVectorStore()
    # dense prefers the two generic articles; only BM25 sees "322"
    store.add(settings.REGULATIONS_COLLECTION, "Genel hükümler", {}, [1.0, 0.0, 0.0, 0.1])
    store.add(settings.REGULATIONS_COLLECTION, "Amaç ve kapsam", {}, [1.0, 0.0, 0.0, 0.1])
    store.add(settings.REGULATIONS_COLLECTION, "Madde 322: mazeret", {}, [0.0, 0.0, 1.0, 0.1])
    retriever = Retriever(store, FakeEmbedder())
    # the old pipeline let BM25 re-rank only dense's top 2: this never surfaced
    results = retriever.retrieve_regulations("sınav maddesi 322", top_k=1)
    assert results[0]["text"] == "Madde 322: mazeret"


def test_courses_list_chunk_admitted_only_when_it_wins():
    store = InMemoryVectorStore()
    store.add(settings.COURSES_COLLECTION, "CENG 311 sınav ağırlıklı ders",
              {"kind": "course"}, [1.0, 0.0, 0.0, 0.1])
    store.add(settings.COURSES_COLLECTION, "Lisans dersleri tam liste: kayıt",
              {"kind": "list"}, [0.0, 1.0, 0.0, 0.1])
    retriever = Retriever(store, FakeEmbedder())
    assert retriever.retrieve_courses("kayıt dersleri liste")[0]["metadata"]["kind"] == "list"
    results = retriever.retrieve_courses("CENG 311 sınav")
    assert [r["metadata"]["kind"] for r in results] == ["course"]


def make_faq_retriever():
    store = InMemoryVectorStore()
    # the same question exists in both audience corpora, like the 34
    # verbatim duplicates across the lisans / lisansüstü FAQ PDFs
    for audience in ("lisans", "lisansustu"):
        store.add(settings.FAQ_COLLECTION, "Sınav sonucuna itiraz edebilir miyim?",
                  {"audience": audience, "answer": f"{audience} cevabı"},
                  [1.0, 0.0, 0.0, 0.1])
    store.add(settings.FAQ_COLLECTION, "Kayıt dondurma nasıl yapılır?",
              {"audience": "lisans", "answer": "dilekçe ile"},
              [0.0, 1.0, 0.0, 0.1])
    return Retriever(InMemoryVectorStoreWrapper(store), FakeEmbedder())


def test_faq_audience_filter_selects_the_matching_copy():
    retriever = make_faq_retriever()
    results = retriever.retrieve_faq("sınav sonucuna itiraz", audience="lisansustu")
    assert retriever._store.last_filters == {"audience": "lisansustu"}
    matches = [r for r in results if "itiraz" in r["text"]]
    assert len(matches) == 1
    assert matches[0]["metadata"]["audience"] == "lisansustu"


def test_faq_without_profile_dedupes_duplicate_questions():
    retriever = make_faq_retriever()
    results = retriever.retrieve_faq("sınav sonucuna itiraz")
    assert retriever._store.last_filters is None
    matches = [r for r in results if "itiraz" in r["text"]]
    assert len(matches) == 1
