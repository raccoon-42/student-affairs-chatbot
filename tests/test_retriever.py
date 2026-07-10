"""Retriever tested through its real interface, offline: the in-memory
adapter stands in for Qdrant, a fake embedder stands in for the model."""
from app.retrieval import Retriever, InMemoryVectorStore
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

    def search(self, collection, vector, limit, filters=None):
        self.last_filters = filters
        return self.store.search(collection, vector, limit, filters)


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
                            "people", "courses", "guides"}


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


def test_hybrid_score_blends_semantic_and_bm25():
    store = InMemoryVectorStore()
    # Same vector: semantic scores tie, so BM25 on the text must break the tie
    store.add(settings.REGULATIONS_COLLECTION, "kayıt dondurma", {}, [1.0, 0.0, 0.0, 0.1])
    store.add(settings.REGULATIONS_COLLECTION, "başka bir madde", {}, [1.0, 0.0, 0.0, 0.1])
    retriever = Retriever(store, FakeEmbedder())
    results = retriever.retrieve_regulations("kayıt dondurma")
    assert "kayıt dondurma" in results[0]["text"]


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
