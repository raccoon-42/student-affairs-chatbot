"""The citation check: markers whose chunk doesn't back the text are removed."""
import time

from app.citations import (add_date_markers, add_missing_markers, add_phrase_markers,
                           check_citations, correct_markers)
from app.conversation import Conversation
from tests.test_conversation import FakeLLM, FakeRetriever

CALENDAR = [
    "[2026-2027 Güz Yarıyılı] Tarihi ayrıca duyurulacak: Lisans programlarına yurtdışından kabul kontenjanları",
    "[2026-2027 Güz Yarıyılı] 24 Aralık 2026 Perşembe: Bahar Yarıyılı lisansüstü başvuruları",
    "[2026-2027 Bahar Yarıyılı] 15 Şubat 2027 Pazartesi: Derslerin başlaması",
]
ERASMUS = ("Bilgisayar Mühendisliği Erasmus+ anlaşmaları: Almanya - Ruprecht Karls Universität "
           "Heidelberg, Universität Bonn; Polonya - Bialystok University of Technology")


def test_footnote_style_renumbering_is_removed():
    # message 503: the facts came from chunk [4], the model wrote [1] and
    # closed with a footnote line; chunks [1]-[3] are calendar entries
    answer = ("Anlaşmalı üniversiteler şunlar:\n\n"
              "- **Almanya:** Ruprecht Karls Universität Heidelberg, Universität Bonn [1]\n"
              "- **Polonya:** Bialystok University of Technology [1]\n\n"
              "[1] [2] [3]")

    cleaned, removed = check_citations(answer, CALENDAR + [ERASMUS])

    assert cleaned == ("Anlaşmalı üniversiteler şunlar:\n\n"
                       "- **Almanya:** Ruprecht Karls Universität Heidelberg, Universität Bonn\n"
                       "- **Polonya:** Bialystok University of Technology")
    assert sorted(removed) == [1, 1, 1, 2, 3]


def test_correct_citations_are_kept():
    answer = "- **Almanya:** Ruprecht Karls Universität Heidelberg, Universität Bonn [4]"
    assert check_citations(answer, CALENDAR + [ERASMUS]) == (answer, [])


def test_a_comma_list_of_markers_becomes_separate_markers():
    # a judge run: "[26, 27]" read as no marker at all, so chunk 27 never
    # reached the judge, and the UI drew no chips for it
    answer = "Ofis: 127, e-posta: hoca@iyte.edu.tr [1, 2]. Diğer [1,2 , 3]."
    cleaned, removed = check_citations(answer, ["ofis 127", "hoca@iyte.edu.tr", "diğer"])
    assert cleaned == "Ofis: 127, e-posta: hoca@iyte.edu.tr [1][2]. Diğer [1][2][3]."
    assert removed == []


def test_only_the_wrong_marker_of_a_run_is_removed():
    # markers [1]..[2]: the renumbering shape, so the words are checked
    answer = "Almanya'da Heidelberg ve Bonn üniversiteleri var [1][2]. Başka soru?"
    cleaned, removed = check_citations(answer, [CALENDAR[0], ERASMUS])
    assert cleaned == "Almanya'da Heidelberg ve Bonn üniversiteleri var [2]. Başka soru?"
    assert removed == [1]


def test_a_translated_citation_is_kept():
    # the course catalog is English and the answer Turkish: no shared word,
    # yet the citation is right. Outside the renumbering shape it isn't judged
    course = ("CENG 322 — Operating Systems (Bilgisayar Mühendisliği, lisans dersi)\n"
              "Classification and structure of operating systems. Scheduling algorithms.")
    answer = "- İşletim sistemlerinin sınıflandırılması ve yapısı [4]"
    assert check_citations(answer, CALENDAR + [course]) == (answer, [])


def test_short_text_keeps_its_marker():
    # three words are too few to judge by shared words
    assert check_citations("Evet, Bonn var [1].", CALENDAR + [ERASMUS]) == ("Evet, Bonn var [1].", [])


def test_markers_past_the_reference_are_removed():
    cleaned, removed = check_citations("Bahar yarıyılı dersleri 15 Şubat'ta başlar [9].", CALENDAR)
    assert cleaned == "Bahar yarıyılı dersleri 15 Şubat'ta başlar."
    assert removed == [9]


def test_a_marker_too_long_for_int_is_removed_without_an_error():
    # int() raises past 4300 digits: a model told to close with such a
    # marker turned a delivered answer into a failed turn
    answer = "Bahar yarıyılı dersleri 15 Şubat'ta başlar [3][12345][" + "9" * 4301 + "]."
    cleaned, removed = check_citations(answer, CALENDAR)
    assert cleaned == "Bahar yarıyılı dersleri 15 Şubat'ta başlar [3]."
    assert removed == [12345]


def test_words_in_most_chunks_do_not_count():
    chunks = [f"Öğrenci konu {i}: ayrıntı{i}" for i in range(12)]
    answer = "Öğrenci kimlik kartı yenileme işlemi [1]."
    # "öğrenci" is in every chunk, so it says nothing about chunk [1]
    assert check_citations(answer, chunks) == ("Öğrenci kimlik kartı yenileme işlemi.", [1])
    # with too few chunks to tell common words apart, any shared word counts
    assert check_citations(answer, chunks[:5]) == (answer, [])


def test_a_marker_line_in_the_middle_is_not_a_footer():
    answer = "Bonn ve Heidelberg Almanya'da [4]\n[4]\nBaşka soru?"
    assert check_citations(answer, CALENDAR + [ERASMUS]) == (answer, [])


# FakeRetriever numbers: [1] calendar "Final sınavları 5 Ocak'ta", [3] faq
# "Kimlik kartım kayboldu? Dilekçe verin."
WRONG_THEN_RIGHT = "Kimlik kartı kaybolursa dilekçe verilir [1][3].\n\n[1] [3]"
CHECKED = "Kimlik kartı kaybolursa dilekçe verilir [3]."


class TokenLLM(FakeLLM):
    """Streams the reply as given, one string per token."""

    def chat_stream(self, model, messages):
        self.calls.append((model, [dict(m) for m in messages]))
        yield from self.reply


def test_respond_returns_and_remembers_the_checked_answer():
    conversation = Conversation(FakeLLM(reply=WRONG_THEN_RIGHT), FakeRetriever(), model="m")

    assert conversation.respond("kimlik kaybettim") == CHECKED
    assert conversation.last_answer == CHECKED
    assert conversation._messages[-1]["content"] == CHECKED
    assert any(line.startswith("citations") and "[1]" in line for line in conversation.last_debug)


def test_respond_checks_an_answer_with_a_marker_too_long_for_int():
    reply = "Kimlik kartı kaybolursa dilekçe verilir [3][" + "9" * 4301 + "]."
    conversation = Conversation(FakeLLM(reply=reply), FakeRetriever(), model="m")

    assert conversation.respond("kimlik kaybettim") == CHECKED


def test_stream_sends_the_tokens_but_keeps_the_checked_answer():
    reply = ["Kimlik kartı kaybolursa ", "dilekçe verilir [1][3].", "\n\n[1] [3]"]
    conversation = Conversation(TokenLLM(reply=reply), FakeRetriever(), model="m")

    tokens = list(conversation.respond_stream("kimlik kaybettim"))

    assert tokens[-3:] == reply  # already on screen; the UI re-renders at the end
    assert conversation.last_answer == CHECKED
    assert conversation._messages[-1]["content"] == CHECKED


def test_api_stores_and_serves_the_checked_answer(monkeypatch):
    from app import storage
    from app.api import api
    from tests.test_auth import make_client

    user = {"email": "ali@gmail.com", "name": "Ali", "picture": "", "member": False}
    client = make_client(monkeypatch, user)
    client.post("/auth/google", json={"credential": "fake"})
    conversation = Conversation(TokenLLM(reply=[WRONG_THEN_RIGHT]), FakeRetriever(), model="m")
    monkeypatch.setitem(api._sessions, ("openrouter", "cite"), (conversation, time.time()))

    streamed = client.post("/chat/stream", json={"query": "kimlik kaybettim", "session_id": "cite"})
    assert WRONG_THEN_RIGHT in streamed.text
    assert client.get("/chat/sources", params={"session_id": "cite"}).json()["answer"] == CHECKED

    # the transcript is written off-thread after the stream
    for _ in range(100):
        stored = storage.conversation_messages("cite")
        if stored:
            break
        time.sleep(0.02)
    assert stored[-1]["content"] == CHECKED


# --- add_missing_markers: real answers the judge failed for a bare fact

GYM_BIG = ("Büyük Salon'un 2025-2026 dönemi haftalık çalışma programı: Pazartesi günü 16.00-20.00 "
           "saatleri arasında Voleybol, Çarşamba günü 16.00-20.00 Voleybol yapılmaktadır.")
GYM_MIRROR = ("Aynalı Salon'un 2025-2026 dönemi haftalık çalışma programı: Pazartesi günü 16.00-20.00 "
              "saatleri arasında Pilates, Çarşamba günü 16.00-20.00 Zumba yapılmaktadır.")


def test_a_list_under_a_bare_lead_in_gets_the_source_that_holds_it():
    # two gym timetables share every day and hour; only one says
    # "Büyük" and "voleybol"
    answer = ("Büyük Salon'u yanıtlıyorum [1]. Aynalı Salon voleybol için kullanılmıyor [2].\n\n"
              "Büyük Salon'da voleybol, 2025-2026 dönemi programına göre:\n"
              "- Pazartesi: 16.00-20.00 arası\n- Çarşamba: 16.00-20.00 arası\n\n"
              "Kesin saatler için SKS Daire Başkanlığı'na danışmanızı öneririm.")
    fixed, added = add_missing_markers(answer, [GYM_BIG, GYM_MIRROR])
    assert added == [1]
    assert "Büyük Salon'da voleybol, 2025-2026 dönemi programına göre [1]:" in fixed
    assert fixed.endswith("danışmanızı öneririm.")  # advice stays bare


def test_a_list_split_by_headings_is_covered_from_its_lead_in():
    departments = ("Çift Ana Dal Programı Olan Bölümler\nMimarlık Fakültesi:\n- Mimarlık\n"
                   "- Şehir ve Bölge Planlama\nFen Fakültesi:\n- Fizik\n- Kimya")
    application = "Soru: Çift Ana Dal için nereye başvurulur?\nCevap: Öğrenci İşleri Daire Başkanlığına."
    answer = ("Çift anadal programı için şu bölümler geçerli:\n\n**Mimarlık Fakültesi**\n- Mimarlık\n"
              "- Şehir ve Bölge Planlama\n\n**Fen Fakültesi**\n- Fizik\n- Kimya\n\n"
              "Başvuru Öğrenci İşleri Daire Başkanlığına yapılır [2]. Yönerge için bakabilirsin [1].")
    fixed, added = add_missing_markers(answer, [departments, application])
    assert added == [1]
    assert fixed.startswith("Çift anadal programı için şu bölümler geçerli [1]:")


def test_a_sentence_after_the_paragraphs_marker_gets_its_own():
    profile = "Onur Demirörs — Professor, Bilgisayar Mühendisliği\nofis: 107 | e-posta: onurdemirors@iyte.edu.tr"
    answer = "Onur Demirörs'ün ofisi 107 numarada [1]. Bölümün Bilgisayar Mühendisliği olduğunu da ekleyeyim."
    fixed, added = add_missing_markers(answer, [profile])
    assert added == [1]
    assert fixed.endswith("Bölümün Bilgisayar Mühendisliği olduğunu da ekleyeyim [1].")


def test_nothing_changes_when_every_fact_is_covered_or_unclear():
    covered = "Final sınavları 11-22 Ocak 2027 arasında [1].\n\nDersler 6 Ocak'ta biter [2]."
    assert add_missing_markers(covered, ["11 Ocak 2027 ile 22 Ocak 2027 sınavlar", "6 Ocak dersler"]) == (covered, [])
    # neither cited source has words of its own in the bare sentence
    unclear = "Kayıt 8 Temmuz'da [1]. Ücret 9 Temmuz'da yatar [2].\n\nİkisi de Temmuz 2027'de yapılır."
    chunks = ["8 Temmuz 2027 kayıt", "9 Temmuz 2027 ücret"]
    assert add_missing_markers(unclear, chunks) == (unclear, [])
    # a source the answer never cites is never added
    uncited = "Dersler 6 Ocak 2027'de biter."
    assert add_missing_markers(uncited, ["6 Ocak 2027 dersler biter"]) == (uncited, [])


def test_the_answer_gets_the_missing_marker_and_the_dev_log_says_so():
    # FakeRetriever's [3] is the FAQ "Kimlik kartım kayboldu? Dilekçe verin."
    reply = "Kimlik kartı kaybolursa dilekçe verilir [3]. Kayıp Kimlik Kartı için Dilekçe yeterli."
    conversation = Conversation(FakeLLM(reply=reply), FakeRetriever(), model="m")

    assert conversation.respond("kimlik kaybettim").endswith("Dilekçe yeterli [3].")
    assert any(line.startswith("markers") and "[3]" in line for line in conversation.last_debug)


# --- correct_markers: a number the cited chunk lacks and one other chunk has

ONUR = ("Onur Demirörs — Professor / Chair, Bilgisayar Mühendisliği\nHis research areas include "
        "software process improvement.\nofis: 107 | e-posta: onurdemirors@iyte.edu.tr")
HUSEYIN = ("Hüseyin Ünlü — Dr., Bilgisayar Mühendisliği\nHe continues his PhD studies under supervision "
           "of Prof. Dr. Onur Demirörs.\nofis: 138 | e-posta: huseyinunlu@iyte.edu.tr")


def test_a_marker_moves_to_the_one_chunk_that_has_the_number():
    # the two profiles sat side by side and both name Onur Demirörs; the
    # answer said 107 but cited his student's profile (office 138)
    answer = "Onur Demirörs'ün ofisi **107** numaralıdır [2]."
    assert correct_markers(answer, [ONUR, HUSEYIN]) == (
        "Onur Demirörs'ün ofisi **107** numaralıdır [1].", [(2, 1)])


def test_a_marker_stays_when_its_chunk_has_the_number_or_no_single_chunk_does():
    right = "Onur Demirörs'ün ofisi 107 numaralıdır [1]."
    assert correct_markers(right, [ONUR, HUSEYIN]) == (right, [])
    # 2.50 is in both transfer directives: no telling which was meant
    answer = "Genel not ortalaması en az 2.50 olmalı [1]."
    chunks = ["Kurum içi: 2.50 şartı", "Kurumlar arası: 2.50 şartı", "Merkezi yerleştirme: taban puan"]
    assert correct_markers(answer, chunks[2:] + chunks[:2]) == (answer, [])
    # a number in no chunk is the judge's to catch, not ours to move
    invented = "Ofis 999 numaradadır [1]."
    assert correct_markers(invented, [ONUR, HUSEYIN]) == (invented, [])


def test_short_numbers_never_move_a_marker():
    # "6" is in "6 Ocak" of one chunk by chance, not the source of the count
    answer = "Alanı belirtilmemiş 6 araştırma görevlisi var [1]."
    assert correct_markers(answer, ["Kadro listesi", "6 Ocak dersler biter"]) == (answer, [])


def test_a_fact_next_to_advice_still_gets_its_marker():
    # judge run 2026-11-27: the dates were bare because the sentence also
    # sent the student to the Erasmus office, and the whole of it was skipped
    turna = ("TurnaPortal'dan E-Devlet ile online başvuru. Başvurular 17 Mart 2026 - 02 Nisan 2026 "
             "saat 16.00'ya kadar yapılır.")
    answer = ("Başvuru TurnaPortal üzerinden online yapılıyor [1].\n\n"
              "Başvuru dönemi (17 Mart - 2 Nisan 2026) geçtiği için şu an başvuru yapılamıyor; "
              "sonraki dönem için Erasmus birimine danışabilirsin.")
    fixed, added = add_missing_markers(answer, [turna])
    assert added == [1]
    assert fixed.endswith("Erasmus birimine danışabilirsin [1].")


def test_every_chunk_ends_with_its_own_marker():
    # an office number on a profile's last line sat right before the next
    # profile's marker, and the model cited that one (4 of 10 answers)
    conversation = Conversation(FakeLLM(reply="cevap"), FakeRetriever(), model="m")
    conversation.respond("kimlik kaybettim")
    assert conversation.last_reference
    for entry in conversation.last_reference:
        n = entry[:entry.index("]") + 1]
        assert entry.endswith(f" {n}")


LISANS_TERM = ("[2026-2027 Güz Yarıyılı] 28 Eylül 2026 Pazartesi ile 6 Ocak 2027 Çarşamba tarihleri "
               "arasında Lisans/Lisansüstü programlarda I. yarıyıl dersler.")
HAZIRLIK_TERM = ("[2026-2027 Güz Yarıyılı] 28 Eylül 2026 Pazartesi ile 20 Ocak 2027 Çarşamba tarihleri "
                 "arasında Hazırlık Sınıfı I. yarıyıl dersler.")
GUIDE = "Lisans Ders Kaydı Genel Bilgilendirme ve Kayıt İşlem Adımları. Ders seçim işlemleri yapılır."


def test_a_date_the_cited_chunk_lacks_gets_the_marker_of_the_chunk_that_has_it():
    # judge run 2026-11-27: the start of term was cited to the course
    # registration guide; "2026" is in dozens of chunks, so correct_markers
    # couldn't move it. Two calendar lines share the date: the lisans one
    # shares more words with the sentence
    answer = "Lisans dersleri 28 Eylül 2026'da başladı, ders seçimi ÖBS'den yapılır [3]."
    assert add_date_markers(answer, [HAZIRLIK_TERM, LISANS_TERM, GUIDE]) == (
        "Lisans dersleri 28 Eylül 2026'da başladı, ders seçimi ÖBS'den yapılır [3][2].", [2])


def test_a_date_stays_with_a_marker_that_backs_it_or_none_can():
    backed = "Dersler 28 Eylül 2026'da başladı [3][1]."
    assert add_date_markers(backed, [LISANS_TERM, HAZIRLIK_TERM, GUIDE]) == (backed, [])
    invented = "Dersler 29 Eylül 2026'da başladı [3]."  # in no chunk: the judge's to catch
    assert add_date_markers(invented, [LISANS_TERM, GUIDE, GUIDE]) == (invented, [])
    no_year = "Dersler 28 Eylül'de başladı [3]."  # without a year a date is too loose to place
    assert add_date_markers(no_year, [LISANS_TERM, GUIDE, GUIDE]) == (no_year, [])


def test_word_pairs_pick_the_source_when_every_word_is_in_two():
    # judge run 2026-11-27: an unmarked paragraph on askerlik tecili; each
    # of its words was in two or more of the four cited askerlik sources,
    # so no source had a word of its own and none was picked
    right = "Tecil işlemleri Öğrenci İşleri tarafından YÖKSİS üzerinden yapılır."
    shuffled = "YÖKSİS ile işlemleri üzerinden tecil, Öğrenci yapılır tarafından İşleri."
    answer = ("Tecil işlemleri Öğrenci İşleri tarafından YÖKSİS üzerinden yapılır.\n\n"
              "- Takip öğrenciye aittir [1][2].")
    out, added = add_missing_markers(answer, [shuffled, right])
    assert added == [2]
    assert out.startswith("Tecil işlemleri Öğrenci İşleri tarafından YÖKSİS üzerinden yapılır [2].")


ELECTIVES = "Seçmeli ders kaydı: ÖBS'de Ders Kaydı sekmesine tıklanarak ders seçme işlemlerine geçilir."
STEPS = "Kayıt adımları: ders seçim işlemleri yapılır, danışman onayı işlemi tamamlanır."
GUIDE_PDF = ("Öğrenci Ders Kayıt Kılavuzu (PDF): ders kaydı, ders seçme ekranı, şube seçimi ve "
             "danışman onayına gönderme adımlarını ekran görüntüleriyle anlatır.")


def test_a_phrase_only_another_chunk_has_adds_its_marker():
    # judge run 2026-11-28: the steps came from the guide PDF's chunk but
    # cited the two others; "onayına gönder" is in the guide's alone
    answer = "Ders seçme ekranına geçilir, şube seçilir ve danışman onayına gönderilir [1][2]."
    assert add_phrase_markers(answer, [ELECTIVES, STEPS, GUIDE_PDF]) == (
        "Ders seçme ekranına geçilir, şube seçilir ve danışman onayına gönderilir [1][2][3].", [3])


def test_a_phrase_marker_needs_a_closer_chunk_than_the_cited_one():
    # replay of the same run: "Pazar günleri" was in the small gym's
    # timetable alone, but the sentence was about the big hall, which
    # the answer cited and which shares far more of its words
    big = "Büyük Salon voleybol: Pazartesi 16.00-20.00, Çarşamba 16.00-20.00, Pazar 15.00-18.00"
    small = "Aynalı Salon pilates: Pazar günleri 10.00-12.00"
    answer = "Büyük Salon'da voleybol Pazartesi, Çarşamba ve Pazar günleri var [1]."
    assert add_phrase_markers(answer, [big, small]) == (answer, [])
