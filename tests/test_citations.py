"""The citation check: markers whose chunk doesn't back the text are removed."""
import time

from app.citations import check_citations
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

    streamed = client.get("/chat/stream", params={"query": "kimlik kaybettim", "session_id": "cite"})
    assert WRONG_THEN_RIGHT in streamed.text
    assert client.get("/chat/sources", params={"session_id": "cite"}).json()["answer"] == CHECKED

    # the transcript is written off-thread after the stream
    for _ in range(100):
        stored = storage.conversation_messages("cite")
        if stored:
            break
        time.sleep(0.02)
    assert stored[-1]["content"] == CHECKED
