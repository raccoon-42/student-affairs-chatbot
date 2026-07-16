"""Conversation tested with fakes on both seams: no LLM, no vector store."""
from app.conversation import Conversation, QueryRewriter, STAGE_GATING, STAGE_SEARCHING, STAGE_WRITING


class FakeLLM:
    def __init__(self, reply="cevap"):
        self.reply = reply
        self.calls = []

    def chat(self, model, messages):
        self.calls.append((model, [dict(m) for m in messages]))
        return self.reply

    def chat_stream(self, model, messages):
        self.calls.append((model, [dict(m) for m in messages]))
        yield from self.reply.split(" ")


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve_calendar(self, query, top_k=10):
        return [{"text": "📝 Final sınavları 5 Ocak'ta", "score": 1.0, "metadata": {}}]

    def retrieve_regulations(self, query, top_k=3):
        return [{"text": "📖 Madde 12: sınav tekrarı", "score": 1.0, "metadata": {}}]

    def retrieve_faq(self, query, top_k=3):
        return [{"text": "Soru: Kimlik kartım kayboldu?\nCevap: Dilekçe verin.", "score": 1.0, "metadata": {}}]

    def retrieve_forms(self, query, top_k=3):
        return [{"text": "Öğrenci Kimlik Kartı Formu\nKayıp kimlik yerine yenisini almak için doldurulur.",
                 "score": 1.0,
                 "metadata": {"document_title": "Öğrenci Kimlik Kartı Formu",
                              "source_url": "https://example.test/kimlik-formu.pdf"}}]

    def retrieve_sks(self, query, top_k=4):
        return [{"text": "YEMEKHANE HİZMETİ\nKart sistemine para yüklenir.", "score": 1.0,
                 "metadata": {"document_title": "YEMEKHANE HİZMETİ", "topic": "yemekhane",
                              "source_url": "https://example.test/yemekhane/"}}]

    def retrieve_programs(self, query, top_k=3):
        return [{"text": "Bilgisayar Mühendisliği (Lisans, Mühendislik Fakültesi)", "score": 1.0,
                 "metadata": {"document_title": "Bilgisayar Mühendisliği", "level": "lisans",
                              "source_url": "https://example.test/ceng/"}}]

    def retrieve_people(self, query, top_k=3):
        return [{"text": "Selma Tekir — Associate Professor, Bilgisayar Mühendisliği", "score": 1.0,
                 "metadata": {"document_title": "Selma Tekir", "department": "Bilgisayar Mühendisliği",
                              "role": "akademik", "source_url": "https://example.test/selma-tekir/"}}]

    def retrieve_all(self, query, audience=None):
        self.queries.append(query)
        return {
            "calendar": self.retrieve_calendar(query),
            "regulations": self.retrieve_regulations(query),
            "faq": self.retrieve_faq(query),
            "forms": self.retrieve_forms(query),
            "sks": self.retrieve_sks(query),
            "programs": self.retrieve_programs(query),
            "people": self.retrieve_people(query),
        }


def make_conversation(llm=None):
    return Conversation(llm or FakeLLM(), FakeRetriever(), model="test-model")


def test_context_contains_query_and_retrieved_chunks():
    llm = FakeLLM()
    make_conversation(llm).respond("sınavlar ne zaman")

    _, messages = llm.calls[0]
    user_message = messages[-1]["content"]
    assert "sınavlar ne zaman" in user_message
    assert "Final sınavları 5 Ocak'ta" in user_message
    assert "Madde 12" in user_message
    assert "Kimlik kartım kayboldu" in user_message


def test_system_prompt_is_first_message_once():
    llm = FakeLLM()
    conversation = make_conversation(llm)
    conversation.respond("soru 1")
    conversation.respond("soru 2")

    _, messages = llm.calls[1]
    assert messages[0]["role"] == "system"
    assert sum(1 for m in messages if m["role"] == "system") == 1


def test_assistant_replies_are_kept_in_history():
    llm = FakeLLM(reply="ilk cevap")
    conversation = make_conversation(llm)
    conversation.respond("soru 1")
    conversation.respond("soru 2")

    _, messages = llm.calls[1]
    assert {"role": "assistant", "content": "ilk cevap"} in messages


def test_history_is_trimmed_to_max_exchanges():
    llm = FakeLLM()
    conversation = Conversation(llm, FakeRetriever(), model="test-model", max_exchanges=2)
    for i in range(6):
        conversation.respond(f"soru {i}")

    # system + at most 2 exchanges (2 user + 2 assistant)
    assert len(conversation._messages) <= 5
    assert conversation._messages[0]["role"] == "system"


def test_reset_clears_history():
    conversation = make_conversation()
    conversation.respond("soru")
    conversation.reset()
    assert conversation._messages == []


def test_stream_yields_tokens_and_stores_full_answer():
    llm = FakeLLM(reply="parça parça cevap")
    conversation = make_conversation(llm)

    tokens = list(conversation.respond_stream("soru"))

    # the stage markers lead the stream so the UI can animate the cursor
    assert tokens == [STAGE_GATING, STAGE_SEARCHING, STAGE_WRITING, "parça", "parça", "cevap"]
    # history holds the joined answer, not the pieces
    assert conversation._messages[-1] == {"role": "assistant", "content": "parçaparçacevap"}


def test_stream_captures_usage_from_generator_return():
    class UsageLLM(FakeLLM):
        def chat_stream(self, model, messages):
            yield from super().chat_stream(model, messages)
            return {"prompt_tokens": 100, "completion_tokens": 7}

    conversation = make_conversation(UsageLLM(reply="cevap"))
    list(conversation.respond_stream("soru"))

    assert conversation.last_usage == {"prompt_tokens": 100, "completion_tokens": 7}
    assert any(line.startswith("usage") for line in conversation.last_debug)


def test_stream_without_usage_leaves_none():
    conversation = make_conversation(FakeLLM(reply="cevap"))
    list(conversation.respond_stream("soru"))

    assert conversation.last_usage is None


def test_suggest_title_summarizes_first_exchange():
    llm = FakeLLM(reply="Dersten Çekilme Tarihleri")
    conversation = make_conversation(llm)

    assert conversation.suggest_title("model") == (None, None)  # nothing to summarize yet
    conversation.respond("dersten çekilme ne zaman")
    title, _usage = conversation.suggest_title("model")

    assert title == "Dersten Çekilme Tarihleri"
    # the title prompt carries the exchange, not the retrieval context
    prompt = llm.calls[-1][1][-1]["content"]
    assert "dersten çekilme ne zaman" in prompt


def test_stream_history_matches_respond_history():
    streamed = make_conversation(FakeLLM(reply="cevap"))
    list(streamed.respond_stream("soru"))

    plain = make_conversation(FakeLLM(reply="cevap"))
    plain.respond("soru")

    assert streamed._messages == plain._messages


def test_per_call_model_override():
    llm = FakeLLM()
    make_conversation(llm).respond("soru", model="another-model")
    assert llm.calls[0][0] == "another-model"


def test_loaded_history_is_visible_to_the_model():
    llm = FakeLLM()
    conversation = make_conversation(llm)
    conversation.load_history([
        {"role": "user", "content": "eski soru"},
        {"role": "assistant", "content": "eski cevap"},
    ])
    conversation.respond("devam sorusu")

    _, messages = llm.calls[0]
    assert messages[0]["role"] == "system"
    assert {"role": "assistant", "content": "eski cevap"} in messages


def test_education_type_reaches_the_model():
    llm = FakeLLM()
    make_conversation(llm).respond("kaç kredi lazım", education_type="lisans")

    _, messages = llm.calls[0]
    assert "lisans öğrencisi" in messages[-1]["content"]


def test_missing_education_type_is_marked_unknown():
    llm = FakeLLM()
    make_conversation(llm).respond("kaç kredi lazım")

    _, messages = llm.calls[0]
    assert "bilinmiyor" in messages[-1]["content"]


def test_history_keeps_bare_question_not_reference_data():
    conversation = make_conversation()
    conversation.respond("sınavlar ne zaman")

    user_messages = [m for m in conversation._messages if m["role"] == "user"]
    assert user_messages == [{"role": "user", "content": "sınavlar ne zaman"}]


def test_rewriter_feeds_retrieval_but_model_sees_original():
    rewriter_llm = FakeLLM(reply="lisans için kaç kredi gerekiyor?")
    retriever = FakeRetriever()
    main_llm = FakeLLM(reply="cevap")
    conversation = Conversation(main_llm, retriever, model="test-model",
                                rewriter=QueryRewriter(rewriter_llm, "small-model"))
    conversation.respond("kaç kredi lazım")          # first turn: no history, no rewrite
    conversation.respond("lisans icin soruyorum")    # follow-up: rewritten

    assert retriever.queries == ["kaç kredi lazım", "lisans için kaç kredi gerekiyor?"]
    _, messages = main_llm.calls[-1]
    assert "lisans icin soruyorum" in messages[-1]["content"]  # original, not rewrite


def test_rewriter_failure_falls_back_to_raw_query():
    class ExplodingLLM:
        def chat(self, model, messages):
            raise RuntimeError("boom")

    retriever = FakeRetriever()
    conversation = Conversation(FakeLLM(), retriever, model="test-model",
                                rewriter=QueryRewriter(ExplodingLLM(), "small-model"))
    conversation.respond("soru 1")
    conversation.respond("soru 2")
    assert retriever.queries == ["soru 1", "soru 2"]


def test_api_sessions_are_isolated_and_expire():
    import time as time_module
    from app.api import api

    a = api.get_conversation("session-a")
    b = api.get_conversation("session-b")
    assert a is not b
    assert api.get_conversation("session-a") is a  # same id -> same history

    # idle sessions are evicted after the TTL
    key = ("openrouter", "session-a")
    conversation, last_used = api._sessions[key]
    api._sessions[key] = (conversation, last_used - api.SESSION_TTL_SECONDS - 1)
    api.get_conversation("session-b")  # any call triggers eviction
    assert key not in api._sessions
