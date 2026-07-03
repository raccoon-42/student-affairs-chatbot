"""Scope gate tested offline with a fake LLM."""
from app.conversation import Conversation
from app.guardrails import ScopeGate, REFUSAL_MESSAGE


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, model, messages):
        self.calls.append((model, messages))
        return self.reply


class ExplodingRetriever:
    """Fails the test if retrieval is reached at all."""

    def retrieve_calendar(self, query, top_k=10):
        raise AssertionError("retrieval must not run for blocked queries")

    def retrieve_regulations(self, query, top_k=3):
        raise AssertionError("retrieval must not run for blocked queries")


def test_evet_allows():
    assert ScopeGate(FakeLLM("evet"), "m").allows("sınavlar ne zaman") is True


def test_hayir_blocks():
    assert ScopeGate(FakeLLM("Hayır."), "m").allows("bana bir şiir yaz") is False


def test_unclear_reply_falls_open():
    # matches the prompt's "emin değilsen evet" — only explicit hayır blocks
    assert ScopeGate(FakeLLM("Bu soru kısmen ilgili olabilir"), "m").allows("soru") is True


def test_empty_reply_falls_open():
    assert ScopeGate(FakeLLM("  "), "m").allows("soru") is True


def test_gate_prompt_contains_the_query():
    llm = FakeLLM("evet")
    ScopeGate(llm, "m").allows("harç ücreti ne kadar")
    assert "harç ücreti ne kadar" in llm.calls[0][1][0]["content"]


def test_blocked_query_returns_refusal_and_touches_nothing():
    gate = ScopeGate(FakeLLM("hayır"), "gate-model")
    main_llm = FakeLLM("asıl cevap")
    conversation = Conversation(main_llm, ExplodingRetriever(), "main-model", gate=gate)

    answer = conversation.respond("bitcoin alsam mı")

    assert answer == REFUSAL_MESSAGE
    assert main_llm.calls == []  # main model never called
    assert conversation._messages == []  # nothing enters history


def test_blocked_query_streams_only_the_refusal():
    gate = ScopeGate(FakeLLM("hayır"), "gate-model")
    conversation = Conversation(FakeLLM("asıl cevap"), ExplodingRetriever(), "main-model", gate=gate)

    assert list(conversation.respond_stream("bana kod yaz")) == [REFUSAL_MESSAGE]


def test_allowed_query_flows_through():
    class FakeRetriever:
        def retrieve_calendar(self, query, top_k=10):
            return []

        def retrieve_regulations(self, query, top_k=3):
            return []

    gate = ScopeGate(FakeLLM("evet"), "gate-model")
    main_llm = FakeLLM("asıl cevap")
    conversation = Conversation(main_llm, FakeRetriever(), "main-model", gate=gate)

    assert conversation.respond("sınavlar ne zaman") == "asıl cevap"
