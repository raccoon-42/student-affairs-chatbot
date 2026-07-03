"""Conversation tested with fakes on both seams: no LLM, no vector store."""
from app.conversation import Conversation


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
    def retrieve_calendar(self, query, top_k=10):
        return [{"text": "📝 Final sınavları 5 Ocak'ta", "score": 1.0, "metadata": {}}]

    def retrieve_regulations(self, query, top_k=3):
        return [{"text": "📖 Madde 12: sınav tekrarı", "score": 1.0, "metadata": {}}]

    def retrieve_faq(self, query, top_k=3):
        return [{"text": "Soru: Kimlik kartım kayboldu?\nCevap: Dilekçe verin.", "score": 1.0, "metadata": {}}]

    def retrieve_all(self, query):
        return {
            "calendar": self.retrieve_calendar(query),
            "regulations": self.retrieve_regulations(query),
            "faq": self.retrieve_faq(query),
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

    assert tokens == ["parça", "parça", "cevap"]
    # history holds the joined answer, not the pieces
    assert conversation._messages[-1] == {"role": "assistant", "content": "parçaparçacevap"}


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
