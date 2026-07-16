"""Chat titling through the API with fakes on both seams.

The title must survive the conditions that used to strand conversations
on the "Yeni konuşma" placeholder: a server restart / cache eviction
between the answer and the title fetch, and a failed title call whose
retry only a later page session could issue.
"""
from fastapi.testclient import TestClient

from app import auth, storage
from app.api import api
from config import settings


class FakeLLM:
    def __init__(self, reply="cevap", title="Final Sınavı Tarihleri"):
        self.reply = reply
        self.title = title

    @staticmethod
    def _is_title_prompt(messages):
        return "dört kelimelik" in messages[-1]["content"]

    def chat(self, model, messages):
        return self.title if self._is_title_prompt(messages) else self.reply

    def chat_stream(self, model, messages):
        yield from self.reply.split(" ")


class FakeRetriever:
    def retrieve_all(self, query, audience=None):
        return {}


def make_client(monkeypatch, llm=None):
    monkeypatch.setattr(settings, "RATELIMIT_EXEMPT", {"testclient"})
    backend = (llm or FakeLLM(), FakeRetriever(), "test-model", "small-model")
    monkeypatch.setattr(api, "_backend", lambda name: backend)
    api._sessions.clear()
    user = {"email": "ali@gmail.com", "name": "Ali", "picture": "", "member": False}
    monkeypatch.setattr(auth, "verify_google_token", lambda credential: user)
    client = TestClient(api.app)
    client.post("/auth/google", json={"credential": "fake"})
    return client


def stored_title(session_id):
    with storage._connect() as conn:
        row = conn.execute("SELECT title FROM conversations WHERE id = ?",
                           (session_id,)).fetchone()
    return row["title"] if row else None


def test_title_lands_after_first_exchange(monkeypatch):
    client = make_client(monkeypatch)
    client.get("/chat", params={"query": "final ne zaman", "session_id": "conv1"})
    # no language is baked into the stored placeholder — clients localize it
    assert stored_title("conv1") == ""

    assert client.get("/chat/title",
                      params={"session_id": "conv1"}).json()["title"] == "Final Sınavı Tarihleri"
    assert stored_title("conv1") == "Final Sınavı Tarihleri"


def test_title_survives_cache_eviction(monkeypatch):
    """A restart or the 30-min TTL between answer and title fetch must not
    strand the placeholder — the endpoint rehydrates from SQLite."""
    client = make_client(monkeypatch)
    client.get("/chat", params={"query": "final ne zaman", "session_id": "conv2"})
    api._sessions.clear()

    assert client.get("/chat/title",
                      params={"session_id": "conv2"}).json()["title"] == "Final Sınavı Tarihleri"
    assert stored_title("conv2") == "Final Sınavı Tarihleri"


def test_unknown_session_is_not_rehydrated(monkeypatch):
    """No conversation row -> no title and, critically, no cache entry
    (an unauthenticated caller must not be able to grow the cache)."""
    client = make_client(monkeypatch)
    assert client.get("/chat/title", params={"session_id": "ghost"}).json()["title"] is None
    assert ("openrouter", "ghost") not in api._sessions


def test_legacy_placeholder_rows_count_as_pending(monkeypatch):
    """Rows persisted before the empty-title scheme carry the old fixed
    Turkish placeholder — they must surface as pending too."""
    client = make_client(monkeypatch)
    storage.record_exchange("conv-legacy", "ali@gmail.com", "soru", "cevap")
    storage.set_conversation_title("conv-legacy", storage.LEGACY_PLACEHOLDER_TITLE)

    [row] = client.get("/conversations").json()
    assert row["pending"] is True


def test_failed_title_is_flagged_for_later_retry(monkeypatch):
    """One stalled title call must leave a signal a future page session can
    act on: /conversations marks placeholder rows as pending."""
    class FlakyLLM(FakeLLM):
        def chat(self, model, messages):
            if self._is_title_prompt(messages):
                raise RuntimeError("upstream stall")
            return self.reply

    client = make_client(monkeypatch, llm=FlakyLLM())
    client.get("/chat", params={"query": "final ne zaman", "session_id": "conv3"})
    assert client.get("/chat/title", params={"session_id": "conv3"}).json()["title"] is None

    [row] = client.get("/conversations").json()
    assert row["pending"] is True

    # the retry (next page session, after the upstream recovered): the same
    # endpoint now succeeds and the row stops being pending
    monkeypatch.setattr(api, "_backend",
                        lambda name: (FakeLLM(), FakeRetriever(), "test-model", "small-model"))
    api._sessions.clear()
    assert client.get("/chat/title",
                      params={"session_id": "conv3"}).json()["title"] == "Final Sınavı Tarihleri"
    [row] = client.get("/conversations").json()
    assert row["pending"] is False
