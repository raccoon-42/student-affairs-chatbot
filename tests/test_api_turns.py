"""One chat turn through the API, with fakes on both seams: what a turn
leaves behind when it fails, is stopped or overlaps another."""
from fastapi.testclient import TestClient

from app import auth
from app.api import api
from config import settings


class FakeLLM:
    def __init__(self, reply="bir iki üç"):
        self.reply = reply

    def chat(self, model, messages):
        return self.reply

    def chat_stream(self, model, messages):
        yield from self.reply.split(" ")


class FakeRetriever:
    def retrieve_all(self, query, audience=None, also=(), parts=()):
        return {}


def make_client(monkeypatch, llm=None, exempt=True):
    if exempt:
        monkeypatch.setattr(settings, "RATELIMIT_EXEMPT", {"testclient"})
    backend = (llm or FakeLLM(), FakeRetriever(), "test-model", "small-model")
    monkeypatch.setattr(api, "_backend", lambda name: backend)
    api._sessions.clear()
    user = {"email": "ali@gmail.com", "name": "Ali", "picture": "", "member": False}
    monkeypatch.setattr(auth, "verify_google_token", lambda credential: user)
    client = TestClient(api.app)
    client.post("/auth/google", json={"credential": "fake"})
    return client


def anonymous_request():
    from starlette.requests import Request
    return Request({"type": "http", "client": ("testclient", 0), "headers": [],
                    "query_string": b"", "method": "POST", "path": "/chat/stream"})


def test_a_stopped_stream_leaves_no_trace_and_still_counts_its_calls(monkeypatch):
    # regression (#4): Stop closed the HTTP stream mid-answer; the turn's
    # reference data stayed in the history, and its usage was never logged.
    # A TestClient stream runs the whole answer, so the body generator is
    # taken from _stream_chat and closed after its first token
    logged = []
    monkeypatch.setattr(api, "_record_turn_usage", lambda key, entries: logged.append(key))
    monkeypatch.setattr(api, "StreamingResponse", lambda body, **kwargs: body)
    make_client(monkeypatch)
    body = api._stream_chat(anonymous_request(), "final ne zaman", "stopped", None, None)
    for token in body:
        if token not in api.STAGE_MARKERS:
            break
    body.close()
    conversation = api._sessions[("openrouter", "stopped")][0]
    assert [m["role"] for m in conversation._messages] in ([], ["system"])
    assert logged == ["testclient"]


def on_event_loop():
    import asyncio
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def test_llm_and_speech_calls_run_off_the_event_loop(monkeypatch):
    # regression (#11): /chat and /chat/title were async defs calling the
    # LLM synchronously, and /transcribe waited on Groq the same way, so
    # every other user's stream froze until the call returned
    seen = []

    class RecordingLLM(FakeLLM):
        def chat(self, model, messages):
            seen.append(on_event_loop())
            return "Final Tarihleri" if "dört kelimelik" in messages[-1]["content"] else "cevap"

    client = make_client(monkeypatch, llm=RecordingLLM())
    client.get("/chat", params={"query": "final ne zaman", "session_id": "loop"})
    client.get("/chat/title", params={"session_id": "loop"})

    class Groq:
        status_code = 200

        def json(self):
            return {"text": "merhaba"}

    def fake_post(*args, **kwargs):
        seen.append(on_event_loop())
        return Groq()

    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr(api.requests, "post", fake_post)
    assert client.post("/transcribe", content=b"audio").json() == {"text": "merhaba"}
    assert seen and not any(seen)


def test_a_conversation_takes_one_turn_at_a_time(monkeypatch):
    # regression (#2): two tabs sending to one conversation at once shared
    # its history, sources and citation check
    import time
    client = make_client(monkeypatch)
    assert client.get("/chat", params={"query": "ilk", "session_id": "busy"}).status_code == 200
    conversation = api._sessions[("openrouter", "busy")][0]
    assert conversation.turn_started is None  # released after the turn

    conversation.turn_started = time.time()  # another tab's answer is being written
    blocked = client.post("/chat/stream", json={"query": "ikinci", "session_id": "busy"})
    assert blocked.status_code == 409
    assert "hâlâ yazılıyor" in blocked.json()["detail"]

    # a claim left by a stream that never ran lapses instead of locking the chat
    conversation.turn_started = time.time() - api.TURN_TIMEOUT_SECONDS - 1
    with client.stream("POST", "/chat/stream",
                       json={"query": "üçüncü", "session_id": "busy"}) as response:
        assert response.status_code == 200
        "".join(response.iter_text())
    assert conversation.turn_started is None
