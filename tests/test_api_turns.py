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
