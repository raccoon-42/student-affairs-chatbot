"""Auth tested through the API with the Google verification faked out."""
from fastapi.testclient import TestClient

from app import auth
from app.api import api


def make_client(monkeypatch, user=None):
    def fake_verify(credential):
        if user is None:
            raise ValueError("bad token")
        return user
    monkeypatch.setattr(auth, "verify_google_token", fake_verify)
    return TestClient(api.app)


def test_member_domains():
    assert auth.is_member("ali@iyte.edu.tr")
    assert auth.is_member("ali@std.iyte.edu.tr")
    assert not auth.is_member("ali@gmail.com")
    assert not auth.is_member("ali@fakeiyte.edu.tr")


def test_signin_sets_cookie_and_me_returns_user(monkeypatch):
    user = {"email": "ali@std.iyte.edu.tr", "name": "Ali", "picture": "", "member": True}
    client = make_client(monkeypatch, user)

    response = client.post("/auth/google", json={"credential": "fake"})
    assert response.status_code == 200
    assert response.json()["member"] is True
    assert "auth_token" in response.cookies

    me = client.get("/auth/me").json()
    assert me["email"] == user["email"]
    assert me["education_type"] is None  # not asked yet


def test_invalid_token_is_rejected(monkeypatch):
    client = make_client(monkeypatch, user=None)
    response = client.post("/auth/google", json={"credential": "garbage"})
    assert response.status_code == 401


def test_logout_ends_the_session(monkeypatch):
    user = {"email": "ali@gmail.com", "name": "Ali", "picture": "", "member": False}
    client = make_client(monkeypatch, user)
    client.post("/auth/google", json={"credential": "fake"})

    client.post("/auth/logout")
    assert client.get("/auth/me").json() is None


def test_me_without_session_is_null():
    client = TestClient(api.app)
    assert client.get("/auth/me").json() is None


def test_profile_is_saved_on_the_session(monkeypatch):
    user = {"email": "ali@gmail.com", "name": "Ali", "picture": "", "member": False}
    client = make_client(monkeypatch, user)
    client.post("/auth/google", json={"credential": "fake"})

    response = client.post("/auth/profile", json={"education_type": "lisans"})
    assert response.status_code == 200
    assert client.get("/auth/me").json()["education_type"] == "lisans"

    assert client.post("/auth/profile", json={"education_type": "astronot"}).status_code == 422


def test_profile_requires_a_session():
    client = TestClient(api.app)
    assert client.post("/auth/profile", json={"education_type": "lisans"}).status_code == 401


def test_conversations_belong_to_their_owner(monkeypatch):
    from app import storage

    ali = {"email": "ali@gmail.com", "name": "Ali", "picture": "", "member": False}
    client = make_client(monkeypatch, ali)
    client.post("/auth/google", json={"credential": "fake"})
    storage.record_exchange("conv-ali", ali["email"], "soru", "cevap")

    assert client.get("/conversations").json()[0]["id"] == "conv-ali"
    detail = client.get("/conversations/conv-ali").json()
    assert detail["messages"][0]["content"] == "soru"

    # another signed-in user gets 403 on Ali's conversation
    veli = {"email": "veli@gmail.com", "name": "Veli", "picture": "", "member": False}
    other = make_client(monkeypatch, veli)
    other.post("/auth/google", json={"credential": "fake"})
    assert other.get("/conversations/conv-ali").status_code == 403
    assert other.get("/conversations").json() == []

    # anonymous gets 401 for the list, 403 for the detail
    anon = TestClient(api.app)
    assert anon.get("/conversations").status_code == 401
    assert anon.get("/conversations/conv-ali").status_code == 403


def test_import_adopts_anonymous_conversation(monkeypatch):
    ali = {"email": "ali@gmail.com", "name": "Ali", "picture": "", "member": False}
    client = make_client(monkeypatch, ali)
    client.post("/auth/google", json={"credential": "fake"})

    payload = {"id": "anon-conv", "messages": [
        {"role": "user", "text": "soru"}, {"role": "bot", "text": "cevap"}]}
    assert client.post("/conversations/import", json=payload).status_code == 200
    assert client.post("/conversations/import", json=payload).status_code == 200  # idempotent

    [conversation] = client.get("/conversations").json()
    # never the first message — a placeholder until the model titles it
    assert conversation["title"] == "Yeni konuşma"
    messages = client.get("/conversations/anon-conv").json()["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "soru"), ("assistant", "cevap")]

    # someone else's conversation id can't be hijacked
    veli = {"email": "veli@gmail.com", "name": "Veli", "picture": "", "member": False}
    other = make_client(monkeypatch, veli)
    other.post("/auth/google", json={"credential": "fake"})
    assert other.post("/conversations/import", json=payload).status_code == 403

    assert TestClient(api.app).post("/conversations/import", json=payload).status_code == 401


def test_anonymous_rate_limit_returns_429(monkeypatch):
    from app.ratelimit import RateLimiter
    # limit 0 blocks immediately — the check runs before any model call,
    # so the test stays offline
    monkeypatch.setattr(api, "_anon_limiter", RateLimiter(limit=0))

    client = TestClient(api.app)
    blocked = client.get("/chat/stream", params={"query": "merhaba", "session_id": "rl"})
    assert blocked.status_code == 429
    assert "giriş" in blocked.json()["detail"].lower()


def test_repeated_refusals_trigger_the_spam_brake(monkeypatch):
    from app.ratelimit import RateLimiter

    monkeypatch.setattr(api, "_refusal_limiter", RateLimiter(limit=2, window_seconds=600))
    client = TestClient(api.app)

    # two refusals recorded (as if the gate had blocked two messages)...
    class FakeRequest:
        client = type("C", (), {"host": "testclient"})()
    for _ in range(2):
        api._register_refusal(FakeRequest(), None, list(api.REFUSALS.values())[0])

    # ...and the third message is blocked before any model call
    blocked = client.get("/chat/stream", params={"query": "asdf", "session_id": "spam"})
    assert blocked.status_code == 429
    assert "konu dışı" in blocked.json()["detail"]


def test_one_abusive_message_blocks_the_sender(monkeypatch):
    from app.guardrails import ABUSE_MESSAGE
    from app.ratelimit import RateLimiter

    monkeypatch.setattr(api, "_abuse_limiter", RateLimiter(limit=1, window_seconds=600))
    client = TestClient(api.app)

    class FakeRequest:
        client = type("C", (), {"host": "testclient"})()
    api._register_refusal(FakeRequest(), None, ABUSE_MESSAGE)

    blocked = client.get("/chat/stream", params={"query": "merhaba", "session_id": "x"})
    assert blocked.status_code == 429
    assert blocked.headers.get("X-Block-Reason") == "abuse"


def test_exempt_key_bypasses_only_the_abuse_block(monkeypatch):
    from app.guardrails import ABUSE_MESSAGE, REFUSAL_MESSAGE
    from app.ratelimit import RateLimiter
    from config import settings

    monkeypatch.setattr(settings, "ABUSE_EXEMPT", {"testclient"})
    monkeypatch.setattr(api, "_abuse_limiter", RateLimiter(limit=1, window_seconds=600))
    monkeypatch.setattr(api, "_refusal_limiter", RateLimiter(limit=3, window_seconds=600))

    class FakeRequest:
        client = type("C", (), {"host": "testclient"})()
    api._register_refusal(FakeRequest(), None, ABUSE_MESSAGE)

    client = TestClient(api.app)
    assert client.get("/auth/config").json()["abuse_exempt"] is True
    # the abuse block is skipped...
    assert not api._abuse_limiter.is_blocked("testclient")
    # ...but the refusal still counted toward the spam brake
    api._register_refusal(FakeRequest(), None, REFUSAL_MESSAGE)
    api._register_refusal(FakeRequest(), None, REFUSAL_MESSAGE)
    assert api._refusal_limiter.is_blocked("testclient")


def test_normal_answers_do_not_count_as_refusals():
    class FakeRequest:
        client = type("C", (), {"host": "1.2.3.4"})()
    api._register_refusal(FakeRequest(), None, "normal bir cevap")
    assert not api._refusal_limiter.is_blocked("1.2.3.4")


def test_expired_session_is_rejected_and_pruned():
    import time

    from app import storage
    from config import settings

    storage.upsert_user({"email": "old@gmail.com", "name": "", "picture": "", "member": False})
    storage.create_auth_session("tok-old", "old@gmail.com")
    assert storage.get_auth_user("tok-old") is not None

    # age the token past the TTL: it must no longer authenticate...
    stale = time.time() - (settings.AUTH_SESSION_TTL_DAYS + 1) * 86400
    with storage._connect() as conn:
        conn.execute("UPDATE auth_sessions SET created_at = ? WHERE token = ?",
                     (stale, "tok-old"))
    assert storage.get_auth_user("tok-old") is None

    # ...and the next login prunes the expired row rather than letting it linger
    storage.create_auth_session("tok-new", "old@gmail.com")
    with storage._connect() as conn:
        tokens = {r["token"] for r in conn.execute("SELECT token FROM auth_sessions")}
    assert tokens == {"tok-new"}


def test_delete_conversation(monkeypatch):
    from app import storage

    ali = {"email": "ali@gmail.com", "name": "Ali", "picture": "", "member": False}
    client = make_client(monkeypatch, ali)
    client.post("/auth/google", json={"credential": "fake"})
    storage.record_exchange("conv-ali", ali["email"], "soru", "cevap")

    assert client.delete("/conversations/conv-ali").status_code == 200
    assert client.get("/conversations").json() == []
