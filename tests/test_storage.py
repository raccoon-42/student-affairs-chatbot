"""Persistence tested against a throwaway SQLite file (conftest fixture)."""
from app import storage

USER = {"email": "ali@std.iyte.edu.tr", "name": "Ali", "picture": "", "member": True}


def test_education_type_survives_re_login():
    storage.upsert_user(USER)
    storage.set_education_type(USER["email"], "lisans")

    storage.upsert_user(USER)  # signing in again must not wipe the profile
    storage.create_auth_session("token2", USER["email"])
    assert storage.get_auth_user("token2")["education_type"] == "lisans"


def test_auth_session_roundtrip():
    storage.upsert_user(USER)
    storage.create_auth_session("token1", USER["email"])

    user = storage.get_auth_user("token1")
    assert user["email"] == USER["email"]
    assert user["member"] is True

    storage.drop_auth_session("token1")
    assert storage.get_auth_user("token1") is None


def test_exchanges_accumulate_with_placeholder_title():
    storage.upsert_user(USER)
    storage.record_exchange("conv1", USER["email"], "ilk soru", "ilk cevap")
    storage.record_exchange("conv1", USER["email"], "ikinci soru", "ikinci cevap")

    [conversation] = storage.list_conversations(USER["email"])
    # never the question text — empty until the model writes the title,
    # and flagged pending so clients show their localized placeholder
    assert conversation["title"] == ""
    assert conversation["pending"] is True

    storage.set_conversation_title("conv1", "Ders Kaydı")
    [conversation] = storage.list_conversations(USER["email"])
    assert conversation["title"] == "Ders Kaydı"
    assert conversation["pending"] is False
    messages = storage.conversation_messages("conv1")
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "ilk soru"), ("assistant", "ilk cevap"),
        ("user", "ikinci soru"), ("assistant", "ikinci cevap"),
    ]
    assert all(m["created_at"] for m in messages)


def test_sources_roundtrip():
    storage.upsert_user(USER)
    sources = [{"type": "SSS", "label": "Kaç AKTS ile mezun olabilirim?", "url": "https://x"}]
    storage.record_exchange("conv-s", USER["email"], "soru", "cevap", sources=sources)

    messages = storage.conversation_messages("conv-s")
    assert messages[0]["sources"] is None       # user message carries none
    assert messages[1]["sources"] == sources    # assistant message does


def test_ownership_and_delete():
    storage.upsert_user(USER)
    storage.record_exchange("conv1", USER["email"], "soru", "cevap")

    assert storage.conversation_owner("conv1") == USER["email"]
    assert storage.conversation_owner("unknown") is None

    storage.delete_conversation("conv1", "someone@else.com")  # not the owner: no-op
    assert storage.conversation_owner("conv1") == USER["email"]

    storage.delete_conversation("conv1", USER["email"])
    assert storage.conversation_owner("conv1") is None
    assert storage.conversation_messages("conv1") == []  # cascade removed them
