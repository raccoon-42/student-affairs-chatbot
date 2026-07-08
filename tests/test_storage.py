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


def test_exchanges_accumulate_with_title_from_first_question():
    storage.upsert_user(USER)
    storage.record_exchange("conv1", USER["email"], "ilk soru", "ilk cevap")
    storage.record_exchange("conv1", USER["email"], "ikinci soru", "ikinci cevap")

    [conversation] = storage.list_conversations(USER["email"])
    assert conversation["title"] == "ilk soru"
    assert storage.conversation_messages("conv1") == [
        {"role": "user", "content": "ilk soru"},
        {"role": "assistant", "content": "ilk cevap"},
        {"role": "user", "content": "ikinci soru"},
        {"role": "assistant", "content": "ikinci cevap"},
    ]


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
