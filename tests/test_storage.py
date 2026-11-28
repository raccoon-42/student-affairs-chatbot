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


def test_a_copy_of_the_database_holds_no_working_session():
    storage.upsert_user(USER)
    storage.create_auth_session("secret-token", USER["email"])

    with storage._connect() as conn:
        [stored] = [row[0] for row in conn.execute("SELECT token_hash FROM auth_sessions")]
    assert stored != "secret-token"
    # the stored value itself, sent as the cookie, signs nobody in
    assert storage.get_auth_user(stored) is None
    assert storage.get_auth_user("secret-token")["email"] == USER["email"]


def test_cleartext_sessions_are_dropped_by_the_migration():
    import sqlite3

    from config import settings

    # a database from before hashing: the token itself is the key
    conn = sqlite3.connect(settings.DB_PATH)
    conn.executescript("""
        CREATE TABLE users (email TEXT PRIMARY KEY, name TEXT NOT NULL DEFAULT '',
            picture TEXT NOT NULL DEFAULT '', member INTEGER NOT NULL DEFAULT 0,
            education_type TEXT, created_at REAL NOT NULL);
        CREATE TABLE auth_sessions (token TEXT PRIMARY KEY,
            email TEXT NOT NULL REFERENCES users(email) ON DELETE CASCADE,
            created_at REAL NOT NULL);
        INSERT INTO users (email, created_at) VALUES ('ali@std.iyte.edu.tr', 0);
        INSERT INTO auth_sessions VALUES ('legacy-token', 'ali@std.iyte.edu.tr', 1e12);
    """)
    conn.commit()
    conn.close()

    # a token taken from an old copy of the file no longer signs in...
    assert storage.get_auth_user("legacy-token") is None
    # ...and new sessions work after the upgrade
    storage.create_auth_session("fresh-token", USER["email"])
    assert storage.get_auth_user("fresh-token")["email"] == USER["email"]


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
