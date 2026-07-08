"""SQLite persistence: one module owns the schema and every query.

Users, auth sessions, and the conversations of signed-in users live
here, so login and chat history survive restarts. Anonymous chats are
never written — they exist only in the browser and in the in-memory
conversation cache.

sqlite3 from the stdlib, one short-lived connection per operation:
no pooling, no ORM, no extra dependency. Fine at this scale.
"""
import json
import sqlite3
import time
from pathlib import Path

from config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    email TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    picture TEXT NOT NULL DEFAULT '',
    member INTEGER NOT NULL DEFAULT 0,
    education_type TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_sessions (
    token TEXT PRIMARY KEY,
    email TEXT NOT NULL REFERENCES users(email) ON DELETE CASCADE,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_email TEXT NOT NULL REFERENCES users(email) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


def _connect():
    Path(settings.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    # migration: sources column arrived after the first deployments
    columns = [row[1] for row in conn.execute("PRAGMA table_info(messages)")]
    if "sources" not in columns:
        conn.execute("ALTER TABLE messages ADD COLUMN sources TEXT")
    return conn


def _user_dict(row) -> dict:
    return {
        "email": row["email"],
        "name": row["name"],
        "picture": row["picture"],
        "member": bool(row["member"]),
        "education_type": row["education_type"],
    }


# ---------- users & auth sessions ----------

def upsert_user(user: dict):
    """Insert or refresh identity fields; education_type is never
    overwritten here — the profile prompt owns it."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO users (email, name, picture, member, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(email) DO UPDATE SET
                 name = excluded.name, picture = excluded.picture,
                 member = excluded.member""",
            (user["email"], user["name"], user["picture"], int(user["member"]), time.time()),
        )


def set_education_type(email: str, education_type: str):
    with _connect() as conn:
        conn.execute("UPDATE users SET education_type = ? WHERE email = ?",
                     (education_type, email))


def create_auth_session(token: str, email: str):
    with _connect() as conn:
        conn.execute("INSERT INTO auth_sessions (token, email, created_at) VALUES (?, ?, ?)",
                     (token, email, time.time()))


def get_auth_user(token) -> dict | None:
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            """SELECT u.* FROM auth_sessions s JOIN users u ON u.email = s.email
               WHERE s.token = ?""", (token,)).fetchone()
    return _user_dict(row) if row else None


def drop_auth_session(token):
    if not token:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM auth_sessions WHERE token = ?", (token,))


# ---------- conversations & messages ----------

def record_exchange(conversation_id: str, email: str, query: str, answer: str, sources=None):
    """One completed turn: creates the conversation row on first use
    (titled by the first question) and appends both messages. `sources`
    documents what the answer was grounded on."""
    now = time.time()
    title = query if len(query) <= 60 else query[:60] + "…"
    with _connect() as conn:
        conn.execute(
            """INSERT INTO conversations (id, user_email, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at""",
            (conversation_id, email, title, now, now),
        )
        conn.executemany(
            """INSERT INTO messages (conversation_id, role, content, created_at, sources)
               VALUES (?, ?, ?, ?, ?)""",
            [(conversation_id, "user", query, now, None),
             (conversation_id, "assistant", answer, now,
              json.dumps(sources, ensure_ascii=False) if sources else None)],
        )


def import_conversation(conversation_id: str, email: str, messages: list):
    """Adopt an anonymous conversation into an account: the browser sends
    the transcript it kept locally, we persist it as if it had always
    been the user's. Caller must have checked ownership."""
    now = time.time()
    first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
    title = first_user if len(first_user) <= 60 else first_user[:60] + "…"
    with _connect() as conn:
        conn.execute(
            """INSERT INTO conversations (id, user_email, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (conversation_id, email, title, now, now),
        )
        conn.executemany(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            [(conversation_id, m["role"], m["content"], now) for m in messages],
        )


def conversation_owner(conversation_id: str) -> str | None:
    with _connect() as conn:
        row = conn.execute("SELECT user_email FROM conversations WHERE id = ?",
                           (conversation_id,)).fetchone()
    return row["user_email"] if row else None


def list_conversations(email: str) -> list:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, title, updated_at FROM conversations
               WHERE user_email = ? ORDER BY updated_at DESC""", (email,)).fetchall()
    return [{"id": r["id"], "title": r["title"], "updated": r["updated_at"]} for r in rows]


def conversation_messages(conversation_id: str) -> list:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT role, content, created_at, sources FROM messages
               WHERE conversation_id = ? ORDER BY id""", (conversation_id,)).fetchall()
    return [
        {"role": r["role"], "content": r["content"], "created_at": r["created_at"],
         "sources": json.loads(r["sources"]) if r["sources"] else None}
        for r in rows
    ]


def delete_conversation(conversation_id: str, email: str):
    with _connect() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ? AND user_email = ?",
                     (conversation_id, email))
