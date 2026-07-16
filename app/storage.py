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
CREATE TABLE IF NOT EXISTS usage_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS usage_log_key_time ON usage_log (key, created_at);
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
    # migration: cost column arrived a day after usage_log
    columns = [row[1] for row in conn.execute("PRAGMA table_info(usage_log)")]
    if "cost" not in columns:
        conn.execute("ALTER TABLE usage_log ADD COLUMN cost REAL")
    # migration: per-message usage, so conversation totals survive reloads
    columns = [row[1] for row in conn.execute("PRAGMA table_info(messages)")]
    if "usage" not in columns:
        conn.execute('ALTER TABLE messages ADD COLUMN "usage" TEXT')
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

def record_exchange(conversation_id: str, email: str, query: str, answer: str, sources=None,
                    usage=None):
    """One completed turn: creates the conversation row on first use
    (titled by the first question) and appends both messages. `sources`
    documents what the answer was grounded on, `usage` what it cost."""
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
            """INSERT INTO messages (conversation_id, role, content, created_at, sources, "usage")
               VALUES (?, ?, ?, ?, ?, ?)""",
            [(conversation_id, "user", query, now, None, None),
             (conversation_id, "assistant", answer, now,
              json.dumps(sources, ensure_ascii=False) if sources else None,
              json.dumps(usage) if usage else None)],
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


# ---------- token usage ----------

def record_usage(key: str, model: str, prompt_tokens: int, completion_tokens: int,
                 cost: float = None):
    """One main-model call. `key` is the rate-limit identity: the email of
    a signed-in user, otherwise the client IP — anonymous traffic counts
    too, unlike conversation persistence. `cost` is the charged USD amount
    OpenRouter reported, when it did."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO usage_log (key, model, prompt_tokens, completion_tokens, cost, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (key, model, prompt_tokens, completion_tokens, cost, time.time()),
        )


def usage_by_key(days: float = 7) -> list:
    """Per-person totals over the window, biggest spender first — the
    outlier view."""
    since = time.time() - days * 86400
    with _connect() as conn:
        rows = conn.execute(
            """SELECT key, COUNT(*) AS messages,
                      SUM(prompt_tokens) AS prompt_tokens,
                      SUM(completion_tokens) AS completion_tokens,
                      SUM(cost) AS cost,
                      MAX(created_at) AS last_seen
               FROM usage_log WHERE created_at >= ?
               GROUP BY key
               ORDER BY SUM(prompt_tokens) + SUM(completion_tokens) DESC""",
            (since,)).fetchall()
    return [dict(row) for row in rows]


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
            """SELECT role, content, created_at, sources, "usage" FROM messages
               WHERE conversation_id = ? ORDER BY id""", (conversation_id,)).fetchall()
    return [
        {"role": r["role"], "content": r["content"], "created_at": r["created_at"],
         "sources": json.loads(r["sources"]) if r["sources"] else None,
         "usage": json.loads(r["usage"]) if r["usage"] else None}
        for r in rows
    ]


def delete_conversation(conversation_id: str, email: str):
    with _connect() as conn:
        conn.execute("DELETE FROM conversations WHERE id = ? AND user_email = ?",
                     (conversation_id, email))
