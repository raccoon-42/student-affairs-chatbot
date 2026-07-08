"""Single place where configuration is resolved.

Every other module imports values from here instead of reading .env or
hardcoding constants itself, so the indexer and the retriever can never
disagree on which model or collection is in force.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# Vector store
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CALENDAR_COLLECTION = os.getenv("CALENDAR_COLLECTION", "academic_calendar_2025")
REGULATIONS_COLLECTION = os.getenv("REGULATIONS_COLLECTION", "regulations")
FAQ_COLLECTION = os.getenv("FAQ_COLLECTION", "faq")

# Embeddings (used by BOTH indexing and querying)
# "openrouter" (hosted) or "local" (sentence-transformers). Collections
# embedded with one backend are not searchable with the other — re-index
# from the source files (vectorizer) after switching.
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "openrouter")
OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "google/gemini-embedding-2")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large-instruct")
EMBED_INSTRUCTION = (
    "Üniversite yönetmeliği veya akademik takvim ile ilgili Türkçe bir soruya "
    "yanıt verebilecek ilgili pasajları getir"
)

# Hybrid retrieval score = SEMANTIC_WEIGHT * cosine + BM25_WEIGHT * bm25
SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT = 0.3

# LLM backends
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-5")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

# LLM-as-judge (evaluation) — OpenRouter by default
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-5")

# Scope gate: cheap model that decides if a question is university-related
GUARD_MODEL = os.getenv("GUARD_MODEL", "anthropic/claude-haiku-4.5")

# Google Sign-In (optional — the UI hides the button when unset)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

# SQLite persistence (users, auth sessions, conversations)
DB_PATH = os.getenv("DB_PATH", str(ROOT / "data" / "app.db"))

# Chat rate limits (messages per hour); anonymous users are nudged to sign in
CHAT_LIMIT_ANON = int(os.getenv("CHAT_LIMIT_ANON", "5"))
CHAT_LIMIT_USER = int(os.getenv("CHAT_LIMIT_USER", "60"))

# Spam brake: this many scope-gate refusals within 10 minutes blocks further
# messages until the window clears
CHAT_REFUSAL_LIMIT = int(os.getenv("CHAT_REFUSAL_LIMIT", "3"))

# A single abusive message blocks the sender for this long (minutes)
ABUSE_BLOCK_MINUTES = int(os.getenv("ABUSE_BLOCK_MINUTES", "10"))

# Comma-separated emails/IPs exempt from the abuse block only (dev use);
# message limits and the spam brake still apply to these keys
ABUSE_EXEMPT = {key.strip() for key in os.getenv("ABUSE_EXEMPT", "").split(",") if key.strip()}

# API (used by integration tests)
API_URL = os.getenv("API_URL", "http://localhost:8000")

SYSTEM_PROMPT_PATH = ROOT / "config" / "prompts" / "system_prompt.txt"


def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()
