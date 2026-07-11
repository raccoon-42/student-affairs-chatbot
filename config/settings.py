"""Single place where configuration is resolved.

Every other module imports values from here instead of reading .env or
hardcoding constants itself, so the indexer and the retriever can never
disagree on which model or collection is in force.
"""
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# Vector store
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
CALENDAR_COLLECTION = os.getenv("CALENDAR_COLLECTION", "academic_calendar_2025")
REGULATIONS_COLLECTION = os.getenv("REGULATIONS_COLLECTION", "regulations")
FAQ_COLLECTION = os.getenv("FAQ_COLLECTION", "faq")
FORMS_COLLECTION = os.getenv("FORMS_COLLECTION", "forms")
SKS_COLLECTION = os.getenv("SKS_COLLECTION", "sks")
PROGRAMS_COLLECTION = os.getenv("PROGRAMS_COLLECTION", "programs")
PEOPLE_COLLECTION = os.getenv("PEOPLE_COLLECTION", "people")
COURSES_COLLECTION = os.getenv("COURSES_COLLECTION", "courses")
GUIDES_COLLECTION = os.getenv("GUIDES_COLLECTION", "guides")

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

# Where "view sources" links point for chunks that have no URL of their
# own (calendar/regulations come from PDFs; FAQ entries carry their page)
CALENDAR_SOURCE_URL = os.getenv("CALENDAR_SOURCE_URL", "")
REGULATIONS_SOURCE_URL = os.getenv("REGULATIONS_SOURCE_URL", "")
# forms chunks always carry their own PDF URL; this is just the fallback
FORMS_SOURCE_URL = os.getenv("FORMS_SOURCE_URL", "https://ogrenciisleri.iyte.edu.tr/formlar/")
SKS_SOURCE_URL = os.getenv("SKS_SOURCE_URL", "https://sks.iyte.edu.tr/")
PROGRAMS_SOURCE_URL = os.getenv("PROGRAMS_SOURCE_URL", "https://iyte.edu.tr/akademik/lisans-programlari/")

# Per-year calendar PDFs: CALENDAR_25_26_URL="https://..." becomes
# {"2025-2026": "https://..."}. The vectorizer stamps each calendar chunk
# with its year's URL at index time, so citation chips link to the right
# PDF; add a new env var when a new academic year is published.
CALENDAR_SOURCE_URLS = {
    f"20{m.group(1)}-20{m.group(2)}": value
    for key, value in os.environ.items()
    if (m := re.fullmatch(r"CALENDAR_(\d{2})_(\d{2})_URL", key))
}

# Hybrid retrieval score = SEMANTIC_WEIGHT * cosine + BM25_WEIGHT * bm25
SEMANTIC_WEIGHT = 0.7
BM25_WEIGHT = 0.3

# LLM backends
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")

# Models a client may request via model_name; anything else is rejected so
# the public API can't be pointed at an arbitrary (expensive) model
ALLOWED_CHAT_MODELS = {m.strip() for m in os.getenv(
    "ALLOWED_CHAT_MODELS",
    "anthropic/claude-haiku-4.5,anthropic/claude-sonnet-5").split(",") if m.strip()}
# answer length cap; unset upstream defaults have truncated mid-sentence
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# Per-attempt HTTP timeout for LLM calls (the OpenAI SDK retries twice on
# top of this). Without it a stalled connection hangs a request ~10 min.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

# LLM-as-judge (evaluation) — OpenRouter by default
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-5")

# Scope gate: cheap model that decides if a question is university-related
GUARD_MODEL = os.getenv("GUARD_MODEL", "anthropic/claude-haiku-4.5")

# Speech-to-text: Groq-hosted Whisper (the UI hides the mic button when unset)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")

# Google Sign-In (optional — the UI hides the button when unset)
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

# SQLite persistence (users, auth sessions, conversations)
DB_PATH = os.getenv("DB_PATH", str(ROOT / "data" / "app.db"))

# Chat rate limits (messages per hour); anonymous users are nudged to sign in
CHAT_LIMIT_ANON = int(os.getenv("CHAT_LIMIT_ANON", "5"))
CHAT_LIMIT_USER = int(os.getenv("CHAT_LIMIT_USER", "60"))

# Voice transcriptions per hour (per user/IP, separate from the chat quota)
STT_LIMIT = int(os.getenv("STT_LIMIT", "30"))

# Spam brake: this many scope-gate refusals within 10 minutes blocks further
# messages until the window clears
CHAT_REFUSAL_LIMIT = int(os.getenv("CHAT_REFUSAL_LIMIT", "3"))

# A single abusive message blocks the sender for this long (minutes)
ABUSE_BLOCK_MINUTES = int(os.getenv("ABUSE_BLOCK_MINUTES", "10"))

# Comma-separated emails/IPs exempt from the abuse block only (dev use);
# message limits and the spam brake still apply to these keys
ABUSE_EXEMPT = {key.strip() for key in os.getenv("ABUSE_EXEMPT", "").split(",") if key.strip()}

# Keys (IPs or signed-in emails) exempt from ALL rate limiting — for the
# judge suite and local eval runs, which burn through the anonymous
# quota by design. Don't put real users here.
RATELIMIT_EXEMPT = {key.strip() for key in os.getenv("RATELIMIT_EXEMPT", "").split(",") if key.strip()}

# Set to 1 in production: the auth cookie is then only sent over HTTPS.
# Off by default because local dev runs plain http://localhost.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "") == "1"

# API (used by integration tests)
API_URL = os.getenv("API_URL", "http://localhost:8000")

SYSTEM_PROMPT_PATH = ROOT / "config" / "prompts" / "system_prompt.txt"


def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()
