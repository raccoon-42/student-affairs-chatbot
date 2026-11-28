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
CALENDAR_COLLECTION = os.getenv("CALENDAR_COLLECTION", "calendar")
REGULATIONS_COLLECTION = os.getenv("REGULATIONS_COLLECTION", "regulations")
FAQ_COLLECTION = os.getenv("FAQ_COLLECTION", "faq")
FORMS_COLLECTION = os.getenv("FORMS_COLLECTION", "forms")
SKS_COLLECTION = os.getenv("SKS_COLLECTION", "sks")
PROGRAMS_COLLECTION = os.getenv("PROGRAMS_COLLECTION", "programs")
PEOPLE_COLLECTION = os.getenv("PEOPLE_COLLECTION", "people")
COURSES_COLLECTION = os.getenv("COURSES_COLLECTION", "courses")
GUIDES_COLLECTION = os.getenv("GUIDES_COLLECTION", "guides")

# Embeddings (used by BOTH indexing and querying). Collections embedded
# with one model are not searchable with another — re-index from the
# source files (vectorizer) after switching.
OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "google/gemini-embedding-2")

# Hybrid retrieval: dense and BM25 each rank the collection, then the two
# rankings are fused: RRF score = sum over both of 1 / (rank + RRF_K),
# rank counted from 0
# Qdrant's default. Small k lets a #1 in either ranking win: an exact course
# code that dense ranks 23rd but BM25 ranks 1st lands 2nd (with the RRF
# paper's k=60 it landed 7th, past the courses top_k)
RRF_K = 2
# The calendar holds ~100 near-identical exam events; there BM25's top ranks
# are ties ("sınav" matches them all), and k=2 let them push the right event
# from 5th to 12th. A flatter k keeps dense's picks in play.
CALENDAR_RRF_K = 20
HYBRID_PREFETCH = 50  # how deep each ranking goes before fusion
# A collection whose closest chunk (dense cosine) is more than this below the
# best collection's is left out of the answer call. On the retrieval eval a
# needed collection fell at most 0.162 below (docs/adr/0013). None keeps all.
ROUTE_GAP = 0.20

# LLM backends
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-haiku-5.5")

# Models a client may request via model_name; anything else is rejected so
# the public API can't be pointed at an arbitrary (expensive) model
ALLOWED_CHAT_MODELS = {m.strip() for m in os.getenv(
    "ALLOWED_CHAT_MODELS",
    "anthropic/claude-haiku-5.5,anthropic/claude-sonnet-5.5").split(",") if m.strip()}
# Models run with thinking off. Haiku 5.5 thinks by default, which pushed the
# first visible text from ~3s to 6-11s with no answer gain measured; Sonnet
# 5.5 rejects thinking-off with a 400, so it must never be listed here.
REASONING_OFF_MODELS = {m.strip() for m in os.getenv(
    "REASONING_OFF_MODELS", "anthropic/claude-haiku-5.5").split(",") if m.strip()}
# answer length cap; unset upstream defaults have truncated mid-sentence.
# Haiku 5.5 / Sonnet 5.5 think by default and thinking counts against it,
# so it leaves room for thinking plus a full course list
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
# chat title cap: four words fit easily, and a first message that talks the
# model into an essay stops here, not at LLM_MAX_TOKENS (over 80 characters
# is thrown away anyway)
TITLE_MAX_TOKENS = int(os.getenv("TITLE_MAX_TOKENS", "64"))

# context window of the chat models (Haiku 5.5 and Sonnet 5.5 are both 1M);
# only used for the context-fill percentage shown in the UI
LLM_CONTEXT_WINDOW = int(os.getenv("LLM_CONTEXT_WINDOW", "1000000"))

# Per-attempt HTTP timeout for LLM calls (the OpenAI SDK retries twice on
# top of this). Without it a stalled connection hangs a request ~10 min.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))

# LLM-as-judge (evaluation) — OpenRouter by default
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-5.5")
# comparing one answer with an expected one needs little thinking, and
# thinking is billed as output
JUDGE_REASONING_EFFORT = os.getenv("JUDGE_REASONING_EFFORT", "low")
# "claude-cli" runs the judge through the local `claude` CLI on a Claude
# Code subscription instead of billed OpenRouter calls; the chatbot under
# test always answers through OpenRouter, as in production
JUDGE_BACKEND = os.getenv("JUDGE_BACKEND", "openrouter")

# Scope gate: cheap model that decides if a question is university-related
GUARD_MODEL = os.getenv("GUARD_MODEL", "anthropic/claude-haiku-5.5")

# Reading an attached image into a search query: a misread screenshot (a
# calendar table, a form) derails the whole answer, so not the cheap model
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "anthropic/claude-sonnet-5.5")

# Corpus parsers (PDF/HTML -> chunks, descriptions, area tags). Separate from
# the chat model: a parsing mistake stays invisible until a student gets a
# wrong date, and parsing only re-runs for documents that changed.
PARSER_MODEL = os.getenv("PARSER_MODEL", "anthropic/claude-sonnet-5.5")
# covers thinking plus the JSON; the longest parsed regulation is ~58k
# characters of JSON, roughly 25k tokens
PARSER_MAX_TOKENS = int(os.getenv("PARSER_MAX_TOKENS", "64000"))

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

# Chat title calls per hour (per user/IP). Only calls that reach the model
# count: a chat is titled after its first answer, and a failed title retries
# after the next answer or when the chat is opened. Twice the signed-in chat
# quota leaves room for those retries.
TITLE_LIMIT = int(os.getenv("TITLE_LIMIT", str(2 * CHAT_LIMIT_USER)))

# Spam brake: this many scope-gate refusals within 10 minutes blocks further
# messages until the window clears
CHAT_REFUSAL_LIMIT = int(os.getenv("CHAT_REFUSAL_LIMIT", "3"))

# A single abusive message blocks the sender for this long (minutes)
ABUSE_BLOCK_MINUTES = int(os.getenv("ABUSE_BLOCK_MINUTES", "10"))

# Conversation imports (sign-in adopts the browser's local chat) per hour,
# and the stored-message size past which an account can't import more
IMPORT_LIMIT = int(os.getenv("IMPORT_LIMIT", "5"))
IMPORT_MAX_MB = int(os.getenv("IMPORT_MAX_MB", "50"))

# Comma-separated emails/IPs exempt from the abuse block only (dev use);
# message limits and the spam brake still apply to these keys
ABUSE_EXEMPT = {key.strip() for key in os.getenv("ABUSE_EXEMPT", "").split(",") if key.strip()}

# Keys (IPs or signed-in emails) exempt from ALL rate limiting — for the
# judge suite and local eval runs, which burn through the anonymous
# quota by design. Don't put real users here.
RATELIMIT_EXEMPT = {key.strip() for key in os.getenv("RATELIMIT_EXEMPT", "").split(",") if key.strip()}

# Signed-in emails allowed to read /admin/usage (per-person token totals).
# Empty (the default) disables the endpoint entirely.
ADMIN_EMAILS = {key.strip() for key in os.getenv("ADMIN_EMAILS", "").split(",") if key.strip()}

# Set to 1 in production: the auth cookie is then only sent over HTTPS.
# Off by default because local dev runs plain http://localhost.
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "") == "1"

# Auth sessions expire this many days after creation. get_auth_user rejects
# older tokens and create_auth_session prunes them, so a leaked token can't
# grant access forever. Matches the auth cookie's max-age.
AUTH_SESSION_TTL_DAYS = int(os.getenv("AUTH_SESSION_TTL_DAYS", "30"))

# API (used by integration tests)
API_URL = os.getenv("API_URL", "http://localhost:8000")

SYSTEM_PROMPT_PATH = ROOT / "config" / "prompts" / "system_prompt.txt"


def load_system_prompt() -> str:
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()
