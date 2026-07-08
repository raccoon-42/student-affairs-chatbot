# Student Affairs RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for university academic calendars, regulations and FAQs (Turkish). Hybrid retrieval (semantic + BM25) over Qdrant, answered by an LLM via OpenRouter or a local Ollama model, with a ChatGPT-style web UI: streaming answers with inline source citations, Google Sign-In with per-user conversation history, voice input (Whisper) and image questions.

See `CONTEXT.md` for the domain vocabulary.

## Architecture

```
app/
├── api/api.py        # FastAPI endpoints (thin shell)
├── conversation.py   # Conversation: prompt assembly + history + pipeline stages
├── llm.py            # LLM seam: OpenRouterLLM, OllamaLLM adapters
├── embeddings.py     # embedding seam: OpenRouter (default) | local sentence-transformers
├── retrieval.py      # Retriever: vector search + hybrid scoring + audience filters
│                     #   vector store seam: QdrantVectorStore | InMemoryVectorStore
├── guardrails.py     # scope gate (runs concurrently with retrieval)
├── auth.py           # Google Sign-In token verification + sessions
├── ratelimit.py      # per-user/per-IP limits, spam brake, abuse block
└── storage.py        # SQLite: users, sessions, conversations, sources
web/                  # vanilla JS chat UI (streaming, citations, i18n, settings)
config/
├── settings.py       # ALL configuration resolves here (.env read once)
└── prompts/          # system prompt
preprocessing/
├── extraction.py     # date/event extraction (shared by indexing + retrieval)
├── indexing/         # text_splitter, vectorizer, bm25
├── parsers/          # PDF -> structured JSON (LLM parsers)
└── scrapers/         # mevzuat PDFs, academic calendars, FAQ pages
tests/                # offline unit tests + `integration`-marked judge eval
scripts/              # experiments (embedding model comparison)
```

## Quick start (Docker)

```bash
cp .env.example .env   # add your OPENROUTER_API_KEY
docker compose up --build
```

This starts the API (`:8000`), Qdrant (`:6333`), and Ollama (`:11434`).

Index your documents (once Qdrant is up):

```bash
# Regulations: scrape all mevzuat PDFs, parse them, index the whole set
uv run python -m preprocessing.scrapers.mevzuat_scraper
uv run python preprocessing/parsers/regulation_parser_llm.py --all
uv run python -m preprocessing.indexing.vectorizer preprocessing/data/processed/mevzuat --type regulations

# Calendar: scrape, parse each academic year, index the whole set
uv run python -m preprocessing.scrapers.takvim_scraper
uv run python preprocessing/parsers/academic_calendar_parser_llm.py preprocessing/data/raw/takvim/2025-2026-akademik-takvimi.pdf
uv run python preprocessing/parsers/academic_calendar_parser_llm.py preprocessing/data/raw/takvim/2026-2027-akademik-takvimi.pdf
uv run python -m preprocessing.indexing.vectorizer preprocessing/data/processed/takvim --type calendar

# FAQ: scrape the university FAQ pages, then index
uv run python -m preprocessing.scrapers.faq_scraper
uv run python -m preprocessing.indexing.vectorizer preprocessing/data/processed/faq/faq.json --type faq
```

To use a local model, pull one into the Ollama container:

```bash
docker compose exec ollama ollama pull gemma3:4b
```

## Local development

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                        # create .venv from uv.lock
uv run pytest                                  # offline unit tests (no services needed)
uv run uvicorn app.api.api:app --reload        # API + web UI at http://localhost:8000
uv run python -m app.conversation --backend openrouter   # CLI chat
```

Note: `--reload` watches only `.py` files — after editing `.env`, touch a `.py` file or restart.

## API

- `POST /chat/stream` — streaming answer (body: `{query, session_id, image?, lang?}`); the stream carries in-band stage markers (`\x02` gate, `\x03` retrieval, `\x01` writing) the UI turns into cursor animations
- `GET /chat?query=...` — non-streaming answer via OpenRouter; `GET /chat_local` for Ollama
- `GET /chat/sources?session_id=...` — what the latest answer was grounded on (citation chips)
- `GET /chat/debug?session_id=...` — per-turn timing/retrieval logs (dev mode)
- `POST /transcribe` — voice input via Whisper (Groq)
- `/auth/*`, `/conversations*` — Google Sign-In and per-user history

## Evaluation

LLM-as-judge evaluation, one test per case (judge model configurable via `JUDGE_MODEL`). Needs the full stack running:

```bash
uv run pytest -m integration tests/test_llm_judge.py -v        # all cases
uv run pytest -m integration tests/test_llm_judge.py -k dersten  # one case
```

Test cases live in `tests/test_cases/*.json` as `{query, expected, description}`. Exempt your dev machine from rate limits during eval runs with `RATELIMIT_EXEMPT=127.0.0.1,::1` in `.env` (never in production).

## Configuration

Everything is an environment variable with a sane default — see `.env.example` and `config/settings.py`. The same embedding backend and collection names are used by both the indexer and the retriever, so they cannot diverge. Per-year calendar PDFs for citation links: `CALENDAR_25_26_URL`, `CALENDAR_26_27_URL`, ...

## License

CC BY-NC 4.0 — non-commercial use only. See LICENSE.
