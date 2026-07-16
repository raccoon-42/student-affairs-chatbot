# Student Affairs RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for university life (Turkish): academic calendars, regulations, FAQs, forms, campus services, degree programs, department people, courses and student guides — nine Qdrant collections kept fresh by a daily update check. Hybrid retrieval (semantic + BM25) over Qdrant, answered by an LLM via OpenRouter, with a ChatGPT-style web UI: streaming answers with inline source citations, Google Sign-In with per-user conversation history, voice input (Whisper) and image questions.

See `CONTEXT.md` for the domain vocabulary.

## Architecture

```
app/
├── api/api.py        # FastAPI endpoints (thin shell)
├── conversation.py   # Conversation: prompt assembly + history + pipeline stages
├── llm.py            # LLM seam: OpenRouterLLM adapter
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
├── check_updates.py  # daily change check: re-scrape, hash, re-parse/reindex only what changed
├── extraction.py     # date/event extraction (shared by indexing + retrieval)
├── indexing/         # text_splitter, vectorizer, bm25
├── parsers/          # PDF/HTML -> structured JSON (LLM parsers)
└── scrapers/         # mevzuat, takvim, faq, forms, sks, programs, people, courses, guides
tests/                # offline unit tests + `integration`-marked judge eval
scripts/              # experiments (embedding model comparison)
```

## Quick start (Docker)

```bash
cp .env.example .env   # add your OPENROUTER_API_KEY
docker compose up --build
```

This starts three services:

- **api** (`:8000`) — FastAPI + the web UI
- **qdrant** (`:6333`) — vector store, data in the `qdrant_storage` named volume
- **check-updates** — daily corpus check at 06:30 UTC (09:30 TR), see below

On a fresh clone Qdrant starts empty, and `check-updates` handles that by
itself: with no recorded index state it runs a baseline check on startup,
indexing all nine collections (calendar, regulations, faq, forms, sks,
programs, people, courses, guides) from the git-tracked processed corpora
and re-downloading the gitignored raw PDFs. Watch it with
`docker compose logs -f check-updates` — every corpus should end with
"recording current files as the baseline". Then open http://localhost:8000
and ask something.

To force a check without waiting for the daily run:

```bash
docker compose run --rm check-updates python -m preprocessing.check_updates
```

What persists across rebuilds: the Qdrant index and embedding-model cache
(named volumes `qdrant_storage`, `hf_cache`), SQLite users/conversations
(`./data`, bind-mounted), and corpora + index state (`./preprocessing/data`,
bind-mounted, shared with manual pipeline runs on the host).

For production (VPS, nginx, TLS, backups) see `deploy/README.md` and
`docker-compose.prod.yml`.

## Keeping the corpora fresh

The `check-updates` service re-runs `preprocessing.check_updates` daily.
It re-scrapes every source (cheap), hashes the artifacts, and runs the
expensive steps — LLM parsing, embedding — only for what actually changed;
state lives in `preprocessing/data/index_state.json`. Run it manually with:

```bash
uv run python -m preprocessing.check_updates             # all corpora
uv run python -m preprocessing.check_updates faq sks     # a subset
uv run python -m preprocessing.check_updates --dry-run   # scrape + report, change nothing
```

## Rebuilding a corpus from scratch

Each collection has a scraper in `preprocessing/scrapers/`, most have an
LLM parser in `preprocessing/parsers/`, and everything is indexed by the
vectorizer (`--type` selects the collection). The pattern, shown for three
corpora:

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

# FAQ: scrape the university FAQ pages, then index (no LLM step)
uv run python -m preprocessing.scrapers.faq_scraper
uv run python -m preprocessing.indexing.vectorizer preprocessing/data/processed/faq/faq.json --type faq
```

The remaining corpora (forms, sks, programs, people, courses, guides)
follow the same scrape → parse → index shape; `preprocessing/check_updates.py`
is the authoritative reference for each one's exact steps. The vectorizer
is incremental (point id = content hash), so reindexing embeds only
changed chunks.

## Local development

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync                                        # create .venv from uv.lock
uv run pytest                                  # offline unit tests (no services needed)
uv run uvicorn app.api.api:app --reload        # API + web UI at http://localhost:8000
uv run python -m app.conversation              # CLI chat
```

Note: `--reload` watches only `.py` files — after editing `.env`, touch a `.py` file or restart.

## API

- `POST /chat/stream` — streaming answer (body: `{query, session_id, image?, lang?}`); the stream carries in-band stage markers (`\x02` gate, `\x03` retrieval, `\x01` writing) the UI turns into cursor animations
- `GET /chat?query=...` — non-streaming answer via OpenRouter
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

Everything is an environment variable with a sane default — see `.env.example` and `config/settings.py`. The same embedding backend and collection names are used by both the indexer and the retriever, so they cannot diverge. Citation links come from the scrape manifests (each chunk carries its source PDF/page URL), so no per-document URLs need configuring.

## License

CC BY-NC 4.0 — non-commercial use only. See LICENSE.
