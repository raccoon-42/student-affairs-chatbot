# Student Affairs RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for university academic calendars and regulations (Turkish). Hybrid retrieval (semantic + BM25) over Qdrant, answered by an LLM via OpenRouter or a local Ollama model.

See `CONTEXT.md` for the domain vocabulary and `docs/adr/` for the key design decisions.

## Architecture

```
app/
├── api/api.py        # FastAPI endpoints (thin shell)
├── conversation.py   # Conversation: prompt assembly + history
├── llm.py            # LLM seam: OpenRouterLLM, OllamaLLM adapters
└── retrieval.py      # Retriever: embedding + vector search + hybrid scoring
                      #   vector store seam: QdrantVectorStore | InMemoryVectorStore
config/
├── settings.py       # ALL configuration resolves here (.env read once)
└── prompts/          # system prompt
preprocessing/
├── extraction.py     # date/event extraction (shared by indexing + retrieval)
├── indexing/         # text_splitter, vectorizer, bm25
├── parsers/          # PDF -> text
└── scrapers/         # university FAQ sources -> faq.json
tests/                # offline unit tests + one `integration`-marked E2E eval
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
uv run uvicorn app.api.api:app --reload        # API against local Qdrant/Ollama
uv run python -m app.conversation --backend openrouter   # CLI chat
```

## API

- `GET /chat?query=...&model_name=...` — answer via OpenRouter (model optional)
- `GET /chat_local?query=...&model_name=...` — answer via Ollama (model optional)

## Evaluation

LLM-as-judge evaluation (judge runs on OpenRouter by default, configurable via `JUDGE_MODEL`). Needs the full stack running:

```bash
uv run pytest -m integration tests/test_llm_judge.py
```

Test cases live in `tests/test_cases/*.json` as `{query, expected, description}`.

## Configuration

Everything is an environment variable with a sane default — see `.env.example` and `config/settings.py`. The same `EMBEDDING_MODEL` and collection names are used by both the indexer and the retriever, so they cannot diverge.

## License

CC BY-NC 4.0 — non-commercial use only. See LICENSE.
