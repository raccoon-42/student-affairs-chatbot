<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/logo-dark.svg">
  <img alt="İyteBot logo" src=".github/assets/logo-light.svg" width="72">
</picture>

# İyteBot

[![tests](https://github.com/raccoon-42/student-affairs-chatbot/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/raccoon-42/student-affairs-chatbot/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%2B-3776AB)
[![license](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)](LICENSE)

A Turkish chatbot for İYTE students. It answers from the university's own
documents — academic calendar, regulations, FAQs, forms, campus services,
programs, people, courses and student guides — and cites a source for every
fact.

FastAPI, Qdrant and a vanilla JS web UI. Models run through OpenRouter:
Claude Haiku 5.5 writes the answers, Sonnet 5.5 parses documents and judges
the eval.

![A student asks about Erasmus requirements and gets a cited answer](.github/assets/demo.gif)

## Answering a question

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/answer-dark.svg">
  <img alt="Rewrite, scope gate and hybrid search start together; routing drops far-off collections; the prompt carries [n]-marked chunks; the answer model streams; a citation check fixes markers; the answer shows source chips." src=".github/assets/answer-light.svg">
</picture>

- The gate and the search start on the student's own words while the
  rewrite runs; they run again only if the rewrite changes the words.
- Search is dense + BM25 in every collection. A collection whose best match
  is 0.20 below the top one is left out.
- Every chunk carries a `[n]` marker. After the answer, a rule-based check
  drops markers the chunk doesn't back, moves misplaced ones and adds
  missing ones.

## Keeping the data fresh

<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/data-dark.svg">
  <img alt="A daily scrape of the university sites; only changed sources are parsed, embedded and written to Qdrant." src=".github/assets/data-light.svg">
</picture>

The `check-updates` service runs this every day at 06:30 UTC. Only changed
sources are parsed and embedded again.

## Run

```bash
cp .env.example .env        # add OPENROUTER_API_KEY
docker compose up --build   # http://localhost:8000
```

On a fresh clone, `check-updates` indexes everything on first start:
`docker compose logs -f check-updates`. Production setup: `deploy/README.md`.

## Develop

```bash
uv sync
uv run pytest                                            # offline tests
uv run uvicorn app.api.api:app --reload                  # http://localhost:8000
uv run python -m preprocessing.check_updates --dry-run   # what changed upstream
```

The LLM judge grades answers against `tests/test_cases/*.json`. It needs
the stack running and costs money:

```bash
uv run pytest -m integration tests/test_llm_judge.py -v
```

## More

- `CONTEXT.md`: domain vocabulary
- `.env.example`, `config/settings.py`: configuration
- `CONTRIBUTING.md`: how changes get in

## License

CC BY-NC 4.0, non-commercial use only. See `LICENSE`.
