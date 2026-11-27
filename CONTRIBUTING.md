# Contributing

`main` is protected: a PR needs the codeowner's (@raccoon-42) approval and
a green `pytest` run, and the branch must be current with `main`.

## Before you start

Open an issue or comment on one; wait for it to be assigned to you. Then
say how you'll do it:

```
Approach: what you'll change, and where
Test: how you'll prove it works
Unsure about: anything you're unsure of, or "nothing"
```

`Unsure about` is the important line: a question costs a sentence now and
a rewrite later.

## Getting code in

```bash
git switch main && git pull
git switch -c fix/12-gate-refusal   # <type>/<issue>-<slug>
uv run pytest                       # green before you push
git push -u origin fix/12-gate-refusal
gh pr create --fill
```

Commits follow conventional commits (`fix(chat): ...`). Every bug fix
ships with a test that fails before it and passes after.

## Tests

```bash
uv run pytest        # offline: no services, no network, no API keys
uv run pytest -m integration tests/test_llm_judge.py -v   # LLM judge, costs money
```

## Cost

You need your own OpenRouter key in `.env` (gitignored). These cost money:

- the judge eval
- the first `docker compose up` on an empty Qdrant: embeds every corpus
- `vectorizer --recreate`: re-embeds a whole collection
- `check_updates` without `--dry-run`: runs the LLM parsers
