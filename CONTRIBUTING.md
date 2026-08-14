# Contributing

## Roles

- **@raccoon-42** — codeowner. Assigns issues, reviews and merges.
- **@DrHalley** — contributor. Pushes branches, opens PRs.

## Getting code in

`main` is protected: a PR needs the codeowner's approval and a green
`pytest` run, and the branch must be current with `main`.

Work on what is assigned to you; ask before picking up something else.

Before writing code, comment on the issue:

```
Approach: what you'll change, and where
Test: how you'll prove it works
Unsure about: anything you're unsure of, or "nothing"
```

`Unsure about` is the important line — a question costs a sentence now and
a rewrite later. Scale the rest to the task: a one-line fix with the
assertion already written in the issue needs little, a design change
deserves the detail.

```bash
git switch main && git pull
git switch -c fix/1-scopegate-hayir   # <type>/<issue>-<slug>
uv run pytest                         # green before you push
git push -u origin fix/1-scopegate-hayir
gh pr create --fill
```

Branches are short-lived; merged ones delete themselves. Every bug fix
ships with a test that fails before it and passes after.

## Tests

```bash
uv run pytest        # offline suite — no services, no network, no API keys
```

`pytest.ini` deselects the `integration` marker; those tests hit real
services and cost money:

```bash
uv run pytest -m integration tests/test_llm_judge.py -v
```

## Cost

The codeowner funds the contributor's OpenRouter key. Keep it in your
local `.env` (gitignored); never share or commit it.

What costs money:

- first `docker compose up` on an empty Qdrant — embeds every corpus once
- `vectorizer --recreate` — re-embeds an entire collection
- `check_updates` without `--dry-run` — runs the LLM parsing steps
