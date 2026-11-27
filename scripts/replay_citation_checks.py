"""Replay the citation checks on logged judge failures, offline.

Each failure the judge logs since 2026-11-28 carries the model's answer
before the checks ("raw_response") and every chunk of the turn
("reference"). This runs today's checks (Conversation._check_citations)
on them and prints the failures whose answer would now differ from the
logged one, with the check's log rows: a change to app/citations.py or
app/links.py can be tried on real failures without a judge run. Whether
a changed answer would pass is still the judge's call.

    uv run python scripts/replay_citation_checks.py [--since 2026-11-28]
"""
import argparse
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.conversation import Conversation  # noqa: E402
from tests.evaluators.llm_judge import FAILURES  # noqa: E402

_ENTRY = re.compile(r"^\[(\d+)\] (.*) \[\1\]$", re.DOTALL)


def chunk_texts(entries):
    """The chunk texts in marker order, from the "[n] text [n]" entries."""
    texts = {}
    for entry in entries:
        match = _ENTRY.match(entry.strip())
        if match:
            texts[int(match.group(1))] = match.group(2)
    return [texts.get(n, "") for n in range(1, max(texts, default=0) + 1)]


def replay(raw, chunks):
    conversation = Conversation(None, None, "replay")
    conversation._chunk_texts = chunks
    conversation.last_debug = []
    conversation._log = conversation.last_debug.append  # the rows, without the stderr copy
    answer = conversation._check_citations(raw)
    return answer, [line for line in conversation.last_debug
                    if line.startswith(("citations", "markers", "links"))]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--since", default="", help="only failures logged at or after this ISO time")
    args = parser.parse_args()

    rows = [json.loads(line) for line in FAILURES.read_text(encoding="utf-8").splitlines() if line]
    usable = [row for row in rows if row.get("at", "") >= args.since
              and row.get("raw_response") and row.get("reference")]
    changed = 0
    for row in usable:
        answer, log = replay(row["raw_response"], chunk_texts(row["reference"]))
        if answer == row["response"]:
            continue
        changed += 1
        print(f"=== {row['at']}  {row['case']}")
        for line in log:
            print(f"    {line}")
        for line in difflib.unified_diff(row["response"].splitlines(), answer.splitlines(),
                                         "logged", "now", lineterm="", n=0):
            print(f"    {line}")
    print(f"{changed} of {len(usable)} replayable failures would change "
          f"({len(rows) - len(usable)} logged before raw answers and chunks were kept)")


if __name__ == "__main__":
    main()
