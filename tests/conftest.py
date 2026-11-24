import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Every test gets a fresh SQLite file; storage reads the path per call."""
    monkeypatch.setattr(settings, "DB_PATH", str(tmp_path / "test.db"))


def pytest_terminal_summary(terminalreporter):
    """A judge run's real spend, from the usage OpenRouter returns."""
    from tests.evaluators.llm_judge import FAILURES, RUN_COSTS

    if RUN_COSTS["cases"]:
        total = RUN_COSTS["judge"] + RUN_COSTS["chatbot"]
        terminalreporter.write_line(
            f"judge eval cost: ${total:.3f} for {RUN_COSTS['cases']} cases "
            f"(judge ${RUN_COSTS['judge']:.3f}, chatbot answers ${RUN_COSTS['chatbot']:.3f})")
        if RUN_COSTS["failed"]:
            terminalreporter.write_line(
                f"{RUN_COSTS['failed']} failed case(s) logged with answer and reasoning to {FAILURES}")
