"""The judge's claude CLI adapter, with the CLI faked."""
import json
import subprocess

import pytest

from tests.evaluators import claude_cli
from tests.evaluators.claude_cli import ClaudeCLI, cli_model

RESULT = {"type": "result", "subtype": "success", "is_error": False,
          "result": "Puan: 1\nMantık yürütme: doğru",
          "usage": {"input_tokens": 2, "cache_read_input_tokens": 600,
                    "cache_creation_input_tokens": 100, "output_tokens": 40}}


def fake_run(stdout, returncode=0, calls=None):
    def run(command, **kwargs):
        if calls is not None:
            calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="boom")
    return run


def test_model_ids_map_from_openrouter():
    assert cli_model("anthropic/claude-sonnet-5.5") == "claude-sonnet-5-5"
    assert cli_model("claude-haiku-5-5") == "claude-haiku-5-5"


def test_one_call_with_no_tools_or_personal_context(monkeypatch):
    calls = []
    # the CLI prints a list of events; the answer is the last "result"
    events = [{"type": "system", "subtype": "init"}, {"type": "assistant"}, RESULT]
    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run(json.dumps(events), calls=calls))

    text, usage = ClaudeCLI(effort="low").chat_with_usage(
        "anthropic/claude-sonnet-5.5", [{"role": "user", "content": "değerlendir"}])

    assert text == RESULT["result"]
    assert usage == {"prompt_tokens": 702, "completion_tokens": 40, "cost": 0.0}
    command, kwargs = calls[0]
    assert command[command.index("--model") + 1] == "claude-sonnet-5-5"
    assert command[command.index("--effort") + 1] == "low"
    assert command[command.index("--tools") + 1] == ""
    assert "--safe-mode" in command and "--no-session-persistence" in command
    # Claude Code's own system prompt is replaced; the prompt goes on stdin
    assert command[command.index("--system-prompt") + 1]
    assert kwargs["input"] == "değerlendir"


def test_a_system_message_becomes_the_system_prompt(monkeypatch):
    calls = []
    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run(json.dumps(RESULT), calls=calls))

    ClaudeCLI().chat("m", [{"role": "system", "content": "Sen bir hakemsin."},
                           {"role": "user", "content": "soru"}])

    command = calls[0][0]
    assert command[command.index("--system-prompt") + 1] == "Sen bir hakemsin."
    assert "--effort" not in command


@pytest.mark.parametrize("stdout, returncode", [
    ("", 1),                                                   # the CLI failed
    (json.dumps([{**RESULT, "is_error": True}]), 0),           # it answered with an error
    (json.dumps([{"type": "system"}]), 0),                     # no result at all
])
def test_failures_raise_instead_of_scoring(monkeypatch, stdout, returncode):
    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run(stdout, returncode))
    with pytest.raises(RuntimeError):
        ClaudeCLI().chat("m", [{"role": "user", "content": "soru"}])


def test_more_than_one_turn_is_refused():
    with pytest.raises(ValueError):
        ClaudeCLI().chat("m", [{"role": "user", "content": "a"},
                               {"role": "assistant", "content": "b"},
                               {"role": "user", "content": "c"}])
