"""Judge adapter that runs the headless `claude` CLI, so a local judge run
uses the Claude Code subscription instead of billed OpenRouter calls."""
import json
import subprocess
import tempfile

# --safe-mode keeps CLAUDE.md files, skills, hooks and MCP servers out of
# the judge's context and still uses the CLI's login (--bare would need an
# API key); no tools, and no session left behind per call
_COMMAND = ["claude", "-p", "--safe-mode", "--tools", "", "--no-session-persistence",
            "--output-format", "json"]
# replaces Claude Code's own system prompt; the judge prompt is the user turn
_SYSTEM = "Follow the user's instructions exactly."
TIMEOUT_SECONDS = 300


def cli_model(model: str) -> str:
    """OpenRouter's 'anthropic/claude-sonnet-5.5' -> the CLI's 'claude-sonnet-5-5'."""
    return model.removeprefix("anthropic/").replace(".", "-")


class ClaudeCLI:
    """Same chat interface as OpenRouterLLM, for one-turn calls."""

    def __init__(self, effort=None):
        self._effort = effort  # low, medium, high, xhigh or max

    def chat(self, model, messages):
        return self.chat_with_usage(model, messages)[0]

    def chat_with_usage(self, model, messages):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [m for m in messages if m["role"] != "system"]
        if len(turns) != 1 or turns[0]["role"] != "user":
            raise ValueError("ClaudeCLI takes exactly one user message")
        command = [*_COMMAND, "--model", cli_model(model), "--system-prompt", system or _SYSTEM]
        if self._effort:
            command += ["--effort", self._effort]
        # a neutral folder: the repo's git status would only add noise
        done = subprocess.run(command, input=turns[0]["content"], capture_output=True,
                              text=True, timeout=TIMEOUT_SECONDS, cwd=tempfile.gettempdir())
        if done.returncode != 0:
            raise RuntimeError(f"claude CLI exited {done.returncode}: {done.stderr.strip()[:500]}")

        output = json.loads(done.stdout)
        events = output if isinstance(output, list) else [output]
        result = next((e for e in reversed(events) if e.get("type") == "result"), None)
        if result is None or result.get("is_error"):
            raise RuntimeError(f"claude CLI returned no answer: {str(result)[:500]}")
        usage = result.get("usage") or {}
        prompt_tokens = sum(usage.get(key) or 0 for key in (
            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        return result["result"], {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": usage.get("output_tokens"),
            "cost": 0.0,  # covered by the subscription, not billed per call
        }
