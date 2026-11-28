from types import SimpleNamespace

from app.llm import OpenRouterLLM, _cacheable, _usage, chat_with_usage
from config import settings


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="cevap")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


def make_llm():
    llm = OpenRouterLLM(api_key="test")
    completions = FakeCompletions()
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm, completions


def test_haiku_runs_with_thinking_off():
    # Haiku 5.5 thinks by default: first visible text went from ~3s to 6-11s
    llm, completions = make_llm()
    llm.chat("anthropic/claude-haiku-5.5", [{"role": "user", "content": "soru"}])
    assert completions.kwargs["extra_body"]["reasoning"] == {"effort": "none"}
    assert completions.kwargs["extra_body"]["usage"] == {"include": True}


def test_sonnet_keeps_its_default_thinking():
    # Sonnet 5.5 rejects thinking-off with a 400
    llm, completions = make_llm()
    llm.chat("anthropic/claude-sonnet-5.5", [{"role": "user", "content": "soru"}])
    assert "reasoning" not in completions.kwargs["extra_body"]


def test_an_adapter_effort_applies_to_models_that_think():
    # the judge asks Sonnet for low effort; Haiku stays at thinking off
    llm, completions = make_llm()
    llm._effort = "low"
    llm.chat("anthropic/claude-sonnet-5.5", [{"role": "user", "content": "soru"}])
    assert completions.kwargs["extra_body"]["reasoning"] == {"effort": "low"}
    llm.chat("anthropic/claude-haiku-5.5", [{"role": "user", "content": "soru"}])
    assert completions.kwargs["extra_body"]["reasoning"] == {"effort": "none"}


# --- prompt caching: what gets marked and what the usage reports

SYSTEM = {"role": "system", "content": "Sen İyteBot'sun."}
QUESTION = {"role": "user", "content": "sınavlar ne zaman"}


def test_the_system_prompt_is_marked_for_claude_models():
    sent = _cacheable("anthropic/claude-haiku-5.5", [SYSTEM, QUESTION])
    assert sent[0] == {"role": "system", "content": [
        {"type": "text", "text": "Sen İyteBot'sun.", "cache_control": {"type": "ephemeral"}}]}
    assert sent[1] == QUESTION
    assert SYSTEM["content"] == "Sen İyteBot'sun."  # the conversation's own copy stays a string


def test_other_models_and_calls_without_a_system_prompt_go_as_they_are():
    assert _cacheable("openai/gpt-5", [SYSTEM, QUESTION]) == [SYSTEM, QUESTION]
    assert _cacheable("anthropic/claude-haiku-5.5", [QUESTION]) == [QUESTION]  # gate, rewrite


def test_usage_reports_cache_reads_and_writes():
    details = SimpleNamespace(cached_tokens=1900, cache_write_tokens=0)
    usage = SimpleNamespace(prompt_tokens=9000, completion_tokens=300, cost=0.0012,
                            prompt_tokens_details=details)
    assert _usage(usage) == {"prompt_tokens": 9000, "completion_tokens": 300, "cost": 0.0012,
                             "cached_tokens": 1900, "cache_write_tokens": 0}
    bare = SimpleNamespace(prompt_tokens=10, completion_tokens=2)  # a provider without details
    assert _usage(bare) == {"prompt_tokens": 10, "completion_tokens": 2, "cost": None,
                            "cached_tokens": 0, "cache_write_tokens": 0}


def test_the_adapter_sends_the_marked_system_prompt():
    llm, completions = make_llm()
    llm.chat("anthropic/claude-haiku-5.5", [SYSTEM, QUESTION])
    assert completions.kwargs["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_a_short_call_can_lower_the_reply_cap():
    # titles: a model talked into an essay stops at the call's own cap
    llm, completions = make_llm()
    chat_with_usage(llm, "anthropic/claude-haiku-5.5", [QUESTION], max_tokens=64)
    assert completions.kwargs["max_tokens"] == 64
    chat_with_usage(llm, "anthropic/claude-haiku-5.5", [QUESTION])
    assert completions.kwargs["max_tokens"] == settings.LLM_MAX_TOKENS
