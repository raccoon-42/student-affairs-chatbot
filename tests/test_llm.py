from types import SimpleNamespace

from app.llm import OpenRouterLLM


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
