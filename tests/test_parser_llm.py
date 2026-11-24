"""The parsers' shared OpenRouter call, with the HTTP layer faked."""
import pytest

from config import settings
from preprocessing.parsers import llm_call


class FakeResponse:
    def __init__(self, content, finish_reason="stop"):
        self._body = {"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def test_parsers_run_on_the_parser_model_with_a_token_cap(monkeypatch):
    sent = {}

    def post(url, headers, json, timeout):
        sent.update(json)
        return FakeResponse('```json\n[{"madde": 1}]\n```')

    monkeypatch.setattr(llm_call.requests, "post", post)
    assert llm_call.json_array("prompt") == [{"madde": 1}]
    assert sent["model"] == settings.PARSER_MODEL != settings.OPENROUTER_MODEL
    assert sent["max_tokens"] == settings.PARSER_MAX_TOKENS
    assert sent["usage"] == {"include": True}  # every run reports what it cost


def test_output_cut_off_at_the_cap_fails_loudly(monkeypatch):
    # a truncated array must not reach json.loads as if it were the answer
    monkeypatch.setattr(llm_call.requests, "post",
                        lambda *a, **k: FakeResponse('[{"madde": 1}, {"madde": 2, "metin": "(1) Öğr', "length"))
    with pytest.raises(ValueError, match="PARSER_MAX_TOKENS"):
        llm_call.json_array("prompt")


def test_text_after_the_array_is_ignored(monkeypatch):
    # a note after the array, with brackets of its own, broke the greedy match
    content = 'Parçalar [1]-[2]:\n[{"parca": 1, "sorular": ["ne zaman"]}]\n\nNot: [2] için soru [yok].'
    monkeypatch.setattr(llm_call.requests, "post", lambda *a, **k: FakeResponse(content))
    assert llm_call.json_array("prompt") == [{"parca": 1, "sorular": ["ne zaman"]}]
