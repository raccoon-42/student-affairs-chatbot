"""The LLM seam: one interface, one real adapter.

Anything that needs a language model (conversation, judge) depends on the
interface `chat(model, messages) -> str` — plus `chat_stream(model, messages)`
yielding text deltas — and never on a concrete provider. Tests pass in a
fake with the same methods.
"""
import sys

from config import settings


class OpenRouterLLM:
    """Adapter for OpenRouter (OpenAI-compatible API)."""

    def __init__(self, api_key=None, base_url=None):
        self._api_key = api_key or settings.OPENROUTER_API_KEY
        self._base_url = base_url or settings.OPENROUTER_BASE_URL
        self._client = None  # created on first use, not at import

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            # the SDK default is 600s x 3 attempts — a stalled connection
            # froze a judge run for many minutes before this cap
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key,
                                  timeout=settings.LLM_TIMEOUT_SECONDS)

    def chat(self, model, messages):
        self._ensure_client()
        completion = self._client.chat.completions.create(
            model=model, messages=messages, max_tokens=settings.LLM_MAX_TOKENS)
        if not completion.choices:
            raise RuntimeError(f"OpenRouter returned no choices: {completion}")
        return completion.choices[0].message.content

    def chat_stream(self, model, messages):
        # without an explicit max_tokens OpenRouter picks its own cap, which
        # has cut answers mid-sentence on large (image) requests
        self._ensure_client()
        stream = self._client.chat.completions.create(
            model=model, messages=messages, stream=True,
            max_tokens=settings.LLM_MAX_TOKENS)
        finish_reason = None
        for chunk in stream:
            if chunk.choices:
                finish_reason = chunk.choices[0].finish_reason or finish_reason
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        if finish_reason and finish_reason != "stop":
            print(f"[llm] stream ended early, finish_reason={finish_reason}", file=sys.stderr)
