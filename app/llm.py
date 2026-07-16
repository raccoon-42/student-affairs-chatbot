"""The LLM seam: one interface, one real adapter.

Anything that needs a language model (conversation, judge) depends on the
interface `chat(model, messages) -> str` — plus `chat_stream(model, messages)`
yielding text deltas, whose generator return value is a token-usage dict
(or None) — and never on a concrete provider. Tests pass in a fake with
the same methods.
"""
import sys

from config import settings


def chat_with_usage(llm, model, messages):
    """(text, usage) from any adapter: uses the adapter's usage-aware call
    when it has one, plain chat otherwise (fakes in tests)."""
    if hasattr(llm, "chat_with_usage"):
        return llm.chat_with_usage(model, messages)
    return llm.chat(model, messages), None


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
        return self.chat_with_usage(model, messages)[0]

    def chat_with_usage(self, model, messages):
        """(text, usage) — usage carries prompt/completion tokens and the
        charged USD, so every call kind (gate, rewrite, title, …) can be
        accounted, not just the streamed main answer."""
        self._ensure_client()
        completion = self._client.chat.completions.create(
            model=model, messages=messages, max_tokens=settings.LLM_MAX_TOKENS,
            extra_body={"usage": {"include": True}})
        if not completion.choices:
            raise RuntimeError(f"OpenRouter returned no choices: {completion}")
        usage = None
        if completion.usage:
            usage = {"prompt_tokens": completion.usage.prompt_tokens,
                     "completion_tokens": completion.usage.completion_tokens,
                     "cost": getattr(completion.usage, "cost", None)}
        return completion.choices[0].message.content, usage

    def chat_stream(self, model, messages):
        # without an explicit max_tokens OpenRouter picks its own cap, which
        # has cut answers mid-sentence on large (image) requests
        self._ensure_client()
        stream = self._client.chat.completions.create(
            model=model, messages=messages, stream=True,
            max_tokens=settings.LLM_MAX_TOKENS,
            stream_options={"include_usage": True},
            # OpenRouter extension: adds the charged cost (USD) to usage
            extra_body={"usage": {"include": True}})
        finish_reason = None
        usage = None
        for chunk in stream:
            # with include_usage the last chunk has no choices, only usage
            if getattr(chunk, "usage", None):
                usage = {"prompt_tokens": chunk.usage.prompt_tokens,
                         "completion_tokens": chunk.usage.completion_tokens,
                         "cost": getattr(chunk.usage, "cost", None)}
            if chunk.choices:
                finish_reason = chunk.choices[0].finish_reason or finish_reason
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        if finish_reason and finish_reason != "stop":
            print(f"[llm] stream ended early, finish_reason={finish_reason}", file=sys.stderr)
        return usage  # the generator's StopIteration value; None if absent
