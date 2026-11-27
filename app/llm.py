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


def _extra_body(model, effort=None):
    # OpenRouter extensions: the charged cost (USD) in usage, thinking off
    # for the models listed in REASONING_OFF_MODELS, else the adapter's effort
    body = {"usage": {"include": True}}
    if model in settings.REASONING_OFF_MODELS:
        body["reasoning"] = {"effort": "none"}
    elif effort:
        body["reasoning"] = {"effort": effort}
    return body


def _cacheable(model, messages):
    """The messages with the system prompt marked for Anthropic's prompt
    cache: ~1.9k tokens, identical on every answer call and for every user,
    so a call within 5 minutes of another reads it at 0.1x the input price
    (writing it costs 1.25x). Other models, and calls with no system
    prompt (gate, rewrite: under the 512-token minimum), go as they are."""
    if (not model.startswith("anthropic/") or not messages
            or messages[0].get("role") != "system" or not isinstance(messages[0]["content"], str)):
        return messages
    marked = {"role": "system", "content": [{"type": "text", "text": messages[0]["content"],
                                             "cache_control": {"type": "ephemeral"}}]}
    return [marked, *messages[1:]]


def _usage(usage):
    """Tokens, the charged USD (cache discounts included) and what the
    prompt cache read and wrote."""
    details = getattr(usage, "prompt_tokens_details", None)
    return {"prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost": getattr(usage, "cost", None),
            "cached_tokens": getattr(details, "cached_tokens", None) or 0,
            "cache_write_tokens": getattr(details, "cache_write_tokens", None) or 0}


class OpenRouterLLM:
    """Adapter for OpenRouter (OpenAI-compatible API)."""

    def __init__(self, api_key=None, base_url=None, reasoning_effort=None):
        self._api_key = api_key or settings.OPENROUTER_API_KEY
        self._base_url = base_url or settings.OPENROUTER_BASE_URL
        self._effort = reasoning_effort  # thinking effort for models that think
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
            model=model, messages=_cacheable(model, messages), max_tokens=settings.LLM_MAX_TOKENS,
            extra_body=_extra_body(model, self._effort))
        if not completion.choices:
            raise RuntimeError(f"OpenRouter returned no choices: {completion}")
        usage = _usage(completion.usage) if completion.usage else None
        return completion.choices[0].message.content, usage

    def chat_stream(self, model, messages):
        # without an explicit max_tokens OpenRouter picks its own cap, which
        # has cut answers mid-sentence on large (image) requests
        self._ensure_client()
        stream = self._client.chat.completions.create(
            model=model, messages=_cacheable(model, messages), stream=True,
            max_tokens=settings.LLM_MAX_TOKENS,
            stream_options={"include_usage": True},
            extra_body=_extra_body(model, self._effort))
        finish_reason = None
        usage = None
        for chunk in stream:
            # with include_usage the last chunk has no choices, only usage
            if getattr(chunk, "usage", None):
                usage = _usage(chunk.usage)
            if chunk.choices:
                finish_reason = chunk.choices[0].finish_reason or finish_reason
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        if finish_reason and finish_reason != "stop":
            print(f"[llm] stream ended early, finish_reason={finish_reason}", file=sys.stderr)
        return usage  # the generator's StopIteration value; None if absent
