"""The LLM seam: one interface, two adapters.

Anything that needs a language model (conversation, judge) depends on the
interface `chat(model, messages) -> str` — plus `chat_stream(model, messages)`
yielding text deltas — and never on a concrete provider. Tests pass in a
fake with the same methods.
"""
import json
import sys

import requests

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
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)

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


class OllamaLLM:
    """Adapter for a local Ollama server."""

    def __init__(self, url=None):
        self._url = url or settings.OLLAMA_URL

    def chat(self, model, messages):
        response = requests.post(
            f"{self._url}/api/chat",
            data=json.dumps({"model": model, "messages": messages, "stream": False}),
        )
        if response.status_code != 200:
            raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")
        return response.json()["message"]["content"]

    def chat_stream(self, model, messages):
        with requests.post(
            f"{self._url}/api/chat",
            data=json.dumps({"model": model, "messages": messages, "stream": True}),
            stream=True,
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")
            # Ollama streams one JSON object per line
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break
