"""The LLM seam: one interface, two adapters.

Anything that needs a language model (conversation, judge) depends on the
interface `chat(model, messages) -> str` and never on a concrete provider.
Tests pass in a fake with the same method.
"""
import json

import requests

from config import settings


class OpenRouterLLM:
    """Adapter for OpenRouter (OpenAI-compatible API)."""

    def __init__(self, api_key=None, base_url=None):
        self._api_key = api_key or settings.OPENROUTER_API_KEY
        self._base_url = base_url or settings.OPENROUTER_BASE_URL
        self._client = None  # created on first use, not at import

    def chat(self, model, messages):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)

        completion = self._client.chat.completions.create(model=model, messages=messages)
        if not completion.choices:
            raise RuntimeError(f"OpenRouter returned no choices: {completion}")
        return completion.choices[0].message.content


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
