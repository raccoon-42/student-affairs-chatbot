"""The one OpenRouter call every corpus parser makes: a prompt in, a JSON
array out. It runs on PARSER_MODEL (Sonnet), not the chat model: a parsing
mistake (a wrong date, a dropped article) stays invisible until a student
gets a wrong answer, and parsing only re-runs for documents that changed."""
import json
import re
import time

import requests

from config import settings


def json_array(prompt, timeout=600):
    # transient TLS resets and timeouts happen on long uploads: retry first
    for attempt in range(3):
        try:
            response = requests.post(
                f"{settings.OPENROUTER_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {settings.OPENROUTER_API_KEY}"},
                json={
                    "model": settings.PARSER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    # without it OpenRouter applies its own cap, and the
                    # model's thinking counts against the same budget
                    "max_tokens": settings.PARSER_MAX_TOKENS,
                    "usage": {"include": True},  # OpenRouter adds the charged cost
                },
                timeout=timeout,
            )
            break
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt == 2:
                raise
            wait = 5 * (attempt + 1)
            print(f"{e.__class__.__name__}, retrying in {wait}s...")
            time.sleep(wait)
    response.raise_for_status()
    usage = response.json().get("usage") or {}
    if usage:
        print(f"  {usage.get('prompt_tokens')} in + {usage.get('completion_tokens')} out tokens"
              + (f", ${usage['cost']:.4f}" if usage.get("cost") is not None else ""))
    choice = response.json()["choices"][0]
    content = choice["message"]["content"] or ""
    if choice.get("finish_reason") == "length":
        raise ValueError(f"{settings.PARSER_MODEL} stopped at PARSER_MAX_TOKENS="
                         f"{settings.PARSER_MAX_TOKENS}: the JSON is cut off. Raise it or split the input.")
    # the model may wrap the array in a ```json fence or a sentence, or add
    # a note after it that has brackets of its own ("[2] ..."): decode the
    # first complete array of objects and ignore the rest
    decoder = json.JSONDecoder()
    for start in (match.start() for match in re.finditer(r"\[\s*[{\]]", content)):
        try:
            value, _ = decoder.raw_decode(content, start)
        except json.JSONDecodeError:
            continue
        return value
    raise ValueError(f"No JSON array in LLM response:\n{content[:500]}")
