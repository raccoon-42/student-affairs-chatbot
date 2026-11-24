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
    choice = response.json()["choices"][0]
    content = choice["message"]["content"] or ""
    if choice.get("finish_reason") == "length":
        raise ValueError(f"{settings.PARSER_MODEL} stopped at PARSER_MAX_TOKENS="
                         f"{settings.PARSER_MAX_TOKENS}: the JSON is cut off. Raise it or split the input.")
    # the model may wrap the array in a ```json fence or a sentence
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array in LLM response:\n{content[:500]}")
    return json.loads(match.group(0))
