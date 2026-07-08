"""The embedding seam: one interface, two adapters.

Indexing and querying both go through `embed_documents(texts)` /
`embed_query(text)`, so the two sides can never disagree on model or
prefix. `default_embedder()` picks the adapter from settings:
EMBEDDING_BACKEND=openrouter (hosted, the default) or local
(sentence-transformers).
"""
import time
from typing import List

import requests

from config import settings


class SentenceTransformerEmbedder:
    """Local inference. The e5-instruct model wants an instruction prefix
    on queries (not on documents)."""

    def __init__(self, model_name=None):
        self._model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None  # loaded on first use — importing this module stays cheap

    def _encode(self, texts):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name, trust_remote_code=True)
        return self._model.encode(texts)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vector.tolist() for vector in self._encode(texts)]

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"Instruct: {settings.EMBED_INSTRUCTION}\nQuery: {text}"
        return self._encode(prefixed).tolist()


class OpenRouterEmbedder:
    """Hosted embeddings via OpenRouter's OpenAI-compatible endpoint.

    Gemini/OpenAI-style models take the query as-is — no instruction
    prefix. (Instruct-style models like e5/qwen3 would need one; add it
    here if the configured model ever changes to one of those.)
    """

    def __init__(self, model=None, api_key=None, base_url=None):
        self._model = model or settings.OPENROUTER_EMBEDDING_MODEL
        self._api_key = api_key or settings.OPENROUTER_API_KEY
        self._base_url = base_url or settings.OPENROUTER_BASE_URL

    def embed_documents(self, texts: List[str], batch_size=32) -> List[List[float]]:
        vectors = []
        for i in range(0, len(texts), batch_size):
            vectors.extend(self._request(texts[i:i + batch_size]))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self._request([text])[0]

    def _request(self, batch, attempts=3):
        for attempt in range(attempts):
            try:
                response = requests.post(
                    f"{self._base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"model": self._model, "input": batch},
                    timeout=30,
                )
            except requests.RequestException:
                if attempt == attempts - 1:
                    raise
                continue
            if response.status_code == 200:
                data = sorted(response.json()["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in data]
            if response.status_code in (429, 500, 502, 503) and attempt < attempts - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{self._model} error {response.status_code}: {response.text[:300]}")


def default_embedder():
    if settings.EMBEDDING_BACKEND == "local":
        return SentenceTransformerEmbedder()
    return OpenRouterEmbedder()
