"""OpenAI embedding helper with batching and retry."""

import logging
import time

logger = logging.getLogger(__name__)

BATCH_SIZE = 100
MAX_RETRIES = 3


class EmbeddingHelper:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        # Try Langfuse-instrumented client first; fall back to plain OpenAI
        try:
            from langfuse.openai import OpenAI
        except ImportError:
            from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            embeddings = self._embed_with_retry(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(MAX_RETRIES):
            try:
                response = self._client.embeddings.create(input=texts, model=self._model)
                return [item.embedding for item in response.data]
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = 2 ** attempt
                logger.warning("Embedding request failed (attempt %d): %s. Retrying in %ds.", attempt + 1, e, wait)
                time.sleep(wait)
        raise RuntimeError("Unreachable")
