"""Embedding generation via the OpenAI Embeddings API.

Anthropic does not expose a text-embeddings endpoint in the public SDK; we use
OpenAI (default: ``text-embedding-3-small``) for chunk vectors used in ChromaDB.

When ``embedding_model`` is ``\"test\"``, uses deterministic hash vectors (tests only).
"""

from __future__ import annotations

import hashlib
import struct
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from brandvoice_mcp.config import Config


class EmbeddingService:
    """Generate text embeddings for similarity search."""

    def __init__(self, config: Config) -> None:
        self._model = config.embedding_model
        if self._model == "test":
            self._client = None
        else:
            key = config.openai_api_key
            if not key:
                raise EnvironmentError(
                    "OPENAI_API_KEY is required when embedding_model is not 'test'."
                )
            self._client = AsyncOpenAI(api_key=key)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        if self._model == "test":
            return [deterministic_embedding(t) for t in texts]

        assert self._client is not None
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text."""
        results = await self.embed_texts([text])
        return results[0]


def deterministic_embedding(text: str, dimensions: int = 256) -> list[float]:
    """Hash-based pseudo-embedding for testing without API calls.

    Produces a deterministic, fixed-dimension vector from the text's SHA-256
    digest. NOT useful for real semantic similarity — only for testing that
    the storage pipeline works end-to-end.
    """
    digest = hashlib.sha256(text.encode()).digest()
    floats_needed = dimensions
    repeated = digest * ((floats_needed * 4 // len(digest)) + 1)
    values = struct.unpack(f"<{floats_needed}f", repeated[: floats_needed * 4])
    # Normalise to [-1, 1]
    max_abs = max(abs(v) for v in values) or 1.0
    return [v / max_abs for v in values]
