"""Embedding generation via the Anthropic Voyager API.

Falls back to a simple hash-based embedding for testing when no API key
is available (test mode only).
"""

from __future__ import annotations

import hashlib
import struct
from typing import TYPE_CHECKING

import anthropic

if TYPE_CHECKING:
    from brandvoice_mcp.config import Config


# TODO: Verify this works end-to-end with voyage-3 via Anthropic's API.
# The current implementation uses the sync Anthropic client inside async methods.
# Consider switching to anthropic.AsyncAnthropic for proper async I/O,
# or run sync calls in a thread executor.
class EmbeddingService:
    """Generate text embeddings for similarity search."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)
        self._model = config.embedding_model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Uses the Voyage embedding model via Anthropic's API.
        """
        if not texts:
            return []

        response = self._client.embeddings.create(
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
