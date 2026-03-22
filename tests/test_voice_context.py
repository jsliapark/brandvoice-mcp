"""Tests for the get_voice_context tool.

TODO: Add tests/test_integration.py — full MCP protocol integration tests
that spin up the server programmatically, connect with an MCP client, and
run the full flow: ingest → set_guidelines → get_voice_context → check_alignment.
"""

from __future__ import annotations

import pytest

from brandvoice_mcp.config import Config
from brandvoice_mcp.storage.chromadb import VoiceStore
from brandvoice_mcp.tools.ingest import ingest_samples
from brandvoice_mcp.tools.voice_context import get_voice_context
from tests.conftest import FakeEmbeddingService, SAMPLE_BLOG_POST


@pytest.mark.asyncio
async def test_empty_profile_returns_message(
    config: Config, store: VoiceStore, embedding_service: FakeEmbeddingService
) -> None:
    result = await get_voice_context(
        task="Write a blog post",
        platform="general",
        top_k=3,
        config=config,
        store=store,
        embeddings=embedding_service,
    )
    assert "No voice profile exists yet" in result.voice_guidelines
    assert "ingest_samples" in result.prompt_injection
    assert result.similar_samples == []


@pytest.mark.asyncio
async def test_returns_context_after_ingest(
    config: Config, store: VoiceStore, embedding_service: FakeEmbeddingService
) -> None:
    await ingest_samples(
        content=SAMPLE_BLOG_POST,
        source="blog",
        title="React Post",
        url=None,
        config=config,
        store=store,
        embeddings=embedding_service,
    )

    result = await get_voice_context(
        task="Write about React hooks",
        platform="blog",
        top_k=2,
        config=config,
        store=store,
        embeddings=embedding_service,
    )
    assert result.prompt_injection != ""
    assert len(result.similar_samples) > 0
    assert result.tone_profile is not None


@pytest.mark.asyncio
async def test_prompt_injection_contains_guidelines(
    config: Config, store: VoiceStore, embedding_service: FakeEmbeddingService
) -> None:
    from brandvoice_mcp.tools.guidelines import set_guidelines

    await set_guidelines(
        store=store,
        pillars=["practical over theoretical"],
        preferred_vocabulary=["ship", "build"],
    )

    await ingest_samples(
        content=SAMPLE_BLOG_POST,
        source="blog",
        title="Test",
        url=None,
        config=config,
        store=store,
        embeddings=embedding_service,
    )

    result = await get_voice_context(
        task="Write a post",
        platform="general",
        top_k=1,
        config=config,
        store=store,
        embeddings=embedding_service,
    )
    assert "practical over theoretical" in result.prompt_injection
    assert "ship" in result.prompt_injection
    assert "CURRENT TASK:" in result.prompt_injection
