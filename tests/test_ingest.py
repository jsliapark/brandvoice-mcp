"""Tests for the ingest_samples tool."""

from __future__ import annotations

import pytest

from brandvoice_mcp.config import Config
from brandvoice_mcp.storage.chromadb import VoiceStore
from brandvoice_mcp.tools.ingest import ingest_samples
from tests.conftest import FakeEmbeddingService, SAMPLE_BLOG_POST, SAMPLE_LINKEDIN_POST


@pytest.mark.asyncio
async def test_ingest_stores_chunks(
    config: Config, store: VoiceStore, embedding_service: FakeEmbeddingService
) -> None:
    result = await ingest_samples(
        content=SAMPLE_BLOG_POST,
        source="blog",
        title="React Components",
        url=None,
        config=config,
        store=store,
        embeddings=embedding_service,
    )
    assert result.samples_stored > 0
    assert result.total_samples == result.samples_stored
    assert result.style_snapshot.avg_sentence_length > 0


@pytest.mark.asyncio
async def test_ingest_triggers_profile_update(
    config: Config, store: VoiceStore, embedding_service: FakeEmbeddingService
) -> None:
    """Profile should update after reaching the threshold."""
    for i in range(config.profile_reanalysis_threshold):
        result = await ingest_samples(
            content=f"Sample content number {i}. " * 20,
            source="blog",
            title=f"Post {i}",
            url=None,
            config=config,
            store=store,
            embeddings=embedding_service,
        )

    assert result.voice_profile_updated is True
    assert store.get_learned_style() is not None


@pytest.mark.asyncio
async def test_ingest_different_sources(
    config: Config, store: VoiceStore, embedding_service: FakeEmbeddingService
) -> None:
    await ingest_samples(
        content=SAMPLE_BLOG_POST,
        source="blog",
        title="Blog",
        url=None,
        config=config,
        store=store,
        embeddings=embedding_service,
    )
    await ingest_samples(
        content=SAMPLE_LINKEDIN_POST,
        source="social",
        title="LinkedIn",
        url=None,
        config=config,
        store=store,
        embeddings=embedding_service,
    )
    breakdown = store.sources_breakdown()
    assert "blog" in breakdown
    assert "social" in breakdown


@pytest.mark.asyncio
async def test_ingest_short_content(
    config: Config, store: VoiceStore, embedding_service: FakeEmbeddingService
) -> None:
    """Even very short content should be stored as at least one chunk."""
    result = await ingest_samples(
        content="Short tweet-style content.",
        source="social",
        title=None,
        url=None,
        config=config,
        store=store,
        embeddings=embedding_service,
    )
    assert result.samples_stored >= 1
