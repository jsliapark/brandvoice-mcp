"""Tests for delete_samples and related storage helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from brandvoice_mcp.config import Config
from brandvoice_mcp.storage.chromadb import VoiceStore
from brandvoice_mcp.storage.embeddings import deterministic_embedding
from brandvoice_mcp.tools.delete_samples import DeleteSamplesParams, delete_samples
from brandvoice_mcp.tools.ingest import ingest_samples
from tests.conftest import FakeEmbeddingService, SAMPLE_BLOG_POST


def test_delete_samples_params_validation() -> None:
    DeleteSamplesParams(sample_ids=["a"], all=False)
    DeleteSamplesParams(sample_ids=None, all=True)
    DeleteSamplesParams(sample_ids=[], all=True)

    with pytest.raises(ValidationError):
        DeleteSamplesParams(sample_ids=None, all=False)
    with pytest.raises(ValidationError):
        DeleteSamplesParams(sample_ids=[], all=False)
    with pytest.raises(ValidationError):
        DeleteSamplesParams(sample_ids=["x"], all=True)


@pytest.mark.asyncio
async def test_delete_samples_by_id(
    config: Config, store: VoiceStore, embedding_service: FakeEmbeddingService
) -> None:
    await ingest_samples(
        content=SAMPLE_BLOG_POST,
        source="blog",
        title="Post",
        url=None,
        config=config,
        store=store,
        embeddings=embedding_service,
    )
    await ingest_samples(
        content="Second ingest for multi-chunk coverage. " * 20,
        source="blog",
        title="Post 2",
        url=None,
        config=config,
        store=store,
        embeddings=embedding_service,
    )
    entries, total_before = store.list_samples(limit=100, offset=0)
    assert total_before >= 2
    victim = entries[0]["id"]

    result = await delete_samples(
        sample_ids=[victim],
        delete_all=False,
        config=config,
        store=store,
    )
    assert result.deleted_count == 1
    assert result.remaining_count == total_before - 1
    assert store.total_samples == total_before - 1


@pytest.mark.asyncio
async def test_delete_samples_nonexistent_id_graceful(
    config: Config, store: VoiceStore, embedding_service: FakeEmbeddingService
) -> None:
    await ingest_samples(
        content="Some content here for testing deletion. " * 15,
        source="blog",
        title="X",
        url=None,
        config=config,
        store=store,
        embeddings=embedding_service,
    )
    n = store.total_samples
    result = await delete_samples(
        sample_ids=["00000000-0000-0000-0000-000000000000"],
        delete_all=False,
        config=config,
        store=store,
    )
    assert result.deleted_count == 0
    assert result.remaining_count == n


@pytest.mark.asyncio
async def test_delete_all_clears_samples_and_resets_profile(
    config: Config, store: VoiceStore, embedding_service: FakeEmbeddingService
) -> None:
    await ingest_samples(
        content=SAMPLE_BLOG_POST,
        source="blog",
        title="Post",
        url=None,
        config=config,
        store=store,
        embeddings=embedding_service,
    )
    store.save_learned_style({"dominant_tone": "authoritative", "profile_source": "llm"})
    store.save_guidelines({"pillars": ["keep me"]})

    n = store.total_samples
    result = await delete_samples(
        sample_ids=None,
        delete_all=True,
        config=config,
        store=store,
    )
    assert result.deleted_count == n
    assert result.remaining_count == 0
    assert store.total_samples == 0
    assert store.get_learned_style() is None
    assert store.get_guidelines() is None


@pytest.mark.asyncio
async def test_delete_last_sample_resets_profile(
    config: Config, store: VoiceStore, embedding_service: FakeEmbeddingService
) -> None:
    await ingest_samples(
        content="Only one chunk worth of text here. " * 10,
        source="blog",
        title="Solo",
        url=None,
        config=config,
        store=store,
        embeddings=embedding_service,
    )
    assert store.total_samples == 1
    entries, _ = store.list_samples(limit=1, offset=0)
    only_id = entries[0]["id"]
    store.save_learned_style({"dominant_tone": "playful", "profile_source": "heuristic"})

    result = await delete_samples(
        sample_ids=[only_id],
        delete_all=False,
        config=config,
        store=store,
    )
    assert result.deleted_count == 1
    assert result.remaining_count == 0
    assert store.get_learned_style() is None


def test_voice_store_delete_by_ids_and_delete_all(store: VoiceStore) -> None:
    ids = store.add_samples(
        chunks=["a", "b"],
        embeddings=[deterministic_embedding("a"), deterministic_embedding("b")],
        metadata={"source": "blog"},
    )
    deleted, remaining = store.delete_samples_by_ids([ids[0], "nonexistent-uuid"])
    assert deleted == 1
    assert remaining == 1

    n = store.delete_all_writing_samples()
    assert n == 1
    assert store.total_samples == 0
