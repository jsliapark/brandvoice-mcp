"""Tests for the check_alignment tool."""

from __future__ import annotations

import pytest

from brandvoice_mcp.config import Config
from brandvoice_mcp.storage.chromadb import VoiceStore
from brandvoice_mcp.tools.alignment import check_alignment


@pytest.mark.asyncio
async def test_no_profile_returns_unknown(config: Config, store: VoiceStore) -> None:
    result = await check_alignment(
        content="Some content to check.",
        platform="general",
        config=config,
        store=store,
    )
    assert result.alignment_score == 50
    assert result.verdict == "unknown"
    assert len(result.drift_flags) > 0


@pytest.mark.asyncio
async def test_matching_content_scores_high(config: Config, store: VoiceStore) -> None:
    store.save_learned_style(
        {
            "avg_sentence_length": 12.0,
            "vocabulary_richness": 0.7,
            "formality_score": 0.45,
            "dominant_tone": "conversational",
            "rhetorical_patterns": [],
        }
    )

    # Content with ~12-word sentences, moderate formality
    content = (
        "Building components is fun. Each piece handles its own logic. "
        "Testing becomes easier this way. You ship faster with composition."
    )
    result = await check_alignment(
        content=content,
        platform="general",
        config=config,
        store=store,
    )
    assert result.alignment_score >= 60
    assert result.verdict in ("on_brand", "minor_drift")


@pytest.mark.asyncio
async def test_avoided_vocabulary_flags(config: Config, store: VoiceStore) -> None:
    store.save_guidelines(
        {
            "avoided_vocabulary": ["synergy", "leverage"],
        }
    )

    content = "We need to leverage synergy across teams to drive paradigm shifts."
    result = await check_alignment(
        content=content,
        platform="general",
        config=config,
        store=store,
    )
    vocab_flags = [f for f in result.drift_flags if f.category == "vocabulary"]
    assert len(vocab_flags) > 0
    assert "synergy" in vocab_flags[0].issue or "leverage" in vocab_flags[0].issue
