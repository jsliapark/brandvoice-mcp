"""Tests for the check_alignment tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brandvoice_mcp.config import Config
from brandvoice_mcp.storage.chromadb import VoiceStore
from brandvoice_mcp.tools import alignment as alignment_mod
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


@pytest.mark.asyncio
async def test_llm_alignment_parses_response(tmp_data_dir) -> None:
    cfg = Config(
        data_dir=tmp_data_dir,
        anthropic_api_key="sk-test",
        embedding_model="voyage-3",
        analysis_model="claude-sonnet-4-20250514",
        profile_reanalysis_threshold=3,
        chunk_target_tokens=350,
        chunk_min_tokens=50,
        chunk_max_tokens=600,
    )
    cfg.ensure_directories()
    store = VoiceStore(cfg)
    store.save_learned_style(
        {
            "avg_sentence_length": 14.0,
            "vocabulary_richness": 0.6,
            "formality_score": 0.5,
            "dominant_tone": "conversational",
            "rhetorical_patterns": [],
        }
    )

    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.text = (
        '{"alignment_score": 82, "verdict": "on_brand", "drift_flags": [], '
        '"suggestions": ["Keep the hook"], "rewrite_hints": "Nice match."}'
    )
    mock_response.content = [mock_block]

    with patch.object(alignment_mod, "anthropic") as mock_pkg:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_pkg.AsyncAnthropic.return_value = mock_client

        result = await check_alignment(
            content="A short draft that sounds like me.",
            platform="linkedin",
            config=cfg,
            store=store,
        )

    assert result.alignment_score == 82
    assert result.verdict == "on_brand"
    assert result.rewrite_hints == "Nice match."
    mock_client.messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_llm_alignment_falls_back_on_error(
    config: Config, store: VoiceStore
) -> None:
    cfg = Config(
        data_dir=config.data_dir,
        anthropic_api_key="sk-test",
        embedding_model="voyage-3",
        analysis_model="claude-sonnet-4-20250514",
        profile_reanalysis_threshold=config.profile_reanalysis_threshold,
        chunk_target_tokens=config.chunk_target_tokens,
        chunk_min_tokens=config.chunk_min_tokens,
        chunk_max_tokens=config.chunk_max_tokens,
    )
    store.save_learned_style(
        {
            "avg_sentence_length": 12.0,
            "vocabulary_richness": 0.7,
            "formality_score": 0.45,
            "dominant_tone": "conversational",
            "rhetorical_patterns": [],
        }
    )

    with patch.object(alignment_mod, "anthropic") as mock_pkg:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=RuntimeError("rate limit"))
        mock_pkg.AsyncAnthropic.return_value = mock_client

        result = await check_alignment(
            content="Building components is fun. Each piece handles logic.",
            platform="general",
            config=cfg,
            store=store,
        )

    assert result.verdict in ("on_brand", "minor_drift", "significant_drift", "off_brand")
    assert 0 <= result.alignment_score <= 100
