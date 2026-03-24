"""Tests for style analysis (LLM + heuristic)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brandvoice_mcp.analysis import style_analyzer
from brandvoice_mcp.analysis.style_analyzer import (
    aggregate_style_from_corpus,
    analyze_style,
    heuristic_style_snapshot,
    _normalize_snapshot,
)
from brandvoice_mcp.llm_json import extract_json_object
from brandvoice_mcp.config import Config


def test_extract_json_object_strips_fence() -> None:
    raw = '```json\n{"avg_sentence_length": 10.0, "vocabulary_richness": 0.5, "formality_score": 0.4, "dominant_tone": "conversational", "rhetorical_patterns": []}\n```'
    data = extract_json_object(raw)
    assert data["avg_sentence_length"] == 10.0


def test_normalize_snapshot_clamps_tone() -> None:
    snap = _normalize_snapshot(
        {
            "avg_sentence_length": 12.5,
            "vocabulary_richness": 1.5,
            "formality_score": -0.1,
            "dominant_tone": "weird_unknown",
            "rhetorical_patterns": ["a", "b", "c", "d", "e", "f"],
        }
    )
    assert snap.dominant_tone == "conversational"
    assert snap.vocabulary_richness == 1.0
    assert snap.formality_score == 0.0
    assert len(snap.rhetorical_patterns) == 5
    assert snap.profile_source == "llm"


@pytest.mark.asyncio
async def test_analyze_style_uses_heuristic_when_model_is_test(
    tmp_data_dir,
) -> None:
    cfg = Config(
        data_dir=tmp_data_dir,
        anthropic_api_key="x",
        embedding_model="test",
        analysis_model="test",
        profile_reanalysis_threshold=3,
        chunk_target_tokens=350,
        chunk_min_tokens=50,
        chunk_max_tokens=600,
    )
    result = await analyze_style("Hello world. Second sentence here.", cfg)
    assert result.avg_sentence_length > 0
    assert result.dominant_tone == "conversational"
    assert result.profile_source == "heuristic"


@pytest.mark.asyncio
async def test_analyze_style_llm_path_parses_response(tmp_data_dir) -> None:
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

    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.text = (
        '{"avg_sentence_length": 14.2, "vocabulary_richness": 0.72, '
        '"formality_score": 0.45, "humor": 0.2, "technical_depth": 0.7, '
        '"warmth": 0.55, "dominant_tone": "professional", '
        '"rhetorical_patterns": ["uses lists"]}'
    )
    mock_response.content = [mock_block]

    with patch.object(style_analyzer, "anthropic") as mock_anthropic_pkg:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic_pkg.AsyncAnthropic.return_value = mock_client

        result = await analyze_style("Some sample prose for analysis.", cfg)

    assert result.avg_sentence_length == 14.2
    assert result.dominant_tone == "professional"
    assert result.humor == 0.2
    assert result.technical_depth == 0.7
    assert result.warmth == 0.55
    assert "uses lists" in result.rhetorical_patterns
    assert result.profile_source == "llm"
    mock_client.messages.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_analyze_style_falls_back_on_api_error(tmp_data_dir) -> None:
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

    with patch.object(style_analyzer, "anthropic") as mock_anthropic_pkg:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=RuntimeError("API down"))
        mock_anthropic_pkg.AsyncAnthropic.return_value = mock_client

        result = await analyze_style(
            "Short text. Another sentence. " * 5,
            cfg,
        )

    assert result.dominant_tone == "conversational"
    assert result.avg_sentence_length > 0
    assert result.profile_source == "heuristic"


def test_heuristic_nonempty() -> None:
    snap = heuristic_style_snapshot(
        "Furthermore, we must leverage synergy. However, it is casual lol."
    )
    assert snap.formality_score >= 0.0
    assert snap.profile_source == "heuristic"


def test_normalize_snapshot_defaults_tone_dimensions() -> None:
    snap = _normalize_snapshot(
        {
            "avg_sentence_length": 10.0,
            "vocabulary_richness": 0.5,
            "formality_score": 0.5,
            "dominant_tone": "professional",
            "rhetorical_patterns": [],
        }
    )
    assert snap.humor == 0.5
    assert snap.technical_depth == 0.5
    assert snap.warmth == 0.5


@pytest.mark.asyncio
async def test_aggregate_style_from_corpus_test_mode(tmp_data_dir) -> None:
    cfg = Config(
        data_dir=tmp_data_dir,
        anthropic_api_key="x",
        embedding_model="test",
        analysis_model="test",
        profile_reanalysis_threshold=3,
        chunk_target_tokens=350,
        chunk_min_tokens=50,
        chunk_max_tokens=600,
    )
    merged = await aggregate_style_from_corpus(
        "We shipped the API. You will love how fast it is. lol",
        cfg,
    )
    assert merged.profile_source == "heuristic"
    assert merged.avg_sentence_length > 0


@pytest.mark.asyncio
async def test_aggregate_style_from_corpus_llm_path(tmp_data_dir) -> None:
    cfg = Config(
        data_dir=tmp_data_dir,
        anthropic_api_key="sk-test",
        embedding_model="test",
        analysis_model="claude-sonnet-4-20250514",
        profile_reanalysis_threshold=3,
        chunk_target_tokens=350,
        chunk_min_tokens=50,
        chunk_max_tokens=600,
    )
    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.text = (
        '{"avg_sentence_length": 11.0, "vocabulary_richness": 0.6, '
        '"formality_score": 0.5, "humor": 0.3, "technical_depth": 0.8, '
        '"warmth": 0.4, "dominant_tone": "professional", "rhetorical_patterns": []}'
    )
    mock_response.content = [mock_block]
    with patch.object(style_analyzer, "anthropic") as mock_anthropic_pkg:
        mock_client = MagicMock()
        mock_client.messages = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic_pkg.AsyncAnthropic.return_value = mock_client
        merged = await aggregate_style_from_corpus("Chunk one.\n\n---\n\nChunk two.", cfg)
    assert merged.profile_source == "llm"
    assert merged.technical_depth == 0.8
    mock_client.messages.create.assert_awaited_once()
