"""ingest_samples tool — Feed writing samples to learn the user's voice."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brandvoice_mcp.analysis.style_analyzer import (
    analyze_style,
    chunk_content,
    heuristic_style_snapshot,
)
from brandvoice_mcp.models import IngestResult

# Minimum word count for LLM style analysis (below this: heuristic only, no API call).
_MIN_STYLE_ANALYSIS_WORDS = 50

if TYPE_CHECKING:
    from brandvoice_mcp.config import Config
    from brandvoice_mcp.storage.chromadb import VoiceStore
    from brandvoice_mcp.storage.embeddings import EmbeddingService


async def ingest_samples(
    content: str,
    source: str,
    title: str | None,
    url: str | None,
    *,
    config: Config,
    store: VoiceStore,
    embeddings: EmbeddingService,
) -> IngestResult:
    """Process and store a writing sample."""
    word_count = len(content.split())
    analysis_note: str | None = None
    if word_count < _MIN_STYLE_ANALYSIS_WORDS:
        style = heuristic_style_snapshot(content)
        analysis_note = (
            f"Sample stored but too short for LLM style analysis "
            f"(minimum ~{_MIN_STYLE_ANALYSIS_WORDS} words; this has {word_count})."
        )
    else:
        style = await analyze_style(content, config)

    chunks = chunk_content(
        content,
        target_tokens=config.chunk_target_tokens,
        min_tokens=config.chunk_min_tokens,
    )
    if not chunks:
        chunks = [content.strip()]

    vectors = await embeddings.embed_texts(chunks)

    style_markers = {
        "avg_sentence_length": style.avg_sentence_length,
        "vocabulary_richness": style.vocabulary_richness,
        "formality_score": style.formality_score,
        "dominant_tone": style.dominant_tone,
    }
    await store.add_samples_async(
        chunks=chunks,
        embeddings=vectors,
        metadata={
            "source": source,
            "title": title or "",
            "url": url or "",
            "style_markers": style_markers,
        },
    )

    total = await store.sample_count_async()
    profile_updated = False

    # Aggregate profile update: only from LLM-derived snapshots (skip short-only heuristics).
    # TODO: Full corpus re-analysis merging all samples via Claude.
    if (
        total >= config.profile_reanalysis_threshold
        and style.profile_source == "llm"
    ):
        await store.save_learned_style_async(style.model_dump())
        profile_updated = True

    return IngestResult(
        samples_stored=len(chunks),
        total_samples=total,
        voice_profile_updated=profile_updated,
        style_snapshot=style,
        analysis_note=analysis_note,
    )
