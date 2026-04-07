"""ingest_samples tool — Feed writing samples to learn the user's voice."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from brandvoice_mcp.analysis.style_analyzer import (
    aggregate_style_from_corpus,
    analyze_style,
    chunk_content,
    heuristic_style_snapshot,
)
from brandvoice_mcp.models import IngestResult, SourceType, StyleSnapshot

logger = logging.getLogger(__name__)

# Minimum word count for LLM style analysis (below this: heuristic only, no API call).
_MIN_STYLE_ANALYSIS_WORDS = 50

if TYPE_CHECKING:
    from brandvoice_mcp.config import Config
    from brandvoice_mcp.storage.chromadb import VoiceStore
    from brandvoice_mcp.storage.embeddings import EmbeddingService


async def _maybe_update_aggregate_profile(
    *,
    total_samples: int,
    config: Config,
    store: VoiceStore,
    style: StyleSnapshot,
) -> bool:
    """Persist learned style when sample count crosses the reanalysis threshold.

    Production runs merge stored chunks via Claude; test mode saves the latest
    LLM-derived ingest snapshot only. Returns whether ``profile.json`` was written.
    """
    if total_samples < config.profile_reanalysis_threshold:
        return False

    if config.analysis_model != "test":
        excerpts = await store.get_corpus_excerpts_async()
        corpus = "\n\n---\n\n".join(excerpts) if excerpts else ""
        try:
            merged = await aggregate_style_from_corpus(corpus, config)
            await store.save_learned_style_async(merged.model_dump())
            return True
        except Exception as exc:
            logger.warning(
                "Corpus style merge failed (%s); using latest ingest snapshot if LLM-derived",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            if style.profile_source != "llm":
                return False
            await store.save_learned_style_async(style.model_dump())
            return True

    if style.profile_source != "llm":
        return False
    await store.save_learned_style_async(style.model_dump())
    return True


async def refresh_learned_profile_after_samples_change(
    *,
    config: Config,
    store: VoiceStore,
) -> None:
    """Recompute or clear the aggregate learned profile after samples were removed."""
    total = await store.sample_count_async()
    if total == 0:
        await store.reset_profile_to_default_async()
        return

    excerpts = await store.get_corpus_excerpts_async()
    corpus = "\n\n---\n\n".join(excerpts) if excerpts else ""
    try:
        merged = await aggregate_style_from_corpus(corpus, config)
        await store.save_learned_style_async(merged.model_dump())
    except Exception as exc:
        logger.warning(
            "Profile regeneration after sample delete failed (%s); using heuristic snapshot",
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        snapshot = heuristic_style_snapshot(corpus)
        await store.save_learned_style_async(snapshot.model_dump())


async def ingest_samples(
    content: str,
    source: SourceType,
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
        "humor": style.humor,
        "technical_depth": style.technical_depth,
        "warmth": style.warmth,
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
    profile_updated = await _maybe_update_aggregate_profile(
        total_samples=total,
        config=config,
        store=store,
        style=style,
    )

    return IngestResult(
        samples_stored=len(chunks),
        total_samples=total,
        voice_profile_updated=profile_updated,
        style_snapshot=style,
        analysis_note=analysis_note,
    )
