"""ingest_samples tool — Feed writing samples to learn the user's voice."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brandvoice_mcp.analysis.style_analyzer import analyze_style, chunk_content
from brandvoice_mcp.models import IngestResult

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

    # TODO: Aggregate profile re-analysis. Currently saves the latest single-sample
    # analysis. Should pull ALL stored samples and build a comprehensive voice
    # profile via Claude, merging patterns across the full corpus.
    if total >= config.profile_reanalysis_threshold:
        await store.save_learned_style_async(style.model_dump())
        profile_updated = True

    return IngestResult(
        samples_stored=len(chunks),
        total_samples=total,
        voice_profile_updated=profile_updated,
        style_snapshot=style,
    )
