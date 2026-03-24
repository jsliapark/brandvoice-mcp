"""get_voice_context tool — Retrieve voice profile and similar samples for a writing task."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from brandvoice_mcp.analysis.prompt_builder import build_prompt_injection
from brandvoice_mcp.models import ToneConfig, VoiceContext, VoiceSample

if TYPE_CHECKING:
    from brandvoice_mcp.config import Config
    from brandvoice_mcp.storage.chromadb import VoiceStore
    from brandvoice_mcp.storage.embeddings import EmbeddingService

_PLATFORM_SOURCE_MAP = {
    "blog": "blog",
    "linkedin": "social",
    "twitter": "social",
    "email": "email",
}

_MAX_TOP_K = 5

logger = logging.getLogger(__name__)


async def get_voice_context(
    task: str,
    platform: str,
    top_k: int,
    *,
    config: Config,
    store: VoiceStore,
    embeddings: EmbeddingService,
) -> VoiceContext:
    """Build full voice context for a writing task."""
    requested = top_k
    if top_k < 1:
        top_k = 1
        logger.info("top_k=%s is below 1; using 1", requested)
    elif top_k > _MAX_TOP_K:
        top_k = _MAX_TOP_K
        logger.info(
            "top_k=%s exceeds maximum %s; clamping to avoid oversized prompt_injection",
            requested,
            _MAX_TOP_K,
        )

    if await store.sample_count_async() == 0:
        return _empty_context(task)

    learned = await store.get_learned_style_async()
    guidelines = (await store.get_guidelines_async()) or {}

    tone = _resolve_tone(learned, guidelines)
    voice_guidelines = _resolve_guidelines(learned, guidelines)
    vocabulary = {
        "preferred": guidelines.get("preferred_vocabulary", []),
        "avoided": guidelines.get("avoided_vocabulary", []),
    }

    task_embedding = await embeddings.embed_text(task)
    source_filter = _PLATFORM_SOURCE_MAP.get(platform)
    raw_samples = await store.query_samples_async(
        query_embedding=task_embedding,
        top_k=top_k,
        source_filter=source_filter,
    )

    similar_samples = [
        VoiceSample(
            content=s["content"],
            source=s["source"],
            similarity=s["similarity"],
            title=s.get("title"),
        )
        for s in raw_samples
    ]

    prompt_injection = build_prompt_injection(
        voice_guidelines=voice_guidelines,
        tone=tone,
        similar_samples=similar_samples,
        vocabulary=vocabulary,
        platform=platform,
        task=task,
    )

    return VoiceContext(
        voice_guidelines=voice_guidelines,
        tone_profile=tone,
        similar_samples=similar_samples,
        vocabulary=vocabulary,
        prompt_injection=prompt_injection,
    )


def _resolve_tone(
    learned: dict | None,
    guidelines: dict | None,
) -> ToneConfig:
    """Merge learned tone with explicit overrides."""
    base = ToneConfig(formality=0.5, humor=0.3, technical_depth=0.5, warmth=0.5)

    if learned:

        def _tone_f(key: str, default: float) -> float:
            try:
                v = float(learned.get(key, default))
                return max(0.0, min(1.0, v))
            except (TypeError, ValueError):
                return default

        base = ToneConfig(
            formality=_tone_f("formality_score", base.formality),
            humor=_tone_f("humor", base.humor),
            technical_depth=_tone_f("technical_depth", base.technical_depth),
            warmth=_tone_f("warmth", base.warmth),
        )

    explicit_tone = (guidelines or {}).get("tone")
    if explicit_tone and isinstance(explicit_tone, dict):
        base = ToneConfig(
            formality=explicit_tone.get("formality", base.formality),
            humor=explicit_tone.get("humor", base.humor),
            technical_depth=explicit_tone.get("technical_depth", base.technical_depth),
            warmth=explicit_tone.get("warmth", base.warmth),
        )

    return base


def _resolve_guidelines(learned: dict | None, guidelines: dict | None) -> str:
    """Build a human-readable voice guidelines string."""
    parts: list[str] = []

    if guidelines:
        pillars = guidelines.get("pillars")
        if pillars:
            parts.append("Core pillars: " + ", ".join(pillars) + ".")
        custom = guidelines.get("custom_instructions")
        if custom:
            parts.append(custom)

    if learned:
        tone = learned.get("dominant_tone", "")
        if tone:
            parts.append(f"Dominant tone: {tone}.")
        patterns = learned.get("rhetorical_patterns", [])
        if patterns:
            parts.append("Patterns: " + ", ".join(patterns) + ".")

    return " ".join(parts) if parts else "Voice profile is still being built from your samples."


def _empty_context(task: str) -> VoiceContext:
    """Graceful degradation when no writing samples exist yet."""
    neutral = ToneConfig(formality=0.5, humor=0.3, technical_depth=0.5, warmth=0.5)
    guidelines_msg = (
        "No voice profile exists yet. Ingest writing samples so the system can learn your style."
    )
    inner = (
        "Write in a clear, natural tone. The user has not yet provided writing samples.\n\n"
        "After you respond, suggest that they run the ingest_samples tool with examples of "
        "their existing writing (blog posts, emails, social posts) so personalized voice "
        "matching can be enabled.\n\n"
        "CURRENT TASK:\n"
        f"{task.strip() or '(No task description provided.)'}\n"
    )
    prompt_injection = f"<voice_context>\n{inner}\n</voice_context>"
    return VoiceContext(
        voice_guidelines=guidelines_msg,
        tone_profile=neutral,
        similar_samples=[],
        vocabulary={"preferred": [], "avoided": []},
        prompt_injection=prompt_injection,
    )
