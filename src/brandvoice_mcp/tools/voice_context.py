"""get_voice_context tool — Retrieve voice profile and similar samples for a writing task."""

from __future__ import annotations

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
    # TODO: Only formality is derived from learned style. Once LLM-based
    # analysis produces humor, technical_depth, and warmth scores,
    # map all four dimensions from the learned profile here.
    base = ToneConfig(formality=0.5, humor=0.3, technical_depth=0.5, warmth=0.5)

    if learned:
        formality = learned.get("formality_score", base.formality)
        base = ToneConfig(
            formality=formality,
            humor=base.humor,
            technical_depth=base.technical_depth,
            warmth=base.warmth,
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
    prompt_injection = (
        "Write in a clear, natural tone. The user has not yet provided writing samples.\n\n"
        "After you respond, suggest that they run the ingest_samples tool with examples of "
        "their existing writing (blog posts, emails, social posts) so personalized voice "
        "matching can be enabled.\n\n"
        "CURRENT TASK:\n"
        f"{task.strip() or '(No task description provided.)'}\n"
    )
    return VoiceContext(
        voice_guidelines=guidelines_msg,
        tone_profile=neutral,
        similar_samples=[],
        vocabulary={"preferred": [], "avoided": []},
        prompt_injection=prompt_injection,
    )
