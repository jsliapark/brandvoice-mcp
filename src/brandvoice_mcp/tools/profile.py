"""get_profile tool — Return the complete voice profile."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brandvoice_mcp.models import StyleSnapshot, VoiceProfile

if TYPE_CHECKING:
    from brandvoice_mcp.storage.chromadb import VoiceStore


async def get_profile(*, store: VoiceStore) -> VoiceProfile:
    """Assemble and return the full voice profile."""
    learned_raw = await store.get_learned_style_async()
    guidelines = await store.get_guidelines_async()
    total = await store.sample_count_async()
    breakdown = await store.sources_breakdown_async()
    last_updated = await store.get_profile_last_updated_async()

    learned_style = None
    if learned_raw:
        learned_style = StyleSnapshot(
            avg_sentence_length=learned_raw.get("avg_sentence_length", 0),
            vocabulary_richness=learned_raw.get("vocabulary_richness", 0),
            formality_score=learned_raw.get("formality_score", 0.5),
            dominant_tone=learned_raw.get("dominant_tone", "unknown"),
            rhetorical_patterns=learned_raw.get("rhetorical_patterns", []),
        )

    parts: list[str] = []
    if learned_style:
        parts.append(
            f"Tone: {learned_style.dominant_tone}. "
            f"Avg sentence length: {learned_style.avg_sentence_length:.0f} words. "
            f"Formality: {learned_style.formality_score:.0%}."
        )
    if guidelines:
        pillars = guidelines.get("pillars")
        if pillars:
            parts.append("Pillars: " + ", ".join(pillars) + ".")
    if not parts:
        parts.append("No style data yet. Ingest writing samples to build your profile.")

    return VoiceProfile(
        learned_style=learned_style,
        explicit_guidelines=guidelines,
        total_samples=total,
        sources_breakdown=breakdown,
        style_summary=" ".join(parts),
        last_updated=last_updated,
    )
