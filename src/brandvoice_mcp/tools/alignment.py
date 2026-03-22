"""check_alignment tool — Score content against the user's voice profile."""

from __future__ import annotations

from typing import TYPE_CHECKING

from brandvoice_mcp.models import AlignmentResult, DriftFlag

if TYPE_CHECKING:
    from brandvoice_mcp.config import Config
    from brandvoice_mcp.storage.chromadb import VoiceStore


# TODO: Replace heuristic scoring with Claude structured output call.
# Use load_prompt("alignment_check").format(voice_profile=..., samples=...,
# content=...) with config.analysis_model and parse into AlignmentResult.
# The prompt template is ready in prompts/alignment_check.md.
async def check_alignment(
    content: str,
    platform: str,
    *,
    config: Config,
    store: VoiceStore,
) -> AlignmentResult:
    """Compare content against the stored voice profile.

    Currently uses heuristic comparison. Will be upgraded to LLM-based
    evaluation in a later phase.
    """
    learned = await store.get_learned_style_async()
    guidelines = await store.get_guidelines_async()
    total_samples = await store.sample_count_async()

    # Nothing to compare against: no ingested samples and no stored style/guidelines.
    if total_samples == 0 and not learned and not guidelines:
        return AlignmentResult(
            alignment_score=50,
            verdict="unknown",
            drift_flags=[
                DriftFlag(
                    category="profile",
                    issue="No voice profile exists yet — cannot assess alignment. Ingest writing samples first.",
                    severity="medium",
                )
            ],
            suggestions=[
                "Use ingest_samples to teach the system your writing style before checking alignment."
            ],
            rewrite_hints="",
        )

    drift_flags: list[DriftFlag] = []
    score = 70  # baseline

    words = content.split()
    sentences = [s.strip() for s in content.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    avg_sent_len = len(words) / max(len(sentences), 1)

    if learned:
        expected_len = learned.get("avg_sentence_length", 15)
        len_diff = abs(avg_sent_len - expected_len)
        if len_diff > 10:
            drift_flags.append(
                DriftFlag(
                    category="sentence_structure",
                    issue=f"Your writing averages {expected_len:.0f}-word sentences, but this content averages {avg_sent_len:.0f} words.",
                    severity="high" if len_diff > 15 else "medium",
                )
            )
            score -= int(len_diff * 1.5)

        expected_formality = learned.get("formality_score", 0.5)
        formality_markers = {"furthermore", "however", "therefore", "consequently", "nevertheless"}
        casual_markers = {"gonna", "wanna", "kinda", "lol", "btw"}
        unique_words = set(w.lower().strip(".,!?;:'\"") for w in words)
        formal_count = sum(1 for w in unique_words if w in formality_markers)
        casual_count = sum(1 for w in unique_words if w in casual_markers)
        content_formality = min(1.0, max(0.0, 0.5 + (formal_count - casual_count) * 0.1))
        formality_diff = abs(content_formality - expected_formality)
        if formality_diff > 0.3:
            direction = "formal" if content_formality > expected_formality else "casual"
            drift_flags.append(
                DriftFlag(
                    category="formality",
                    issue=f"Content is more {direction} than your usual style.",
                    severity="medium" if formality_diff < 0.5 else "high",
                )
            )
            score -= int(formality_diff * 20)

    if guidelines:
        avoided = guidelines.get("avoided_vocabulary", [])
        used_avoided = [w for w in avoided if w.lower() in content.lower()]
        if used_avoided:
            drift_flags.append(
                DriftFlag(
                    category="vocabulary",
                    issue=f"Uses avoided terms: {', '.join(used_avoided)}",
                    severity="medium",
                )
            )
            score -= len(used_avoided) * 5

    score = max(0, min(100, score))

    if score >= 80:
        verdict = "on_brand"
    elif score >= 60:
        verdict = "minor_drift"
    elif score >= 40:
        verdict = "significant_drift"
    else:
        verdict = "off_brand"

    # TODO: Generate actionable suggestions and rewrite hints via LLM instead
    # of just echoing drift flag issues. LLM should produce specific rewrites.
    suggestions = [f.issue for f in drift_flags]
    rewrite_hints = "; ".join(suggestions) if suggestions else "Content aligns well with your voice profile."

    return AlignmentResult(
        alignment_score=score,
        verdict=verdict,
        drift_flags=drift_flags,
        suggestions=suggestions,
        rewrite_hints=rewrite_hints,
    )
