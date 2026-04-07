"""check_alignment tool — Score content against the user's voice profile."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import anthropic

from brandvoice_mcp.llm_json import extract_json_object
from brandvoice_mcp.models import AlignmentResult, DriftFlag, PlatformType
from brandvoice_mcp.prompts import load_prompt

if TYPE_CHECKING:
    from brandvoice_mcp.config import Config
    from brandvoice_mcp.storage.chromadb import VoiceStore

logger = logging.getLogger(__name__)

_LLM_VERDICTS = frozenset(
    {"on_brand", "minor_drift", "significant_drift", "off_brand"}
)

# Verdict strings produced by the alignment LLM / heuristic paths (not "unknown").
_LlmVerdict = Literal["on_brand", "minor_drift", "significant_drift", "off_brand"]


def _voice_profile_text(
    learned: dict[str, Any] | None,
    guidelines: dict[str, Any] | None,
) -> str:
    parts: list[str] = []
    if learned:
        parts.append("Learned style metrics (JSON):\n" + json.dumps(learned, indent=2))
    if guidelines:
        parts.append("Explicit guidelines (JSON):\n" + json.dumps(guidelines, indent=2))
    if not parts:
        return "(No structured profile JSON; use reference samples only.)"
    return "\n\n".join(parts)


def _normalize_alignment_result(data: dict[str, Any]) -> AlignmentResult:
    raw_verdict = str(data.get("verdict", "minor_drift")).strip()
    verdict: _LlmVerdict = (
        cast(_LlmVerdict, raw_verdict)
        if raw_verdict in _LLM_VERDICTS
        else "minor_drift"
    )

    score = int(float(data.get("alignment_score", 50)))
    score = max(0, min(100, score))

    flags: list[DriftFlag] = []
    for item in data.get("drift_flags") or []:
        if not isinstance(item, dict):
            continue
        sev = item.get("severity", "medium")
        if sev not in ("low", "medium", "high"):
            sev = "medium"
        flags.append(
            DriftFlag(
                category=str(item.get("category", "tone")),
                issue=str(item.get("issue", "")),
                severity=sev,
            )
        )

    suggestions = data.get("suggestions") or []
    if not isinstance(suggestions, list):
        suggestions = []
    suggestions = [str(s) for s in suggestions if s]

    hints = str(data.get("rewrite_hints", "") or "")

    return AlignmentResult(
        alignment_score=score,
        verdict=verdict,
        drift_flags=flags,
        suggestions=suggestions,
        rewrite_hints=hints,
    )


async def _check_alignment_llm(
    content: str,
    platform: PlatformType,
    *,
    config: Config,
    store: VoiceStore,
    learned: dict[str, Any] | None,
    guidelines: dict[str, Any] | None,
) -> AlignmentResult:
    snippets = await store.get_sample_snippets_async(limit=3, max_chars_per_sample=1200)
    samples_block = "\n---\n".join(snippets) if snippets else "(No stored samples.)"

    user_prompt = load_prompt("alignment_check").format(
        voice_profile=_voice_profile_text(learned, guidelines),
        samples=samples_block,
        platform=platform,
        content=content,
    )

    client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
    response = await client.messages.create(
        model=config.analysis_model,
        max_tokens=2048,
        messages=[{"role": "user", "content": user_prompt}],
    )

    text_parts: list[str] = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    raw = "".join(text_parts).strip()
    if not raw:
        raise ValueError("Empty alignment model response")

    data = extract_json_object(raw)
    return _normalize_alignment_result(data)


def _check_alignment_heuristic(
    content: str,
    learned: dict[str, Any] | None,
    guidelines: dict[str, Any] | None,
) -> AlignmentResult:
    """Fast local scoring when LLM is disabled or unavailable."""
    drift_flags: list[DriftFlag] = []
    score = 70

    words = content.split()
    sentences = [
        s.strip()
        for s in content.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    avg_sent_len = len(words) / max(len(sentences), 1)

    if learned:
        expected_len = learned.get("avg_sentence_length", 15)
        len_diff = abs(avg_sent_len - float(expected_len))
        if len_diff > 10:
            drift_flags.append(
                DriftFlag(
                    category="sentence_structure",
                    issue=(
                        f"Your writing averages {float(expected_len):.0f}-word sentences, "
                        f"but this content averages {avg_sent_len:.0f} words."
                    ),
                    severity="high" if len_diff > 15 else "medium",
                )
            )
            score -= int(len_diff * 1.5)

        expected_formality = float(learned.get("formality_score", 0.5))
        formality_markers = {
            "furthermore",
            "however",
            "therefore",
            "consequently",
            "nevertheless",
        }
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

    verdict: _LlmVerdict
    if score >= 80:
        verdict = "on_brand"
    elif score >= 60:
        verdict = "minor_drift"
    elif score >= 40:
        verdict = "significant_drift"
    else:
        verdict = "off_brand"

    suggestions = [f.issue for f in drift_flags]
    rewrite_hints = (
        "; ".join(suggestions) if suggestions else "Content aligns well with your voice profile."
    )

    return AlignmentResult(
        alignment_score=score,
        verdict=verdict,
        drift_flags=drift_flags,
        suggestions=suggestions,
        rewrite_hints=rewrite_hints,
    )


async def check_alignment(
    content: str,
    platform: PlatformType,
    *,
    config: Config,
    store: VoiceStore,
) -> AlignmentResult:
    """Compare content against the stored voice profile.

    Uses Claude + ``prompts/alignment_check.md`` when ``analysis_model`` is not
    ``"test"``. Falls back to heuristics on API/parse errors or in test mode.
    """
    learned = await store.get_learned_style_async()
    guidelines = await store.get_guidelines_async()
    total_samples = await store.sample_count_async()

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

    if config.analysis_model == "test":
        return _check_alignment_heuristic(content, learned, guidelines)

    try:
        return await _check_alignment_llm(
            content,
            platform,
            config=config,
            store=store,
            learned=learned,
            guidelines=guidelines,
        )
    except Exception as exc:
        logger.warning(
            "LLM alignment failed (%s), using heuristic fallback",
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return _check_alignment_heuristic(content, learned, guidelines)
