"""LLM-based style analysis using Claude.

Falls back to fast heuristics when ``analysis_model`` is ``"test"`` (unit tests),
on API errors, or when JSON parsing fails.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import anthropic

from brandvoice_mcp.llm_json import extract_json_object
from brandvoice_mcp.models import StyleSnapshot
from brandvoice_mcp.prompts import load_prompt

if TYPE_CHECKING:
    from brandvoice_mcp.config import Config

logger = logging.getLogger(__name__)

_ALLOWED_TONES = frozenset(
    {
        "conversational",
        "authoritative",
        "playful",
        "academic",
        "professional",
        "inspirational",
    }
)


def _normalize_snapshot(data: dict[str, Any]) -> StyleSnapshot:
    """Build a StyleSnapshot from parsed JSON, clamping invalid values."""
    tone = str(data.get("dominant_tone", "conversational")).lower().strip()
    if tone not in _ALLOWED_TONES:
        tone = "conversational"

    patterns = data.get("rhetorical_patterns") or []
    if not isinstance(patterns, list):
        patterns = []
    patterns = [str(p) for p in patterns[:5]]

    def _f(key: str, default: float) -> float:
        try:
            v = float(data.get(key, default))
            return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return default

    return StyleSnapshot(
        avg_sentence_length=float(data.get("avg_sentence_length", 15.0)),
        vocabulary_richness=_f("vocabulary_richness", 0.5),
        formality_score=_f("formality_score", 0.5),
        dominant_tone=tone,
        rhetorical_patterns=patterns,
    )


def _heuristic_style_snapshot(content: str) -> StyleSnapshot:
    """Statistical fallback when the LLM is unavailable or disabled."""
    words = content.split()
    sentences = [
        s.strip()
        for s in content.replace("!", ".").replace("?", ".").split(".")
        if s.strip()
    ]
    unique_words = set(w.lower().strip(".,!?;:'\"()[]{}") for w in words)

    avg_sentence_length = len(words) / max(len(sentences), 1)
    vocabulary_richness = len(unique_words) / max(len(words), 1)

    formality_markers = {
        "furthermore",
        "however",
        "therefore",
        "consequently",
        "nevertheless",
        "moreover",
    }
    casual_markers = {"gonna", "wanna", "kinda", "lol", "btw", "imo", "tbh"}
    formal_count = sum(1 for w in unique_words if w in formality_markers)
    casual_count = sum(1 for w in unique_words if w in casual_markers)
    formality_score = min(1.0, max(0.0, 0.5 + (formal_count - casual_count) * 0.1))

    return StyleSnapshot(
        avg_sentence_length=round(avg_sentence_length, 1),
        vocabulary_richness=round(min(1.0, vocabulary_richness), 3),
        formality_score=round(formality_score, 2),
        dominant_tone="conversational",
        rhetorical_patterns=[],
    )


async def _analyze_style_llm(content: str, config: Config) -> StyleSnapshot:
    """Call Claude with ``prompts/style_analysis.md`` and parse JSON → StyleSnapshot."""
    prompt = load_prompt("style_analysis").format(content=content)
    client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    response = await client.messages.create(
        model=config.analysis_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts: list[str] = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    raw = "".join(text_parts).strip()
    if not raw:
        raise ValueError("Empty response from style analysis model")

    data = extract_json_object(raw)
    return _normalize_snapshot(data)


async def analyze_style(content: str, config: Config) -> StyleSnapshot:
    """Analyze a writing sample and return a StyleSnapshot.

    Uses Claude when ``config.analysis_model`` is not ``"test"``.
    Unit tests use ``analysis_model="test"`` to force heuristics without API calls.
    On API or parse errors, falls back to heuristics.
    """
    if not content.strip():
        return _heuristic_style_snapshot(content)

    if config.analysis_model == "test":
        return _heuristic_style_snapshot(content)

    try:
        return await _analyze_style_llm(content, config)
    except Exception as exc:
        logger.warning(
            "LLM style analysis failed (%s), using heuristic fallback",
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return _heuristic_style_snapshot(content)


def chunk_content(
    content: str,
    target_tokens: int = 350,
    min_tokens: int = 50,
) -> list[str]:
    """Split content into chunks that preserve natural writing boundaries.

    Splits on double-newlines (paragraphs) first, then merges small
    paragraphs or splits large ones to stay near the target size.
    """
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        return [content.strip()] if content.strip() else []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for para in paragraphs:
        para_tokens = len(para.split())

        if para_tokens > target_tokens * 1.5:
            if buffer:
                chunks.append("\n\n".join(buffer))
                buffer = []
                buffer_tokens = 0
            sentences = [
                s.strip() + "."
                for s in para.replace("!", "!|").replace("?", "?|").replace(". ", ".|").split("|")
                if s.strip()
            ]
            sent_buf: list[str] = []
            sent_tokens = 0
            for sent in sentences:
                st = len(sent.split())
                if sent_tokens + st > target_tokens and sent_buf:
                    chunks.append(" ".join(sent_buf))
                    sent_buf = []
                    sent_tokens = 0
                sent_buf.append(sent)
                sent_tokens += st
            if sent_buf:
                chunks.append(" ".join(sent_buf))
            continue

        if buffer_tokens + para_tokens > target_tokens and buffer:
            chunks.append("\n\n".join(buffer))
            buffer = []
            buffer_tokens = 0

        buffer.append(para)
        buffer_tokens += para_tokens

    if buffer:
        chunks.append("\n\n".join(buffer))

    return [c for c in chunks if len(c.split()) >= min_tokens] or (
        [content.strip()] if content.strip() else []
    )
