"""LLM-based style analysis using Claude.

Falls back to fast heuristics when ``analysis_model`` is ``"test"`` (unit tests),
on API errors, or when JSON parsing fails.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import anthropic

from brandvoice_mcp.llm_json import extract_json_object
from brandvoice_mcp.models import StyleSnapshot
from brandvoice_mcp.prompts import load_prompt

if TYPE_CHECKING:
    from brandvoice_mcp.config import Config

logger = logging.getLogger(__name__)

# ── Heuristic scoring constants ──
_FORMALITY_BASE = 0.5
_FORMALITY_HIT_WEIGHT = 0.1
_HUMOR_BASE = 0.35
_HUMOR_HIT_WEIGHT = 0.12
_TECHNICAL_DEPTH_BASE = 0.4
_TECHNICAL_DEPTH_HIT_WEIGHT = 0.08
_WARMTH_BASE = 0.45
_WARMTH_HIT_WEIGHT = 0.05

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
        humor=_f("humor", 0.5),
        technical_depth=_f("technical_depth", 0.5),
        warmth=_f("warmth", 0.5),
        dominant_tone=tone,
        rhetorical_patterns=patterns,
        profile_source="llm",
    )


def heuristic_style_snapshot(content: str) -> StyleSnapshot:
    """Statistical fallback when the LLM is unavailable, disabled, or skipped."""
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
    formality_score = min(1.0, max(0.0, _FORMALITY_BASE + (formal_count - casual_count) * _FORMALITY_HIT_WEIGHT))

    humor_markers = {"lol", "haha", "heh", "lmao", "jk", "j/k", "funny"}
    humor_hits = sum(1 for w in words if w.lower().strip(".,!?;:'\"()[]{}") in humor_markers)
    humor = min(1.0, max(0.0, _HUMOR_BASE + humor_hits * _HUMOR_HIT_WEIGHT))

    tech_tokens = {
        "api",
        "async",
        "def",
        "import",
        "function",
        "class",
        "http",
        "sql",
        "json",
        "graphql",
        "typescript",
        "python",
        "docker",
        "kubernetes",
        "database",
        "endpoint",
        "schema",
        "deploy",
    }
    tech_hits = sum(1 for w in unique_words if w in tech_tokens)
    technical_depth = min(1.0, max(0.0, _TECHNICAL_DEPTH_BASE + tech_hits * _TECHNICAL_DEPTH_HIT_WEIGHT))

    warm_markers = {"you", "your", "we", "our", "us", "i've", "i'm", "let's"}
    warm_hits = sum(1 for w in unique_words if w in warm_markers)
    warmth = min(1.0, max(0.0, _WARMTH_BASE + warm_hits * _WARMTH_HIT_WEIGHT))

    return StyleSnapshot(
        avg_sentence_length=round(avg_sentence_length, 1),
        vocabulary_richness=round(min(1.0, vocabulary_richness), 3),
        formality_score=round(formality_score, 2),
        humor=round(humor, 2),
        technical_depth=round(technical_depth, 2),
        warmth=round(warmth, 2),
        dominant_tone="conversational",
        rhetorical_patterns=[],
        profile_source="heuristic",
    )


def _thinking_kwargs(config: Config) -> dict[str, Any]:
    """Extra kwargs to pass to ``client.messages.create`` when extended thinking is on.

    ``max_tokens`` must exceed ``budget_tokens``, so we reserve 2 048 tokens for
    the JSON output on top of the thinking budget.
    """
    if config.extended_thinking:
        return {
            "thinking": {"type": "thinking", "budget_tokens": config.thinking_budget},
            "max_tokens": config.thinking_budget + 2048,
        }
    return {"max_tokens": 1024}


async def _analyze_style_llm(content: str, config: Config) -> StyleSnapshot:
    """Call Claude with ``prompts/style_analysis.md`` and parse JSON → StyleSnapshot."""
    prompt = load_prompt("style_analysis").format(content=content)
    client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    response = await client.messages.create(
        model=config.analysis_model,
        messages=[{"role": "user", "content": prompt}],
        **_thinking_kwargs(config),
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
        return heuristic_style_snapshot(content)

    if config.analysis_model == "test":
        return heuristic_style_snapshot(content)

    try:
        return await _analyze_style_llm(content, config)
    except Exception as exc:
        logger.warning(
            "LLM style analysis failed (%s), using heuristic fallback",
            exc,
            exc_info=logger.isEnabledFor(logging.DEBUG),
        )
        return heuristic_style_snapshot(content)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")

_CORPUS_MAX_CHARS = 100_000


async def _aggregate_style_llm(corpus: str, config: Config) -> StyleSnapshot:
    """Call Claude with ``prompts/corpus_aggregate.md`` for a merged StyleSnapshot."""
    trimmed = corpus[:_CORPUS_MAX_CHARS]
    prompt = load_prompt("corpus_aggregate").format(corpus=trimmed)
    client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    response = await client.messages.create(
        model=config.analysis_model,
        messages=[{"role": "user", "content": prompt}],
        **_thinking_kwargs(config),
    )

    text_parts: list[str] = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    raw = "".join(text_parts).strip()
    if not raw:
        raise ValueError("Empty response from corpus aggregate model")

    data = extract_json_object(raw)
    return _normalize_snapshot(data)


async def aggregate_style_from_corpus(corpus_text: str, config: Config) -> StyleSnapshot:
    """Merge many stored excerpts into one aggregate profile (Claude, or heuristics in test mode)."""
    if not corpus_text.strip():
        return heuristic_style_snapshot("")

    if config.analysis_model == "test":
        return heuristic_style_snapshot(corpus_text)

    return await _aggregate_style_llm(corpus_text, config)


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
            sentences = [s.strip() for s in _SENTENCE_RE.split(para) if s.strip()]
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
