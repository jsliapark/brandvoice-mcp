"""LLM-based style analysis using Claude structured outputs.

Analyzes writing samples for sentence structure, vocabulary richness,
formality markers, tone, and rhetorical patterns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from brandvoice_mcp.models import StyleSnapshot
from brandvoice_mcp.prompts import load_prompt

if TYPE_CHECKING:
    from brandvoice_mcp.config import Config


# TODO: Replace heuristic analysis with Claude structured output call.
# Use load_prompt("style_analysis").format(content=content) with
# config.analysis_model and parse the response into a StyleSnapshot.
async def analyze_style(content: str, config: Config) -> StyleSnapshot:
    """Analyze a writing sample and return a StyleSnapshot.

    Calls Claude with a structured prompt to extract style metrics.
    This will be implemented with actual LLM calls in a later phase.
    For now, returns a basic statistical analysis.
    """
    words = content.split()
    sentences = [s.strip() for s in content.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    unique_words = set(w.lower().strip(".,!?;:'\"()[]{}") for w in words)

    avg_sentence_length = len(words) / max(len(sentences), 1)
    vocabulary_richness = len(unique_words) / max(len(words), 1)

    formality_markers = {"furthermore", "however", "therefore", "consequently", "nevertheless", "moreover"}
    casual_markers = {"gonna", "wanna", "kinda", "lol", "btw", "imo", "tbh"}
    formal_count = sum(1 for w in unique_words if w in formality_markers)
    casual_count = sum(1 for w in unique_words if w in casual_markers)
    formality_score = min(1.0, max(0.0, 0.5 + (formal_count - casual_count) * 0.1))

    # TODO: dominant_tone and rhetorical_patterns are hardcoded placeholders.
    # LLM-based analysis will detect these from the actual content.
    return StyleSnapshot(
        avg_sentence_length=round(avg_sentence_length, 1),
        vocabulary_richness=round(min(1.0, vocabulary_richness), 3),
        formality_score=round(formality_score, 2),
        dominant_tone="conversational",
        rhetorical_patterns=[],
    )


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
            sentences = [s.strip() + "." for s in para.replace("!", "!|").replace("?", "?|").replace(". ", ".|").split("|") if s.strip()]
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
