"""Build the prompt_injection string that makes any LLM match the user's voice.

The prompt_injection is the primary output of get_voice_context — a
ready-to-use block that any MCP client can prepend to its system prompt.

The overall template lives in ``prompts/voice_injection.md``. This module
builds the dynamic sections (tone, vocabulary, platform, samples) and
fills them into that template.
"""

from __future__ import annotations

import re
from typing import Any

from brandvoice_mcp.models import ToneConfig, VoiceSample
from brandvoice_mcp.prompts import load_prompt

_PLATFORM_HINTS = {
    "blog": "Format for a blog post: use headers, paragraphs, and a conversational but informative tone.",
    "linkedin": "Format for LinkedIn: professional but personable, use line breaks for readability, include a hook.",
    "twitter": "Format for Twitter/X: be concise, punchy, under 280 characters per tweet. Use threads for longer content.",
    "email": "Format for email: clear subject line, scannable paragraphs, direct call-to-action.",
}


# TODO: Iterate on prompt_injection format with real LLM clients.
# Edit prompts/voice_injection.md and test with Claude Desktop, Cursor,
# and GPT to ensure the format actually changes voice output.
def build_prompt_injection(
    voice_guidelines: str,
    tone: ToneConfig,
    similar_samples: list[VoiceSample],
    vocabulary: dict[str, list[str]],
    platform: str = "general",
) -> str:
    """Assemble the prompt_injection string.

    Loads the template from ``prompts/voice_injection.md`` and fills in
    dynamic sections. The format is designed to work when prepended to any
    LLM's system prompt, regardless of provider (Claude, GPT, etc.).
    """
    template = load_prompt("voice_injection")

    tone_section = (
        f"  Formality: {_score_label(tone.formality)} ({tone.formality:.1f}/1.0)\n"
        f"  Humor: {_score_label(tone.humor)} ({tone.humor:.1f}/1.0)\n"
        f"  Technical depth: {_score_label(tone.technical_depth)} ({tone.technical_depth:.1f}/1.0)\n"
        f"  Warmth: {_score_label(tone.warmth)} ({tone.warmth:.1f}/1.0)"
    )

    vocab_lines: list[str] = []
    preferred = vocabulary.get("preferred", [])
    avoided = vocabulary.get("avoided", [])
    if preferred or avoided:
        vocab_lines.append("VOCABULARY:")
        if preferred:
            vocab_lines.append(f"  Prefer these terms: {', '.join(preferred)}")
        if avoided:
            vocab_lines.append(f"  Avoid these terms: {', '.join(avoided)}")
    vocabulary_section = "\n".join(vocab_lines)

    platform_lines: list[str] = []
    if platform != "general":
        hint = _PLATFORM_HINTS.get(platform)
        if hint:
            platform_lines.append(f"PLATFORM ({platform.upper()}):")
            platform_lines.append(f"  {hint}")
    platform_section = "\n".join(platform_lines)

    sample_lines: list[str] = []
    if similar_samples:
        sample_lines.append("REFERENCE SAMPLES (match this style):")
        for sample in similar_samples:
            sample_lines.append("---")
            if sample.title:
                sample_lines.append(f"[{sample.source}] {sample.title}")
            sample_lines.append(sample.content)
        sample_lines.append("---")
    samples_section = "\n".join(sample_lines)

    result = template.format(
        voice_guidelines=voice_guidelines or "No specific guidelines set yet.",
        tone_section=tone_section,
        vocabulary_section=vocabulary_section,
        platform_section=platform_section,
        samples_section=samples_section,
    )

    return re.sub(r"\n{3,}", "\n\n", result).strip()


def _score_label(score: float) -> str:
    if score < 0.2:
        return "very low"
    if score < 0.4:
        return "low"
    if score < 0.6:
        return "moderate"
    if score < 0.8:
        return "high"
    return "very high"
