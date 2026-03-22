"""Parse JSON objects from LLM text responses (fences, stray whitespace)."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a single JSON object from model output; strip optional markdown fences."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*)\n?```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Model may prefix/suffix chatter — take first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
