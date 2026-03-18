"""Prompt template loader.

All prompt templates live as .md files in this directory. Load them by name
with ``load_prompt("style_analysis")`` and format with ``.format(**kwargs)``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Read and cache a prompt template from ``prompts/{name}.md``.

    Returns the raw string with ``{placeholder}`` markers intact so callers
    can fill them with ``.format()``.
    """
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")
