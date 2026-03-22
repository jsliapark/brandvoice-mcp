"""Tests for shared LLM JSON extraction."""

from __future__ import annotations

import json

import pytest

from brandvoice_mcp.llm_json import extract_json_object


def test_extract_json_prefixed_chatter() -> None:
    raw = (
        'Here is the result:\n{"alignment_score": 88, "verdict": "on_brand", '
        '"drift_flags": [], "suggestions": [], "rewrite_hints": "ok"}\nThanks!'
    )
    data = extract_json_object(raw)
    assert data["alignment_score"] == 88
    assert data["verdict"] == "on_brand"


def test_extract_json_invalid_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        extract_json_object("not json at all")
