"""Tests for the set_guidelines tool."""

from __future__ import annotations

import pytest

from brandvoice_mcp.storage.chromadb import VoiceStore
from brandvoice_mcp.tools.guidelines import set_guidelines


@pytest.mark.asyncio
async def test_set_pillars(store: VoiceStore) -> None:
    result = await set_guidelines(
        store=store,
        pillars=["practical over theoretical", "teach through building"],
    )
    assert "pillars" in result.updated_fields
    assert result.current_guidelines["pillars"] == [
        "practical over theoretical",
        "teach through building",
    ]


@pytest.mark.asyncio
async def test_merge_semantics(store: VoiceStore) -> None:
    """Only provided fields should be updated; others stay intact."""
    await set_guidelines(store=store, pillars=["first pillar"])
    result = await set_guidelines(
        store=store,
        preferred_vocabulary=["ship", "build"],
    )
    assert result.current_guidelines["pillars"] == ["first pillar"]
    assert result.current_guidelines["preferred_vocabulary"] == ["ship", "build"]


@pytest.mark.asyncio
async def test_set_tone(store: VoiceStore) -> None:
    result = await set_guidelines(
        store=store,
        tone={"formality": 0.3, "humor": 0.5, "technical_depth": 0.8, "warmth": 0.7},
    )
    assert "tone" in result.updated_fields
    assert result.current_guidelines["tone"]["formality"] == 0.3
