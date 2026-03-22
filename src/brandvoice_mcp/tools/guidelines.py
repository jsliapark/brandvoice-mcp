"""set_guidelines tool — Explicitly configure brand voice."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from brandvoice_mcp.models import GuidelinesResult

if TYPE_CHECKING:
    from brandvoice_mcp.storage.chromadb import VoiceStore


async def set_guidelines(
    *,
    store: VoiceStore,
    pillars: list[str] | None = None,
    tone: dict[str, float] | None = None,
    preferred_vocabulary: list[str] | None = None,
    avoided_vocabulary: list[str] | None = None,
    topics: list[str] | None = None,
    custom_instructions: str | None = None,
) -> GuidelinesResult:
    """Update explicit brand voice guidelines (merge semantics)."""
    existing = store.get_guidelines() or {}
    updated_fields: list[str] = []

    field_map: dict[str, Any] = {
        "pillars": pillars,
        "tone": tone,
        "preferred_vocabulary": preferred_vocabulary,
        "avoided_vocabulary": avoided_vocabulary,
        "topics": topics,
        "custom_instructions": custom_instructions,
    }

    for field_name, value in field_map.items():
        if value is not None:
            existing[field_name] = value
            updated_fields.append(field_name)

    store.save_guidelines(existing)

    return GuidelinesResult(
        updated_fields=updated_fields,
        current_guidelines=existing,
    )
