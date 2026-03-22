"""list_samples tool — Browse ingested writing samples."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from brandvoice_mcp.models import SampleEntry, SamplesList

if TYPE_CHECKING:
    from brandvoice_mcp.storage.chromadb import VoiceStore


async def list_samples(
    *,
    store: VoiceStore,
    source: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> SamplesList:
    """List stored writing samples with optional filtering."""
    entries_raw, total = await store.list_samples_async(
        source=source, limit=limit, offset=offset
    )

    samples = [
        SampleEntry(
            id=e["id"],
            content_preview=e["content_preview"],
            source=e["source"],
            title=e.get("title"),
            ingested_at=(
                datetime.fromisoformat(e["ingested_at"])
                if e.get("ingested_at")
                else None
            ),
        )
        for e in entries_raw
    ]

    return SamplesList(
        samples=samples,
        total=total,
        offset=offset,
        limit=limit,
    )
