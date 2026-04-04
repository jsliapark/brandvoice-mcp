"""delete_samples tool — Remove ingested writing samples and refresh the voice profile."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, model_validator

from brandvoice_mcp.models import DeleteSamplesResult
from brandvoice_mcp.tools.ingest import refresh_learned_profile_after_samples_change

if TYPE_CHECKING:
    from brandvoice_mcp.config import Config
    from brandvoice_mcp.storage.chromadb import VoiceStore


class DeleteSamplesParams(BaseModel):
    """Validated MCP inputs for ``delete_samples``."""

    model_config = ConfigDict(extra="forbid")

    sample_ids: list[str] | None = None
    all: bool = False

    @model_validator(mode="after")
    def require_mode(self) -> DeleteSamplesParams:
        has_ids = bool(self.sample_ids)
        if self.all and has_ids:
            raise ValueError("When all is true, sample_ids must be empty or omitted")
        if not self.all and not has_ids:
            raise ValueError("Provide non-empty sample_ids or set all to true")
        return self


async def delete_samples(
    *,
    sample_ids: list[str] | None,
    delete_all: bool,
    config: Config,
    store: VoiceStore,
) -> DeleteSamplesResult:
    """Delete samples by ID and/or clear the entire collection, then refresh profile state."""
    DeleteSamplesParams(sample_ids=sample_ids, all=delete_all)

    if delete_all:
        deleted = await store.delete_all_writing_samples_async()
        await refresh_learned_profile_after_samples_change(config=config, store=store)
        return DeleteSamplesResult(deleted_count=deleted, remaining_count=0)

    assert sample_ids is not None
    deleted, remaining = await store.delete_samples_by_ids_async(sample_ids)
    await refresh_learned_profile_after_samples_change(config=config, store=store)
    return DeleteSamplesResult(deleted_count=deleted, remaining_count=remaining)
