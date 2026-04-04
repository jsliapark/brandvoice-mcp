"""ChromaDB storage layer for writing samples and voice profiles.

Uses ChromaDB's persistent client so everything lives on disk under the
configured data directory. No external database service required.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import chromadb
from chromadb.config import Settings

from brandvoice_mcp.config import Config
from brandvoice_mcp.storage.profile_json import (
    default_profile_state,
    load_profile_state,
    save_profile_state,
)

WRITING_SAMPLES_COLLECTION = "writing_samples"
VOICE_PROFILE_COLLECTION = "voice_profile"
GUIDELINES_DOC_ID = "explicit_guidelines"
LEARNED_STYLE_DOC_ID = "learned_style"


class VoiceStore:
    """Manages ChromaDB collections for brand-voice data."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._profile_json_path = config.profile_json_path
        self._client = chromadb.PersistentClient(
            path=str(config.chromadb_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._samples = self._client.get_or_create_collection(
            name=WRITING_SAMPLES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._profile = self._client.get_or_create_collection(
            name=VOICE_PROFILE_COLLECTION,
        )

    # ── Writing samples ──────────────────────────────────────────

    def add_samples(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: dict[str, Any],
    ) -> list[str]:
        """Store chunked writing samples with their embeddings.

        Returns the list of generated IDs.
        """
        ids = [str(uuid.uuid4()) for _ in chunks]
        now = datetime.now(timezone.utc).isoformat()
        metadatas = [
            {
                "source": metadata.get("source", "other"),
                "title": metadata.get("title", ""),
                "url": metadata.get("url", ""),
                "ingested_at": now,
                **{
                    k: v
                    for k, v in metadata.get("style_markers", {}).items()
                    if isinstance(v, (str, int, float, bool))
                },
            }
            for _ in chunks
        ]
        self._samples.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return ids

    def query_samples(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve the most similar writing samples."""
        where = {"source": source_filter} if source_filter else None
        results = self._samples.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.total_samples or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        samples: list[dict[str, Any]] = []
        if not results["documents"] or not results["documents"][0]:
            return samples

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1.0 - dist  # cosine distance → similarity
            samples.append(
                {
                    "content": doc,
                    "source": meta.get("source", "other"),
                    "title": meta.get("title") or None,
                    "similarity": round(max(0.0, min(1.0, similarity)), 4),
                }
            )
        return samples

    def list_samples(
        self,
        source: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List stored samples with optional source filter.

        Returns (samples, total_count).
        """
        where = {"source": source} if source else None
        total = self._samples.count()

        results = self._samples.get(
            where=where,
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )

        entries: list[dict[str, Any]] = []
        if results["ids"]:
            for id_, doc, meta in zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
            ):
                entries.append(
                    {
                        "id": id_,
                        "content_preview": (doc or "")[:200],
                        "source": meta.get("source", "other"),
                        "title": meta.get("title") or None,
                        "ingested_at": meta.get("ingested_at"),
                    }
                )
        return entries, total

    def delete_samples_by_ids(self, ids: list[str]) -> tuple[int, int]:
        """Delete stored chunks with the given Chroma document IDs.

        Missing IDs are ignored. Returns ``(deleted_count, remaining_count)``.
        """
        if not ids:
            return 0, self.total_samples
        results = self._samples.get(ids=ids)
        existing = [i for i in (results.get("ids") or []) if i]
        if not existing:
            return 0, self.total_samples
        self._samples.delete(ids=existing)
        return len(existing), self.total_samples

    def delete_all_writing_samples(self) -> int:
        """Remove every document in the writing-samples collection. Returns how many were removed."""
        n = self.total_samples
        self._client.delete_collection(WRITING_SAMPLES_COLLECTION)
        self._samples = self._client.get_or_create_collection(
            name=WRITING_SAMPLES_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        return n

    def reset_profile_to_default(self) -> None:
        """Reset ``profile.json`` to the default empty state (no learned style or guidelines)."""
        save_profile_state(self._profile_json_path, default_profile_state())

    @property
    def total_samples(self) -> int:
        return self._samples.count()

    def get_sample_snippets(
        self,
        limit: int = 3,
        max_chars_per_sample: int = 1200,
    ) -> list[str]:
        """Return truncated full text of up to ``limit`` stored chunks for LLM context."""
        results = self._samples.get(limit=limit, include=["documents"])
        docs = results.get("documents") or []
        out: list[str] = []
        for doc in docs:
            if doc:
                out.append(doc[:max_chars_per_sample])
        return out

    def get_corpus_excerpts(
        self,
        max_chunks: int = 40,
        max_chars_per_chunk: int = 2000,
        max_total_chars: int = 48_000,
    ) -> list[str]:
        """Return truncated chunk texts for corpus-level style aggregation (LLM context)."""
        cap = min(max_chunks, self.total_samples or 0)
        if cap == 0:
            return []
        results = self._samples.get(limit=cap, include=["documents"])
        docs = [d for d in (results.get("documents") or []) if d]
        out: list[str] = []
        total = 0
        for doc in docs:
            piece = (doc or "")[:max_chars_per_chunk]
            if total + len(piece) > max_total_chars:
                break
            out.append(piece)
            total += len(piece)
        return out

    def sources_breakdown(self) -> dict[str, int]:
        """Count samples per source type."""
        results = self._samples.get(include=["metadatas"])
        breakdown: dict[str, int] = {}
        if results["metadatas"]:
            for meta in results["metadatas"]:
                src = meta.get("source", "other")
                breakdown[src] = breakdown.get(src, 0) + 1
        return breakdown

    # ── Voice profile (profile.json; Chroma collection only for legacy migration) ─

    def _ensure_profile_file_from_legacy_chroma(self) -> None:
        """If ``profile.json`` is missing, migrate learned style + guidelines from Chroma."""
        if self._profile_json_path.exists():
            return
        learned = self._get_profile_doc(LEARNED_STYLE_DOC_ID)
        guidelines = self._get_profile_doc(GUIDELINES_DOC_ID)
        if not learned and not guidelines:
            return
        save_profile_state(
            self._profile_json_path,
            {
                "learned_style": learned,
                "explicit_guidelines": guidelines,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _load_profile_state(self) -> dict[str, Any]:
        self._ensure_profile_file_from_legacy_chroma()
        return load_profile_state(self._profile_json_path)

    def save_learned_style(self, style_data: dict[str, Any]) -> None:
        """Persist the aggregate learned style profile to ``profile.json``."""
        state = self._load_profile_state()
        state["learned_style"] = style_data
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        save_profile_state(self._profile_json_path, state)

    def get_learned_style(self) -> dict[str, Any] | None:
        """Retrieve the learned style profile from ``profile.json``."""
        return self._load_profile_state().get("learned_style")

    def save_guidelines(self, guidelines: dict[str, Any]) -> None:
        """Persist explicit brand voice guidelines to ``profile.json``."""
        state = self._load_profile_state()
        state["explicit_guidelines"] = guidelines
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
        save_profile_state(self._profile_json_path, state)

    def get_guidelines(self) -> dict[str, Any] | None:
        """Retrieve explicit guidelines from ``profile.json``."""
        g = self._load_profile_state().get("explicit_guidelines")
        return g if g else None

    def get_profile_last_updated(self) -> datetime | None:
        """Last time profile.json was updated."""
        raw = self._load_profile_state().get("last_updated")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(str(raw))
        except (TypeError, ValueError):
            return None

    # ── Internal helpers ─────────────────────────────────────────

    def _upsert_profile_doc(
        self, doc_id: str, document: str, metadata: dict[str, Any]
    ) -> None:
        existing = self._profile.get(ids=[doc_id])
        if existing["ids"]:
            self._profile.update(ids=[doc_id], documents=[document], metadatas=[metadata])
        else:
            self._profile.add(ids=[doc_id], documents=[document], metadatas=[metadata])

    def _get_profile_doc(self, doc_id: str) -> dict[str, Any] | None:
        results = self._profile.get(ids=[doc_id], include=["documents"])
        if not results["ids"]:
            return None
        raw = results["documents"][0]
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    # ── Async wrappers (Chroma sync I/O off the event loop) ─────────────

    async def add_samples_async(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        metadata: dict[str, Any],
    ) -> list[str]:
        return await asyncio.to_thread(self.add_samples, chunks, embeddings, metadata)

    async def query_samples_async(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        source_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self.query_samples, query_embedding, top_k, source_filter
        )

    async def list_samples_async(
        self,
        source: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        return await asyncio.to_thread(self.list_samples, source, limit, offset)

    async def delete_samples_by_ids_async(self, ids: list[str]) -> tuple[int, int]:
        return await asyncio.to_thread(self.delete_samples_by_ids, ids)

    async def delete_all_writing_samples_async(self) -> int:
        return await asyncio.to_thread(self.delete_all_writing_samples)

    async def reset_profile_to_default_async(self) -> None:
        await asyncio.to_thread(self.reset_profile_to_default)

    async def sample_count_async(self) -> int:
        return await asyncio.to_thread(lambda: self.total_samples)

    async def get_sample_snippets_async(
        self,
        limit: int = 3,
        max_chars_per_sample: int = 1200,
    ) -> list[str]:
        return await asyncio.to_thread(
            self.get_sample_snippets, limit, max_chars_per_sample
        )

    async def get_corpus_excerpts_async(
        self,
        max_chunks: int = 40,
        max_chars_per_chunk: int = 2000,
        max_total_chars: int = 48_000,
    ) -> list[str]:
        return await asyncio.to_thread(
            self.get_corpus_excerpts,
            max_chunks,
            max_chars_per_chunk,
            max_total_chars,
        )

    async def get_learned_style_async(self) -> dict[str, Any] | None:
        return await asyncio.to_thread(self.get_learned_style)

    async def save_learned_style_async(self, style_data: dict[str, Any]) -> None:
        await asyncio.to_thread(self.save_learned_style, style_data)

    async def get_guidelines_async(self) -> dict[str, Any] | None:
        return await asyncio.to_thread(self.get_guidelines)

    async def save_guidelines_async(self, guidelines: dict[str, Any]) -> None:
        await asyncio.to_thread(self.save_guidelines, guidelines)

    async def sources_breakdown_async(self) -> dict[str, int]:
        return await asyncio.to_thread(self.sources_breakdown)

    async def get_profile_last_updated_async(self) -> datetime | None:
        return await asyncio.to_thread(self.get_profile_last_updated)
