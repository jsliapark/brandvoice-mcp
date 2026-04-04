"""MCP server definition with all brand-voice tools registered."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any, Literal, TypeVar

from mcp.server import FastMCP

from brandvoice_mcp.config import Config, load_config
from brandvoice_mcp.storage.chromadb import VoiceStore
from brandvoice_mcp.storage.embeddings import EmbeddingService
from brandvoice_mcp.tools import (
    alignment as alignment_tool,
    delete_samples as delete_samples_tool,
    guidelines as guidelines_tool,
    ingest as ingest_tool,
    profile as profile_tool,
    samples as samples_tool,
    voice_context as voice_context_tool,
)

logger = logging.getLogger("brandvoice-mcp")

# Per-tool ceiling so a stuck embedding/LLM call cannot block the MCP session indefinitely.
_TOOL_TIMEOUT_SEC = 300

_T = TypeVar("_T")


async def _call_tool(name: str, coro: Awaitable[_T]) -> _T:
    try:
        return await asyncio.wait_for(coro, timeout=_TOOL_TIMEOUT_SEC)
    except TimeoutError as exc:
        logger.error("Tool %s exceeded %ss timeout", name, _TOOL_TIMEOUT_SEC)
        raise RuntimeError(
            f"{name} timed out after {_TOOL_TIMEOUT_SEC}s. Try smaller content, fewer samples, "
            "or check network/API availability."
        ) from exc
    except EnvironmentError:
        raise
    except Exception as exc:
        logger.exception("%s failed", name)
        raise RuntimeError(f"{name} failed: {exc}") from exc


def create_server() -> tuple[FastMCP, Config, VoiceStore, EmbeddingService]:
    """Create and configure the MCP server with all tools registered."""
    config = load_config()
    store = VoiceStore(config)
    embedding_service = EmbeddingService(config)
    mcp = FastMCP("brandvoice-mcp")

    @mcp.tool()
    async def ingest_samples(
        content: str,
        source: Literal["blog", "social", "email", "doc", "other"] = "other",
        title: str | None = None,
        url: str | None = None,
    ) -> dict[str, Any]:
        """Ingest a writing sample to learn your voice.

        Analyzes the content for style patterns (sentence structure, vocabulary,
        tone markers, technical depth), chunks it, generates embeddings, and
        stores everything in your local voice profile.

        The more samples you feed, the better the voice matching becomes.
        Feed it blog posts, tweets, emails, docs — anything that represents
        how you write.
        """
        result = await _call_tool(
            "ingest_samples",
            ingest_tool.ingest_samples(
                content=content,
                source=source,
                title=title,
                url=url,
                config=config,
                store=store,
                embeddings=embedding_service,
            ),
        )
        return result.model_dump(mode="json")

    @mcp.tool()
    async def get_voice_context(
        task: str,
        platform: Literal["blog", "linkedin", "twitter", "email", "general"] = "general",
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Get your voice context for a writing task.

        Retrieves your voice profile and the most relevant past writing samples
        for the given task. Returns a ready-to-use prompt injection string that
        makes any LLM generate text matching your style.

        Call this BEFORE generating any content. Inject the returned
        prompt_injection into your system prompt or prepend it to your request.
        """
        result = await _call_tool(
            "get_voice_context",
            voice_context_tool.get_voice_context(
                task=task,
                platform=platform,
                top_k=top_k,
                config=config,
                store=store,
                embeddings=embedding_service,
            ),
        )
        return result.model_dump(mode="json")

    @mcp.tool()
    async def set_guidelines(
        pillars: list[str] | None = None,
        tone: dict[str, float] | None = None,
        preferred_vocabulary: list[str] | None = None,
        avoided_vocabulary: list[str] | None = None,
        topics: list[str] | None = None,
        custom_instructions: str | None = None,
    ) -> dict[str, Any]:
        """Set or update your brand voice guidelines.

        These explicit guidelines complement the style learned from your
        writing samples. Use this to override auto-detected patterns or
        add preferences that aren't visible in your existing content.

        All fields are optional — only provided fields are updated.
        """
        result = await _call_tool(
            "set_guidelines",
            guidelines_tool.set_guidelines(
                store=store,
                pillars=pillars,
                tone=tone,
                preferred_vocabulary=preferred_vocabulary,
                avoided_vocabulary=avoided_vocabulary,
                topics=topics,
                custom_instructions=custom_instructions,
            ),
        )
        return result.model_dump(mode="json")

    @mcp.tool()
    async def check_alignment(
        content: str,
        platform: Literal["blog", "linkedin", "twitter", "email", "general"] = "general",
    ) -> dict[str, Any]:
        """Check how well content matches your voice profile.

        Uses Claude with your stored style and sample snippets when configured;
        falls back to fast heuristics if the API is unavailable. Returns a 0-100
        alignment score with drift flags and rewrite hints. Use as a quality
        gate before publishing.
        """
        result = await _call_tool(
            "check_alignment",
            alignment_tool.check_alignment(
                content=content,
                platform=platform,
                config=config,
                store=store,
            ),
        )
        return result.model_dump(mode="json")

    @mcp.tool()
    async def get_profile() -> dict[str, Any]:
        """Get your complete voice profile.

        Returns learned style patterns, explicit guidelines, sample counts,
        and a human-readable style summary. Useful for reviewing what the
        system has learned about your writing.
        """
        result = await _call_tool(
            "get_profile",
            profile_tool.get_profile(store=store),
        )
        return result.model_dump(mode="json")

    @mcp.tool()
    async def list_samples(
        source: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """List your ingested writing samples.

        Browse samples with optional filtering by source type.
        Use this to review what's been ingested or find samples to remove.
        """
        result = await _call_tool(
            "list_samples",
            samples_tool.list_samples(
                store=store,
                source=source,
                limit=limit,
                offset=offset,
            ),
        )
        return result.model_dump(mode="json")

    @mcp.tool()
    async def delete_samples(
        sample_ids: list[str] | None = None,
        all: bool = False,
    ) -> dict[str, Any]:
        """Delete ingested writing samples by Chroma document ID or clear the entire collection.

        Pass ``sample_ids`` (from ``list_samples``) to remove specific chunks, or set ``all``
        to true to wipe every stored sample. When ``all`` is true, ``sample_ids`` must be
        omitted or empty. After deletion, the learned voice profile is regenerated from any
        remaining samples, or reset to the default empty state if none remain.
        """
        result = await _call_tool(
            "delete_samples",
            delete_samples_tool.delete_samples(
                sample_ids=sample_ids,
                delete_all=all,
                config=config,
                store=store,
            ),
        )
        return result.model_dump(mode="json")

    return mcp, config, store, embedding_service


def run_server() -> None:
    """Start the MCP server over stdio."""
    mcp, config, _store, _embeddings = create_server()
    logger.info("brandvoice-mcp server starting (data_dir=%s)", config.data_dir)
    mcp.run(transport="stdio")
