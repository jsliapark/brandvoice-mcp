"""Pydantic models for all brandvoice-mcp tool inputs and outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── Shared / reusable ────────────────────────────────────────────────


class ToneConfig(BaseModel):
    formality: float = Field(ge=0, le=1, description="0=casual, 1=formal")
    humor: float = Field(ge=0, le=1, description="0=serious, 1=humorous")
    technical_depth: float = Field(ge=0, le=1, description="0=beginner, 1=expert")
    warmth: float = Field(ge=0, le=1, description="0=detached, 1=warm")


StyleSource = Literal["llm", "heuristic"]


class StyleSnapshot(BaseModel):
    avg_sentence_length: float
    vocabulary_richness: float = Field(ge=0, le=1)
    formality_score: float = Field(ge=0, le=1)
    #: 0 = serious, 1 = humorous (from LLM analysis or heuristics).
    humor: float = Field(default=0.5, ge=0, le=1)
    #: 0 = beginner-friendly, 1 = expert / dense (from LLM analysis or heuristics).
    technical_depth: float = Field(default=0.5, ge=0, le=1)
    #: 0 = detached, 1 = warm / personal (from LLM analysis or heuristics).
    warmth: float = Field(default=0.5, ge=0, le=1)
    dominant_tone: str
    rhetorical_patterns: list[str] = Field(default_factory=list)
    profile_source: StyleSource = Field(
        default="heuristic",
        description="Whether metrics came from Claude (llm) or statistical fallback (heuristic).",
    )


# ── ingest_samples ───────────────────────────────────────────────────


SourceType = Literal["blog", "social", "email", "doc", "other"]


class IngestResult(BaseModel):
    samples_stored: int
    total_samples: int
    voice_profile_updated: bool
    style_snapshot: StyleSnapshot
    analysis_note: str | None = Field(
        default=None,
        description="Optional note e.g. when sample was too short for LLM style analysis.",
    )


# ── get_voice_context ────────────────────────────────────────────────


PlatformType = Literal["blog", "linkedin", "twitter", "email", "general"]


class VoiceSample(BaseModel):
    content: str
    source: str
    similarity: float = Field(ge=0, le=1)
    title: str | None = None


class VoiceContext(BaseModel):
    voice_guidelines: str
    tone_profile: ToneConfig
    similar_samples: list[VoiceSample]
    vocabulary: dict[str, list[str]]  # {"preferred": [...], "avoided": [...]}
    prompt_injection: str


# ── set_guidelines ───────────────────────────────────────────────────


class GuidelinesResult(BaseModel):
    updated_fields: list[str]
    current_guidelines: dict


# ── check_alignment ──────────────────────────────────────────────────


class DriftFlag(BaseModel):
    category: str
    issue: str
    severity: Literal["low", "medium", "high"]


class AlignmentResult(BaseModel):
    alignment_score: int = Field(ge=0, le=100)
    verdict: Literal[
        "on_brand",
        "minor_drift",
        "significant_drift",
        "off_brand",
        "unknown",
    ]
    drift_flags: list[DriftFlag]
    suggestions: list[str]
    rewrite_hints: str


# ── get_profile ──────────────────────────────────────────────────────


class VoiceProfile(BaseModel):
    learned_style: StyleSnapshot | None = None
    explicit_guidelines: dict | None = None
    total_samples: int = 0
    sources_breakdown: dict[str, int] = Field(default_factory=dict)
    style_summary: str = ""
    last_updated: datetime | None = None


# ── list_samples ─────────────────────────────────────────────────────


class SampleEntry(BaseModel):
    id: str
    content_preview: str
    source: str
    title: str | None = None
    ingested_at: datetime | None = None


class SamplesList(BaseModel):
    samples: list[SampleEntry]
    total: int
    offset: int
    limit: int
