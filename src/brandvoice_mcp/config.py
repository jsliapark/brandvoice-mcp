"""Configuration management for brandvoice-mcp.

Handles data directory setup (~/.brandvoice), API key resolution,
and runtime configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_DEFAULT_DATA_DIR = Path.home() / ".brandvoice"

_PROFILE_REANALYSIS_THRESHOLD = 5

_CHUNK_TARGET_TOKENS = 350
_CHUNK_MIN_TOKENS = 50
_CHUNK_MAX_TOKENS = 600


@dataclass(frozen=True)
class Config:
    """Immutable runtime configuration resolved from environment variables."""

    data_dir: Path
    anthropic_api_key: str
    embedding_model: str
    analysis_model: str
    profile_reanalysis_threshold: int
    chunk_target_tokens: int
    chunk_min_tokens: int
    chunk_max_tokens: int
    #: OpenAI API key for ``EmbeddingService`` (not used when ``embedding_model`` is ``"test"``).
    openai_api_key: str | None = None

    # Derived paths
    @property
    def chromadb_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    @property
    def profile_json_path(self) -> Path:
        """Aggregate voice profile (learned style + guidelines) as JSON."""
        return self.data_dir / "profile.json"

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chromadb_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """Build a Config from environment variables.

    Environment variables:
        ANTHROPIC_API_KEY (required): Anthropic API key for LLM (style analysis, alignment).
        OPENAI_API_KEY (required for embeddings): OpenAI API key for chunk embeddings
            (default model: text-embedding-3-small). Not required when
            BRANDVOICE_EMBEDDING_MODEL=test (unit tests only).
        BRANDVOICE_DATA_DIR: Override default data directory (~/.brandvoice).
        BRANDVOICE_EMBEDDING_MODEL: OpenAI embedding model (default: text-embedding-3-small).
        BRANDVOICE_ANALYSIS_MODEL: LLM model for style analysis (default: claude-sonnet-4-6).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is required for brandvoice-mcp (LLM style analysis, "
            "and alignment). Set it in your environment or in the MCP "
            "server config under 'env', e.g. "
            '"env": { "ANTHROPIC_API_KEY": "sk-ant-..." }. See README for setup.'
        )

    raw_dir = os.environ.get("BRANDVOICE_DATA_DIR", "")
    data_dir = Path(raw_dir).expanduser() if raw_dir else _DEFAULT_DATA_DIR

    embedding_model = os.environ.get("BRANDVOICE_EMBEDDING_MODEL", "text-embedding-3-small")
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip() or None

    if embedding_model != "test" and not openai_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is required for embeddings (OpenAI text-embedding-3-small by default). "
            "Anthropic does not provide a text embeddings API. Add OPENAI_API_KEY to your MCP "
            "config under 'env', e.g. "
            '"env": { "ANTHROPIC_API_KEY": "sk-ant-...", "OPENAI_API_KEY": "sk-..." }'
        )

    config = Config(
        data_dir=data_dir,
        anthropic_api_key=api_key,
        embedding_model=embedding_model,
        analysis_model=os.environ.get("BRANDVOICE_ANALYSIS_MODEL", "claude-sonnet-4-6"),
        profile_reanalysis_threshold=int(
            os.environ.get(
                "BRANDVOICE_PROFILE_THRESHOLD",
                str(_PROFILE_REANALYSIS_THRESHOLD),
            )
        ),
        chunk_target_tokens=_CHUNK_TARGET_TOKENS,
        chunk_min_tokens=_CHUNK_MIN_TOKENS,
        chunk_max_tokens=_CHUNK_MAX_TOKENS,
        openai_api_key=openai_key,
    )
    config.ensure_directories()
    return config
