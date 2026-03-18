"""Configuration management for brandvoice-mcp.

Handles data directory setup (~/.brandvoice), API key resolution,
and runtime configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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

    # Derived paths
    @property
    def chromadb_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chromadb_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    """Build a Config from environment variables.

    Environment variables:
        ANTHROPIC_API_KEY (required): Anthropic API key for LLM + embeddings.
        BRANDVOICE_DATA_DIR: Override default data directory (~/.brandvoice).
        BRANDVOICE_EMBEDDING_MODEL: Embedding model name (default: voyage-3).
        BRANDVOICE_ANALYSIS_MODEL: LLM model for style analysis (default: claude-sonnet-4-20250514).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable is required. "
            "Set it in your shell or in your MCP client configuration."
        )

    raw_dir = os.environ.get("BRANDVOICE_DATA_DIR", "")
    data_dir = Path(raw_dir).expanduser() if raw_dir else _DEFAULT_DATA_DIR

    config = Config(
        data_dir=data_dir,
        anthropic_api_key=api_key,
        embedding_model=os.environ.get("BRANDVOICE_EMBEDDING_MODEL", "voyage-3"),
        analysis_model=os.environ.get("BRANDVOICE_ANALYSIS_MODEL", "claude-sonnet-4-20250514"),
        profile_reanalysis_threshold=int(
            os.environ.get(
                "BRANDVOICE_PROFILE_THRESHOLD",
                str(_PROFILE_REANALYSIS_THRESHOLD),
            )
        ),
        chunk_target_tokens=_CHUNK_TARGET_TOKENS,
        chunk_min_tokens=_CHUNK_MIN_TOKENS,
        chunk_max_tokens=_CHUNK_MAX_TOKENS,
    )
    config.ensure_directories()
    return config
