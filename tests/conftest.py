"""Shared fixtures for brandvoice-mcp tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from brandvoice_mcp.config import Config
from brandvoice_mcp.storage.chromadb import VoiceStore
from brandvoice_mcp.storage.embeddings import deterministic_embedding


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for test isolation."""
    data_dir = tmp_path / "brandvoice_test"
    data_dir.mkdir()
    return data_dir


@pytest.fixture()
def config(tmp_data_dir: Path) -> Config:
    """Test config pointing to a temporary data directory."""
    cfg = Config(
        data_dir=tmp_data_dir,
        anthropic_api_key="test-key-not-real",
        embedding_model="test",
        analysis_model="test",
        profile_reanalysis_threshold=3,
        chunk_target_tokens=350,
        chunk_min_tokens=50,
        chunk_max_tokens=600,
    )
    cfg.ensure_directories()
    return cfg


@pytest.fixture()
def store(config: Config) -> VoiceStore:
    """VoiceStore backed by a fresh temporary ChromaDB."""
    return VoiceStore(config)


# Style analysis LLM path is covered in tests/test_style_analyzer.py (AsyncAnthropic mock).
class FakeEmbeddingService:
    """Deterministic embedding service for tests — no API calls."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [deterministic_embedding(t) for t in texts]

    async def embed_text(self, text: str) -> list[float]:
        return deterministic_embedding(text)


@pytest.fixture()
def embedding_service() -> FakeEmbeddingService:
    return FakeEmbeddingService()


SAMPLE_BLOG_POST = """\
Building React Components That Scale

When I started building design systems at Netflix, the biggest lesson was this: \
every component should be a contract. Not just a visual element, but a promise \
about behavior, accessibility, and performance.

Here's what I mean. A Button component isn't just a styled <button> tag. It's a \
guarantee that keyboard navigation works, that focus states are visible, that \
loading states don't cause layout shift. When 200 engineers depend on your Button, \
you can't ship regressions.

The pattern that saved us? Compound components with context. Instead of prop \
drilling through 15 props, we composed behavior:

<Tabs>
  <Tabs.List>
    <Tabs.Tab>Overview</Tabs.Tab>
    <Tabs.Tab>Details</Tabs.Tab>
  </Tabs.List>
  <Tabs.Panel>First panel</Tabs.Panel>
  <Tabs.Panel>Second panel</Tabs.Panel>
</Tabs>

Each piece owns its own logic. The parent coordinates through context. Testing \
becomes surgical — test each piece in isolation, test the composition separately.

Three rules I follow now for every component:
1. Props are the public API. Type them ruthlessly.
2. Internal state should be invisible to consumers.
3. If you need more than 10 props, you need composition instead.

This isn't theoretical. These patterns ship at Netflix scale. Try them.
"""

SAMPLE_LINKEDIN_POST = """\
Just shipped something I'm proud of: a CLI tool that runs 4 concurrent AI \
review passes on your git diffs before you push.

It catches bugs I would have missed. Not theoretical bugs — actual logic \
errors in production code.

The stack: Python, Claude API with structured outputs, ChromaDB for context.

What I learned building it: structured outputs from LLMs are a game-changer. \
Instead of parsing free-form text, you get typed, validated responses every time.

If you're building AI tools, stop treating LLM output as strings. Treat it \
as data.

Link in comments. It's called preflight-ai and it's on PyPI.

#AI #DeveloperTools #Python
"""

SAMPLE_TECHNICAL_DOC = """\
preflight-ai Architecture

The system runs four independent review agents concurrently using asyncio. \
Each agent receives the same git diff but evaluates it through a different lens:

1. Logic Agent: Checks for logical errors, edge cases, null handling
2. Security Agent: Scans for injection vulnerabilities, exposed secrets
3. Performance Agent: Identifies N+1 queries, unnecessary re-renders
4. Style Agent: Enforces coding conventions and naming patterns

Agent outputs are structured using Pydantic models and Claude's tool_use \
feature. Each finding includes: severity (critical/warning/info), file path, \
line range, description, and a suggested fix.

Results are deduplicated by file+line range, merged into a single report, \
and presented in the terminal with color-coded severity levels.

ChromaDB stores historical review context — past findings, false positive \
markers, and project-specific patterns — so the agents improve over time.
"""
