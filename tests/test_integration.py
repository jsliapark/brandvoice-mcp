"""MCP stdio integration: subprocess server + real ClientSession protocol."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.asyncio
async def test_stdio_server_lists_tools_and_list_samples(tmp_path: Path) -> None:
    data_dir = tmp_path / "bv"
    data_dir.mkdir()
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = "sk-test-mcp-integration"
    env["BRANDVOICE_EMBEDDING_MODEL"] = "test"
    env["BRANDVOICE_DATA_DIR"] = str(data_dir)
    src = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "brandvoice_mcp"],
        env=env,
        cwd=str(REPO_ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            names = {t.name for t in listed.tools}
            assert names == {
                "ingest_samples",
                "get_voice_context",
                "set_guidelines",
                "check_alignment",
                "get_profile",
                "list_samples",
                "delete_samples",
            }
            res = await session.call_tool("list_samples", {})
            assert res.isError is False
