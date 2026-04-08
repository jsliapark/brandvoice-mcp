# brandvoice-mcp
[![PyPI version](https://img.shields.io/pypi/v/brandvoice-mcp.svg)](https://pypi.org/project/brandvoice-mcp/)
[![Python 3.11](https://img.shields.io/pypi/pyversions/brandvoice-mcp.svg)](https://pypi.org/project/brandvoice-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An MCP server that learns your writing style and makes every AI client sound like you.

## Quick start

### Install

```bash
pip install brandvoice-mcp
```

### Configure Claude Desktop

Add to your `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "brandvoice": {
      "command": "python",
      "args": ["-m", "brandvoice_mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "OPENAI_API_KEY": "sk-..."
      }
    }
  }
}
```

Set **`ANTHROPIC_API_KEY`** and **`OPENAI_API_KEY`** in the `env` block (both required for normal use: Claude for analysis/alignment, OpenAI for chunk embeddings). The server will exit with a clear error if either is missing.

### Teach it your voice

In Claude Desktop, ask the model to use **`ingest_samples`** with real writing:

> Use the `ingest_samples` tool to learn my writing style from this blog post: [paste content]

The server chunks the text, stores embeddings in **ChromaDB**, and (for samples of about **50+ words**) runs **LLM style analysis**. Shorter snippets are still stored for retrieval but skip style analysis to avoid unreliable profiles.

### Write in your voice

Before any writing task, call **`get_voice_context`** with your task and platform. The returned **`prompt_injection`** is wrapped in `<voice_context>...</voice_context>` — prepend it to your request or system prompt.

> Use `get_voice_context` for a LinkedIn post about React performance, then write it in my voice.

### Check alignment

After drafting text, call **`check_alignment`** with the draft. You get a 0–100 score, drift flags, and rewrite hints against your stored profile and samples.

> Use `check_alignment` on this draft: [paste text]

## Tools reference

| Tool | Description |
|------|-------------|
| `ingest_samples` | Ingest writing; chunk, embed, and update style profile when thresholds are met |
| `get_voice_context` | Voice guidelines, similar samples, and `prompt_injection` for a task |
| `set_guidelines` | Merge explicit brand voice rules (pillars, tone, vocabulary, etc.) |
| `check_alignment` | Score how well content matches your voice |
| `get_profile` | Full profile: learned style (including `profile_source`), guidelines, counts |
| `list_samples` | Paginated list of ingested samples (each row includes the Chroma document `id` for `delete_samples`) |
| `delete_samples` | Delete samples by `sample_ids` (from `list_samples`) or set `all` to true to clear the collection; refreshes or resets the learned profile |

## How it works

**Ingestion:** Text is split into chunks, embedded with **OpenAI** (default `text-embedding-3-small`), and stored in a local **ChromaDB** collection (`writing_samples`) for similarity search. The aggregate **learned style** and **explicit guidelines** live in **`~/.brandvoice/profile.json`** (human-readable, separate from vectors) so a vector DB issue does not silently wipe your profile alongside embeddings.

**Style analysis:** For sufficiently long samples, Claude analyzes tone and patterns (including humor, technical depth, and warmth scores used in `get_voice_context`). If the API fails, a **heuristic** fallback runs; `profile_source` records `"llm"` vs `"heuristic"`. After enough stored chunks (see `BRANDVOICE_PROFILE_THRESHOLD`), each qualifying ingest **re-merges the corpus** via Claude (`prompts/corpus_aggregate.md`) into a single aggregate profile; on failure, the latest per-sample LLM snapshot is used when available.

**Writing assistance:** For a task, the server retrieves your profile and the top similar chunks, then builds **`prompt_injection`** from markdown templates under `brandvoice_mcp/prompts/`.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key (style analysis, alignment) |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key (chunk embeddings; Anthropic has no embeddings API) |
| `BRANDVOICE_DATA_DIR` | `~/.brandvoice` | Data directory (`profile.json`, Chroma persistence) |
| `BRANDVOICE_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model name |
| `BRANDVOICE_ANALYSIS_MODEL` | `claude-sonnet-4-6` | Model for style analysis |
| `BRANDVOICE_PROFILE_THRESHOLD` | `5` | Minimum stored samples before aggregate profile can update after an LLM-analyzed ingest |

> **Model deprecation:** If the default `claude-sonnet-4-6` is deprecated or unavailable in your region, set `BRANDVOICE_ANALYSIS_MODEL` to a supported model ID (e.g. `claude-opus-4-6` or `claude-haiku-4-5-20251001`). Claude 4 model IDs use no date suffix; check [Anthropic's model documentation](https://docs.anthropic.com/en/docs/about-claude/models) for the current list.

## Limitations

- **Single client:** Designed for one MCP client at a time. Multiple clients sharing the same `~/.brandvoice` directory may hit SQLite/Chroma lock errors.
- **API costs:** Style analysis and alignment use **Anthropic**; chunk embeddings use **OpenAI**. Each `ingest_samples` and `check_alignment` consumes tokens; budget accordingly.

## Requirements

- Python **3.11+**
- **Anthropic API key** (`ANTHROPIC_API_KEY`) — LLM calls
- **OpenAI API key** (`OPENAI_API_KEY`) — embeddings for ChromaDB similarity search

## Development

```bash
git clone https://github.com/jsliapark/brandvoice-mcp.git
cd brandvoice-mcp
pip install -e ".[dev]"
pytest
```

## Architecture (overview)

```
MCP client (Claude Desktop, Cursor, …)
        │ stdio
        ▼
  brandvoice-mcp server
        │
        ├── profile.json     ← aggregate learned style + explicit guidelines
        └── ChromaDB         ← writing_samples (embeddings + chunks)
```

### Manual testing in a terminal

The server speaks **JSON-RPC on stdin/stdout**. When you run `python -m brandvoice_mcp`, it should **block** until the client disconnects or you press **Ctrl+C** — there is no interactive prompt.

- **Do not type** in that terminal while the server is running; random text is not valid JSON-RPC and you will see errors like `Invalid JSON` / `JSONRPCMessage` validation errors.
- **Do not** run two copies of the server on the same stdio session.
- If you use **Cursor / Claude Desktop** with this project, let **only the IDE** spawn the process — don’t also run `python -m brandvoice_mcp` in a terminal unless you are debugging with a real MCP client attached.

If `python -m brandvoice_mcp` crashes with `Server` has no attribute `tool`, your checkout is on the **old** low-level `Server` API — use **`FastMCP`** (`from mcp.server import FastMCP`) and sync `__main__.py` to call `run_server()` without `asyncio.run` (see current `server.py` on `main`).

For a local run, export both **`ANTHROPIC_API_KEY`** and **`OPENAI_API_KEY`** (see Configuration above).

## License

MIT — see [LICENSE](LICENSE).
