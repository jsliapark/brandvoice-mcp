# brandvoice-mcp

An MCP server that learns your writing style and makes every AI client sound like you.

## The Problem

AI-generated text sounds generic. Every time you use Claude Desktop, ChatGPT, or any AI writing tool, you have to manually describe your voice — or accept output that doesn't sound like you. Brand voice consistency requires pasting style guides into every conversation.

**brandvoice-mcp** solves this. Connect it once, and your voice follows you across every MCP-compatible AI tool.

## How It Works

1. **Feed it your writing** — Blog posts, tweets, emails, docs. The more, the better.
2. **It learns your style** — Sentence structure, vocabulary, tone, rhetorical patterns.
3. **Every AI sounds like you** — Any MCP client calls `get_voice_context` and gets a ready-to-use style injection.

## Quick Start

### Install

```bash
pip install brandvoice-mcp
```

### Configure Claude Desktop

Add to your Claude Desktop MCP config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "brandvoice": {
      "command": "python",
      "args": ["-m", "brandvoice_mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Or use uvx (no install needed)

```json
{
  "mcpServers": {
    "brandvoice": {
      "command": "uvx",
      "args": ["brandvoice-mcp"],
      "env": {
        "ANTHROPIC_API_KEY": "your-key-here"
      }
    }
  }
}
```

### Teach It Your Voice

In Claude Desktop, say:

> "Use the ingest_samples tool to learn my writing style from this blog post: [paste your content]"

### Write in Your Voice

> "Use get_voice_context for a LinkedIn post about React performance, then write it in my voice."

## Tools

| Tool | Description |
|------|-------------|
| `ingest_samples` | Feed writing samples to learn your voice |
| `get_voice_context` | Get voice context + prompt injection for any writing task |
| `set_guidelines` | Explicitly configure brand voice (pillars, tone, vocabulary) |
| `check_alignment` | Score content against your voice profile (0-100) |
| `get_profile` | View your complete voice profile |
| `list_samples` | Browse ingested writing samples |

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `BRANDVOICE_DATA_DIR` | `~/.brandvoice` | Where voice data is stored |
| `BRANDVOICE_EMBEDDING_MODEL` | `voyage-3` | Embedding model |
| `BRANDVOICE_ANALYSIS_MODEL` | `claude-sonnet-4-20250514` | LLM for style analysis |

## Architecture

```
┌──────────────────────────────────────────┐
│  Any MCP Client                          │
│  (Claude Desktop, Cursor, custom agents) │
└──────────────┬───────────────────────────┘
               │ MCP Protocol (stdio)
┌──────────────▼───────────────────────────┐
│  brandvoice-mcp server                   │
│                                          │
│  ┌─────────────┐  ┌──────────────────┐   │
│  │ Style       │  │ Voice Profile    │   │
│  │ Analyzer    │  │ Manager          │   │
│  └──────┬──────┘  └────────┬─────────┘   │
│         │                  │             │
│  ┌──────▼──────────────────▼─────────┐   │
│  │  ChromaDB (local, file-based)     │   │
│  │  - writing_samples collection     │   │
│  │  - voice_profile collection       │   │
│  └───────────────────────────────────┘   │
└──────────────────────────────────────────┘
```

## Development

```bash
git clone https://github.com/jinsungpark/brandvoice-mcp.git
cd brandvoice-mcp
pip install -e ".[dev]"
pytest
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
