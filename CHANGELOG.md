# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-04-08

### Added
- MCP Resources: `brandvoice://profile` and `brandvoice://samples` for read-only state snapshots
- MCP Prompts: `write_in_voice` and `check_my_draft` as reusable conversation starters
- Extended thinking support via `BRANDVOICE_EXTENDED_THINKING` and `BRANDVOICE_THINKING_BUDGET` env vars
- Tool timeout wrapper (`_call_tool`) with 300s ceiling and structured error messages
- Ingest size guard (`_MAX_INGEST_CHARS = 100,000`) to prevent accidental large uploads

### Changed
- Style analysis now uses configurable thinking budget for deeper analysis when enabled

### Fixed
- Sentence splitting in `chunk_content` no longer breaks on abbreviations (`e.g.`, `Dr.`, `i.e.`, `vs.`, `etc.`) or URLs containing dots
