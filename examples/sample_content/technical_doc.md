# preflight-ai Architecture

The system runs four independent review agents concurrently using asyncio. Each agent receives the same git diff but evaluates it through a different lens:

1. **Logic Agent**: Checks for logical errors, edge cases, null handling
2. **Security Agent**: Scans for injection vulnerabilities, exposed secrets
3. **Performance Agent**: Identifies N+1 queries, unnecessary re-renders
4. **Style Agent**: Enforces coding conventions and naming patterns

Agent outputs are structured using Pydantic models and Claude's `tool_use` feature. Each finding includes: severity (critical/warning/info), file path, line range, description, and a suggested fix.

Results are deduplicated by file+line range, merged into a single report, and presented in the terminal with color-coded severity levels.

ChromaDB stores historical review context — past findings, false positive markers, and project-specific patterns — so the agents improve over time.
