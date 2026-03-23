"""Aggregate voice profile persistence in ``~/.brandvoice/profile.json``.

Learned style and explicit guidelines are stored here. ChromaDB holds only
embedded writing samples.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def default_profile_state() -> dict[str, Any]:
    return {
        "learned_style": None,
        "explicit_guidelines": None,
        "last_updated": None,
    }


def load_profile_state(path: Path) -> dict[str, Any]:
    """Load profile JSON from disk, or return empty structure if missing."""
    if not path.exists():
        return default_profile_state()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_profile_state()
    out = default_profile_state()
    out["learned_style"] = data.get("learned_style")
    out["explicit_guidelines"] = data.get("explicit_guidelines")
    out["last_updated"] = data.get("last_updated")
    return out


def save_profile_state(path: Path, state: dict[str, Any]) -> None:
    """Atomically write profile state (indent for human inspection)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {**default_profile_state(), **state}
    if state.get("last_updated") is None:
        state["last_updated"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
