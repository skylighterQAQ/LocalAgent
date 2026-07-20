"""
Workspace State – persist the last-used workspace across launches.

Stores a small JSON file under ``~/.local_agent/state.json`` with
information about the last active workspace so it can be auto-loaded
on the next start.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default state location: ~/.local_agent/state.json
STATE_DIR = Path.home() / ".local_agent"
STATE_FILE = STATE_DIR / "state.json"


def _read_state() -> dict:
    """Read the raw state dict. Returns ``{}`` if file is missing/invalid."""
    try:
        if not STATE_FILE.exists():
            return {}
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("Failed to read workspace state: %s", exc)
        return {}


def _write_state(data: dict) -> None:
    """Write the state dict to disk, creating directories as needed."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning("Failed to write workspace state: %s", exc)


def save_last_workspace(path: str | Path) -> None:
    """Persist *path* as the last-used workspace directory."""
    p = Path(path).expanduser().resolve()
    state = _read_state()
    state["last_workspace"] = str(p)
    _write_state(state)
    logger.debug("Saved last workspace: %s", p)


def load_last_workspace() -> Optional[Path]:
    """Return the most recently used workspace directory, if it still exists."""
    state = _read_state()
    last = state.get("last_workspace")
    if not last:
        return None
    p = Path(last).expanduser()
    if not p.exists():
        logger.debug("Recorded last workspace no longer exists: %s", p)
        return None
    return p


def clear_last_workspace() -> None:
    """Forget the last-used workspace."""
    state = _read_state()
    state.pop("last_workspace", None)
    _write_state(state)
