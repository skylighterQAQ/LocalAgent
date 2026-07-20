"""
Workspace Context – process-level active workspace registry.

Tools (filesystem, computer screenshots, etc.) consult this module to
know which workspace directory should host generated files. Setting an
active workspace from ``main.py`` allows downstream tools to resolve
relative paths into the workspace directory automatically.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from local_agent.cli.workspace.manager import WorkspaceManager

logger = logging.getLogger(__name__)

_active_workspace: Optional["WorkspaceManager"] = None


def set_active_workspace(workspace: Optional["WorkspaceManager"]) -> None:
    """Set (or clear) the process-wide active workspace."""
    global _active_workspace
    _active_workspace = workspace
    if workspace is not None:
        try:
            logger.debug("Active workspace set: %s @ %s", workspace.name, workspace.default_dir)
        except Exception:
            pass


def get_active_workspace() -> Optional["WorkspaceManager"]:
    """Return the active workspace if one has been set."""
    return _active_workspace


def get_active_workspace_dir() -> Optional[Path]:
    """Return the active workspace's default_dir as a ``Path`` (or None)."""
    if _active_workspace is None:
        return None
    try:
        return _active_workspace.default_dir
    except Exception:
        return None


def resolve_workspace_path(path: str) -> str:
    """
    Resolve *path* relative to the active workspace.

    Rules:
      - Absolute paths are returned unchanged.
      - ``~`` paths are expanded.
      - If no workspace is active, *path* is returned unchanged.
      - Otherwise, the path is joined with the workspace ``default_dir``.
    """
    if not path:
        return path
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    ws_dir = get_active_workspace_dir()
    if ws_dir is None:
        return str(p)
    resolved = (ws_dir / p).resolve()
    return str(resolved)
