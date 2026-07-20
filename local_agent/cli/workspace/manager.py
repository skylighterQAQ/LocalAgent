"""
Workspace Manager – load, save and manage LocalAgent workspaces.

A workspace is defined by a ``workspace.yaml`` file in the project directory.
The manager handles:
  - Loading workspace config from YAML
  - Saving / initialising workspace configs
  - Resolving file paths relative to the workspace default_dir
  - Providing the active workspace to the rest of LocalAgent
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import yaml

from local_agent.cli.workspace.config import WorkspaceConfig

logger = logging.getLogger(__name__)

# Default workspace config file name
WORKSPACE_FILENAME = "workspace.yaml"


class WorkspaceManager:
    """
    Manages LocalAgent workspace configuration.

    Usage::

        mgr = WorkspaceManager.load("./workspace.yaml")
        path = mgr.resolve_path("output.py")   # → <workspace_dir>/output.py
    """

    def __init__(self, config: WorkspaceConfig, config_path: Optional[Path] = None) -> None:
        self._config = config
        self._config_path = config_path

    # ── Factory methods ───────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path = WORKSPACE_FILENAME) -> Optional["WorkspaceManager"]:
        """
        Load a WorkspaceManager from a workspace.yaml file.
        Returns None if the file does not exist (no workspace configured).
        """
        p = Path(path).expanduser().resolve()
        if not p.exists():
            logger.debug("No workspace file found at: %s", p)
            return None
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            ws_data = data.get("workspace", {})
            config = WorkspaceConfig(**ws_data)
            logger.info("Loaded workspace '%s' from %s", config.name, p)
            return cls(config=config, config_path=p)
        except Exception as exc:
            logger.warning("Failed to load workspace from %s: %s", p, exc)
            return None

    @classmethod
    def from_config(cls, config: WorkspaceConfig) -> "WorkspaceManager":
        """Create a WorkspaceManager directly from a WorkspaceConfig object."""
        return cls(config=config)

    @classmethod
    def init(
        cls,
        directory: str | Path = ".",
        name: str = "default",
        skill: Optional[str] = None,
        description: str = "",
    ) -> "WorkspaceManager":
        """
        Initialise a new workspace in *directory*.

        Creates the workspace directory and writes a ``workspace.yaml`` file.
        Returns the new WorkspaceManager.
        """
        dir_path = Path(directory).expanduser().resolve()
        dir_path.mkdir(parents=True, exist_ok=True)

        config = WorkspaceConfig(
            name=name,
            # Keep the generated file portable: its own directory is the
            # workspace root, wherever the project is later moved.
            default_dir=".",
            terminal_dir="",
            skill=skill,
            description=description or f"LocalAgent workspace: {name}",
        )

        config_path = dir_path / WORKSPACE_FILENAME
        mgr = cls(config=config, config_path=config_path)
        mgr.save()
        logger.info("Initialised workspace '%s' at %s", name, config_path)
        return mgr

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def config(self) -> WorkspaceConfig:
        return self._config

    @property
    def config_path(self) -> Optional[Path]:
        return self._config_path

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def default_dir(self) -> Path:
        return self._resolve_directory(self._config.default_dir)

    @property
    def terminal_dir(self) -> Path:
        return self._resolve_directory(self._config.terminal_dir) if self._config.terminal_dir else self.default_dir

    def _resolve_directory(self, directory: str) -> Path:
        """Resolve a workspace directory relative to its config file.

        Keeping relative paths in ``workspace.yaml`` makes a workspace
        portable when the whole project directory is moved or cloned.
        """
        path = Path(directory).expanduser()
        if not path.is_absolute() and self._config_path is not None:
            path = self._config_path.parent / path
        return path.resolve()

    @property
    def skill(self) -> Optional[str]:
        return self._config.skill

    # ── Path resolution ───────────────────────────────────────────────────────

    def resolve_path(self, relative: str) -> str:
        """
        Resolve a relative file path to the workspace's default_dir.

        If *relative* is already absolute, it is returned unchanged.
        """
        p = Path(relative).expanduser()
        if p.is_absolute():
            return str(p)
        resolved = self.default_dir / p
        return str(resolved)

    def ensure_default_dir(self) -> Path:
        """Create the workspace default_dir if it does not exist."""
        self.default_dir.mkdir(parents=True, exist_ok=True)
        return self.default_dir

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Optional[str | Path] = None) -> Path:
        """
        Save the workspace configuration to YAML.

        Args:
            path: Target file path. Defaults to self.config_path.
                  If neither is set, saves to ./workspace.yaml.
        """
        save_path = Path(path or self._config_path or WORKSPACE_FILENAME).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as fh:
            yaml.dump(
                self._config.to_yaml_dict(),
                fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        self._config_path = save_path
        logger.debug("Saved workspace config to %s", save_path)
        return save_path

    # ── Mutation ──────────────────────────────────────────────────────────────

    def set_skill(self, skill: Optional[str]) -> None:
        """Update the active skill for this workspace (in memory only)."""
        self._config = self._config.model_copy(update={"skill": skill})

    def set_default_dir(self, directory: str) -> None:
        """Update the workspace default_dir (in memory only)."""
        self._config = self._config.model_copy(update={"default_dir": directory})

    def set_terminal_dir(self, directory: str) -> None:
        """Update the workspace terminal_dir (in memory only)."""
        self._config = self._config.model_copy(update={"terminal_dir": directory})

    # ── Display ───────────────────────────────────────────────────────────────

    def get_info(self) -> dict:
        """Return a human-readable info dict for display."""
        return {
            "name": self._config.name,
            "default_dir": str(self.default_dir),
            "terminal_dir": str(self.terminal_dir),
            "skill": self._config.skill or "(none)",
            "model": self._config.model or "(default)",
            "provider": self._config.provider or "(default)",
            "description": self._config.description or "",
            "config_file": str(self._config_path) if self._config_path else "(in memory)",
        }

    def __repr__(self) -> str:
        return f"WorkspaceManager(name={self._config.name!r}, dir={self.default_dir})"


# ── Auto-discovery helper ─────────────────────────────────────────────────────

def find_workspace(start_dir: Optional[str] = None) -> Optional[WorkspaceManager]:
    """
    Walk up the directory tree from *start_dir* looking for a workspace.yaml.
    Returns None if no workspace is found.
    """
    search = Path(start_dir or os.getcwd()).resolve()
    for candidate in [search, *search.parents]:
        cfg_file = candidate / WORKSPACE_FILENAME
        if cfg_file.exists():
            mgr = WorkspaceManager.load(cfg_file)
            if mgr:
                return mgr
        # Stop at filesystem root or home directory
        if candidate == candidate.parent or candidate == Path.home():
            break
    return None
