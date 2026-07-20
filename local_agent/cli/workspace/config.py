"""
Workspace Configuration Model
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class WorkspaceConfig(BaseModel):
    """
    Immutable configuration describing a LocalAgent workspace.

    A workspace defines a project context: where files are saved,
    which skill is active by default, and what the terminal working
    directory should be.
    """

    name: str = Field(default="default", description="Workspace name")
    default_dir: str = Field(
        default="./workspace",
        description="Default directory where all created files are saved",
    )
    terminal_dir: str = Field(
        default="",
        description="Default terminal working directory (falls back to default_dir if empty)",
    )
    skill: Optional[str] = Field(
        default=None,
        description="Default skill to activate for this workspace (e.g. 'code_executor')",
    )
    description: str = Field(default="", description="Human-readable workspace description")
    model: Optional[str] = Field(
        default=None,
        description="Default model to use for this workspace (overrides global default)",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Default LLM provider for this workspace (overrides global default)",
    )

    def get_default_dir(self) -> Path:
        """Return the resolved absolute path of the workspace default_dir."""
        return Path(self.default_dir).expanduser().resolve()

    def get_terminal_dir(self) -> Path:
        """Return the resolved absolute path for terminal working directory."""
        if self.terminal_dir:
            return Path(self.terminal_dir).expanduser().resolve()
        return self.get_default_dir()

    def to_yaml_dict(self) -> dict:
        """Serialize to a dict suitable for YAML output."""
        d: dict = {
            "workspace": {
                "name": self.name,
                "default_dir": self.default_dir,
            }
        }
        if self.terminal_dir:
            d["workspace"]["terminal_dir"] = self.terminal_dir
        if self.skill:
            d["workspace"]["skill"] = self.skill
        if self.description:
            d["workspace"]["description"] = self.description
        if self.model:
            d["workspace"]["model"] = self.model
        if self.provider:
            d["workspace"]["provider"] = self.provider
        return d
