"""
MCP Server Configuration Model.

Reads/writes `config/mcp.json` in the same format as Claude Desktop
so users can share configs across tools.

File format (config/mcp.json):
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "env": {}
    },
    "remote": {
      "url": "http://localhost:8765/sse"
    }
  }
}

Inline YAML configuration (config.yaml mcp.servers):
  mcp:
    servers:
      filesystem:
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
      remote:
        url: "http://localhost:8765/sse"

When both sources are present they are merged; inline servers take precedence
over same-name entries in the JSON file.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = "config/mcp.json"


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server entry."""

    name: str = Field(description="Unique server name (used as tool prefix)")
    # stdio transport
    command: Optional[str] = Field(default=None, description="Executable to launch (stdio mode)")
    args: List[str] = Field(default_factory=list, description="Arguments for the command")
    env: Dict[str, str] = Field(default_factory=dict, description="Extra environment variables")
    # sse transport
    url: Optional[str] = Field(default=None, description="SSE endpoint URL (sse mode)")
    # metadata
    enabled: bool = Field(default=True, description="Set False to skip this server")
    description: str = Field(default="", description="Human-readable description")

    @model_validator(mode="after")
    def _require_command_or_url(self) -> "MCPServerConfig":
        if not self.command and not self.url:
            raise ValueError(
                f"MCP server '{self.name}': must specify either 'command' (stdio) or 'url' (sse)"
            )
        return self

    @property
    def transport(self) -> str:
        """Return 'stdio' or 'sse'."""
        return "sse" if self.url else "stdio"


class MCPConfig:
    """
    Reads and writes the config/mcp.json file.

    Usage::

        cfg = MCPConfig.load()            # load from default path
        cfg = MCPConfig.load("my.json")   # load from custom path

        cfg.servers                       # Dict[str, MCPServerConfig]
        cfg.save()                        # write back to disk
    """

    def __init__(
        self,
        servers: Optional[Dict[str, MCPServerConfig]] = None,
        config_path: str = _DEFAULT_CONFIG_PATH,
    ) -> None:
        self.servers: Dict[str, MCPServerConfig] = servers or {}
        self.config_path = Path(config_path)

    # ── I/O ─────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, config_path: str = _DEFAULT_CONFIG_PATH) -> "MCPConfig":
        """
        Load MCP configuration from *config_path* and merge with any inline
        server definitions from ``config.yaml`` (``mcp.servers``).

        Priority (highest → lowest):
          1. ``config.yaml`` ``mcp.servers`` inline definitions
          2. entries in the JSON file at *config_path*

        Returns an empty MCPConfig if neither source has any servers.
        """
        path = Path(config_path)
        servers: Dict[str, MCPServerConfig] = {}

        # ── 1. Load from JSON file (lower priority) ──────────────────────
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error("Failed to parse MCP config %s: %s", path, exc)
                raw = {}

            for name, entry in raw.get("mcpServers", {}).items():
                try:
                    servers[name] = MCPServerConfig(name=name, **entry)
                except Exception as exc:
                    logger.warning("Skipping MCP server '%s' from file: %s", name, exc)

            if servers:
                logger.info("Loaded %d MCP server(s) from %s", len(servers), path)
        else:
            logger.debug("MCP config file not found: %s – will use inline config only", path)

        # ── 2. Merge inline definitions from config.yaml (higher priority) ─
        try:
            from local_agent.core.config import get_settings  # local import to avoid circular deps
            inline: Dict[str, object] = dict(get_settings().mcp_servers)
        except Exception:
            inline = {}

        for name, entry in inline.items():
            if not isinstance(entry, dict):
                continue
            try:
                srv = MCPServerConfig(name=name, **entry)
                if name in servers:
                    logger.debug("MCP server '%s': inline config overrides file config", name)
                servers[name] = srv
            except Exception as exc:
                logger.warning("Skipping inline MCP server '%s': %s", name, exc)

        if inline:
            logger.info(
                "Merged %d inline MCP server(s) from config.yaml; total: %d",
                len(inline),
                len(servers),
            )

        return cls(servers=servers, config_path=config_path)

    def save(self) -> None:
        """Persist configuration to disk."""
        data: Dict = {"mcpServers": {}}
        for name, srv in self.servers.items():
            entry: Dict = {}
            if srv.command:
                entry["command"] = srv.command
                if srv.args:
                    entry["args"] = srv.args
                if srv.env:
                    entry["env"] = srv.env
            if srv.url:
                entry["url"] = srv.url
            if not srv.enabled:
                entry["enabled"] = False
            if srv.description:
                entry["description"] = srv.description
            data["mcpServers"][name] = entry

        self.config_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.debug("MCP config saved to %s", self.config_path)

    # ── Mutations ────────────────────────────────────────────────────────

    def add_server(self, cfg: MCPServerConfig) -> None:
        """Add or replace a server entry."""
        self.servers[cfg.name] = cfg

    def remove_server(self, name: str) -> bool:
        """Remove a server by name; returns True if it existed."""
        if name in self.servers:
            del self.servers[name]
            return True
        return False

    def get_enabled(self) -> List[MCPServerConfig]:
        """Return only enabled server configs."""
        return [s for s in self.servers.values() if s.enabled]
