"""
MCPManager – manages the lifecycle of all configured MCP servers.

Responsibilities:
  1. Load MCPConfig from disk
  2. For each enabled server, connect a MCPClient and discover tools
  3. Wrap discovered tools as LangChain BaseTool instances
  4. Register them with the global ToolRegistry
  5. Provide reload / shutdown / status APIs

Thread-safety: connect/disconnect operations are guarded by a lock.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import MCPClient
from .config import MCPConfig, MCPServerConfig
from .tool_adapter import build_lc_tools

logger = logging.getLogger(__name__)


class _ServerState:
    """Runtime state for a single MCP server."""

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self.client: Optional[MCPClient] = None
        self.tools: List[Any] = []  # LangChain BaseTool instances
        self.error: Optional[str] = None
        self.connected: bool = False


class MCPManager:
    """
    High-level manager for all MCP servers configured in config/mcp.json.

    Usage (standalone)::

        manager = MCPManager.from_config_path("config/mcp.json")
        tools = manager.load_all()    # returns List[BaseTool]
        manager.stop_all()

    Usage inside LocalAgent::

        manager = MCPManager.from_config_path(settings.mcp_config_path)
        all_tools = manager.load_all()
        registry = ToolRegistry.get_instance()
        for tool in all_tools:
            registry.register(tool)

    Usage as context manager::

        with MCPManager.from_config_path("config/mcp.json") as mgr:
            tools = mgr.load_all()
            # ... use tools ...
        # stop_all() called automatically on exit
    """

    def __init__(self, mcp_config: MCPConfig) -> None:
        self._config = mcp_config
        self._states: Dict[str, _ServerState] = {}
        self._lock = threading.Lock()

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_config_path(cls, config_path: str = "config/mcp.json") -> "MCPManager":
        """Load MCPConfig from disk and return a new MCPManager."""
        cfg = MCPConfig.load(config_path)
        return cls(cfg)

    # ── Public API ────────────────────────────────────────────────────────

    def load_all(self) -> List[Any]:
        """
        Connect to all enabled MCP servers and return all discovered LangChain tools.
        Servers that fail to connect are logged as warnings and skipped.
        """
        enabled = self._config.get_enabled()
        if not enabled:
            logger.info("No enabled MCP servers found")
            return []

        all_tools: List[Any] = []
        for srv_cfg in enabled:
            tools = self._load_server(srv_cfg)
            all_tools.extend(tools)

        logger.info(
            "MCP: loaded %d tool(s) from %d server(s)",
            len(all_tools),
            sum(1 for s in self._states.values() if s.connected),
        )
        return all_tools

    def stop_all(self) -> None:
        """Disconnect from all MCP servers."""
        with self._lock:
            for name, state in list(self._states.items()):
                if state.client and state.connected:
                    try:
                        state.client.close()
                    except Exception as exc:
                        logger.warning("Error closing MCP server '%s': %s", name, exc)
                    state.connected = False
                    state.client = None
            self._states.clear()

    def reload_server(self, name: str) -> List[Any]:
        """
        Reload a single server by name: disconnect, reconnect, re-discover tools.
        Returns the new list of tools for that server.
        """
        with self._lock:
            state = self._states.get(name)
            if state and state.client:
                try:
                    state.client.close()
                except Exception:
                    pass
            if name in self._states:
                del self._states[name]

        srv_cfg = self._config.servers.get(name)
        if not srv_cfg:
            raise ValueError(f"MCP server '{name}' not found in config")
        return self._load_server(srv_cfg)

    def reload_all(self) -> List[Any]:
        """Stop all servers and reload from current config."""
        self.stop_all()
        return self.load_all()

    def status(self) -> List[Dict[str, Any]]:
        """Return status info for all configured servers."""
        result = []
        for name, srv_cfg in self._config.servers.items():
            state = self._states.get(name)
            result.append(
                {
                    "name": name,
                    "transport": srv_cfg.transport,
                    "enabled": srv_cfg.enabled,
                    "connected": state.connected if state else False,
                    "tool_count": len(state.tools) if state else 0,
                    "error": state.error if state else None,
                    "description": srv_cfg.description,
                }
            )
        return result

    def get_tools(self) -> List[Any]:
        """Return all currently loaded LangChain tool instances."""
        tools = []
        for state in self._states.values():
            tools.extend(state.tools)
        return tools

    # ── Internal ──────────────────────────────────────────────────────────

    def _load_server(self, srv_cfg: MCPServerConfig) -> List[Any]:
        """Connect to a single server, discover tools, return BaseTool list."""
        state = _ServerState(srv_cfg)
        client = MCPClient(srv_cfg)

        try:
            client.connect()
            raw_tools = client.list_tools()
            lc_tools = build_lc_tools(srv_cfg, raw_tools)
            state.client = client
            state.tools = lc_tools
            state.connected = True
            logger.info(
                "MCP server '%s' (%s): %d tool(s) loaded",
                srv_cfg.name, srv_cfg.transport, len(lc_tools),
            )
        except Exception as exc:
            state.error = str(exc)
            logger.warning(
                "Failed to connect MCP server '%s': %s", srv_cfg.name, exc
            )
            try:
                client.close()
            except Exception:
                pass

        with self._lock:
            self._states[srv_cfg.name] = state

        return state.tools

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "MCPManager":
        return self

    def __exit__(self, *_) -> None:
        self.stop_all()
