"""
MCP (Model Context Protocol) management API routes.

Endpoints:
    GET  /api/mcp/servers            – list all configured MCP servers with status
    POST /api/mcp/servers            – add a new server config and reload
    DELETE /api/mcp/servers/{name}   – remove a server config
    POST /api/mcp/servers/{name}/reload  – reconnect a single server
    POST /api/mcp/reload             – reload all servers
    GET  /api/mcp/tools              – list all loaded MCP tools
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response schemas ────────────────────────────────────────────────

class MCPServerCreate(BaseModel):
    """Body for adding a new MCP server."""
    name: str = Field(description="Unique server identifier")
    command: Optional[str] = Field(default=None, description="Executable (stdio mode)")
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    url: Optional[str] = Field(default=None, description="SSE endpoint URL")
    enabled: bool = Field(default=True)
    description: str = Field(default="")


class MCPServerStatus(BaseModel):
    name: str
    transport: str
    enabled: bool
    connected: bool
    tool_count: int
    error: Optional[str]
    description: str


class MCPToolInfo(BaseModel):
    name: str
    description: str
    server: str
    category: str = "mcp"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_manager():
    """Return the app-level MCPManager stored in app.state, or create one."""
    from local_agent.core.config import get_settings
    from local_agent.mcp import MCPManager
    settings = get_settings()
    manager = MCPManager.from_config_path(settings.mcp_config_path)
    return manager


def _get_shared_manager():
    """
    Return the shared MCPManager stored in app state (set during lifespan).
    Falls back to creating a new one if not available.
    """
    # Avoid circular import by not importing app at module level
    # Instead create a fresh manager (used for non-lifespan contexts)
    return _get_manager()


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/servers", response_model=List[MCPServerStatus], summary="List MCP servers")
async def list_servers():
    """Return status of all configured MCP servers."""
    try:
        manager = _get_shared_manager()
        return manager.status()
    except Exception as exc:
        logger.error("Failed to get MCP server status: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/servers", response_model=MCPServerStatus, summary="Add MCP server")
async def add_server(body: MCPServerCreate):
    """Add a new MCP server configuration and attempt to connect."""
    try:
        from local_agent.mcp import MCPConfig, MCPServerConfig
        from local_agent.core.config import get_settings

        settings = get_settings()
        cfg = MCPConfig.load(settings.mcp_config_path)

        srv = MCPServerConfig(**body.model_dump())
        cfg.add_server(srv)
        cfg.save()

        # Reload this server in the shared manager
        manager = _get_shared_manager()
        # Force config reload
        manager._config = cfg
        tools = manager._load_server(srv)

        # Register newly loaded tools
        if tools:
            from local_agent.tools.registry import ToolRegistry
            reg = ToolRegistry()
            for tool in tools:
                reg.register(tool)

        state = manager._states.get(body.name)
        return MCPServerStatus(
            name=body.name,
            transport=srv.transport,
            enabled=srv.enabled,
            connected=state.connected if state else False,
            tool_count=len(state.tools) if state else 0,
            error=state.error if state else None,
            description=srv.description,
        )
    except Exception as exc:
        logger.error("Failed to add MCP server: %s", exc, exc_info=True)
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/servers/{name}", summary="Remove MCP server")
async def remove_server(name: str):
    """Remove a server configuration from disk and disconnect."""
    try:
        from local_agent.mcp import MCPConfig
        from local_agent.core.config import get_settings

        settings = get_settings()
        cfg = MCPConfig.load(settings.mcp_config_path)
        removed = cfg.remove_server(name)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
        cfg.save()

        # Disconnect from shared manager
        manager = _get_shared_manager()
        manager._config = cfg
        with manager._lock:
            state = manager._states.pop(name, None)
            if state and state.client:
                try:
                    state.client.close()
                except Exception:
                    pass

        return {"deleted": name}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/servers/{name}/reload", response_model=MCPServerStatus, summary="Reload MCP server")
async def reload_server(name: str):
    """Reconnect a specific MCP server and refresh its tools."""
    try:
        manager = _get_shared_manager()
        tools = manager.reload_server(name)

        # Re-register tools
        if tools:
            from local_agent.tools.registry import ToolRegistry
            reg = ToolRegistry()
            for tool in tools:
                reg.register(tool)

        state = manager._states.get(name)
        srv = manager._config.servers.get(name)
        return MCPServerStatus(
            name=name,
            transport=srv.transport if srv else "unknown",
            enabled=srv.enabled if srv else False,
            connected=state.connected if state else False,
            tool_count=len(state.tools) if state else 0,
            error=state.error if state else None,
            description=srv.description if srv else "",
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/reload", summary="Reload all MCP servers")
async def reload_all():
    """Stop all MCP servers and reload from current config/mcp.json."""
    try:
        manager = _get_shared_manager()
        tools = manager.reload_all()

        if tools:
            from local_agent.tools.registry import ToolRegistry
            reg = ToolRegistry()
            for tool in tools:
                reg.register(tool)

        return {"reloaded": len(manager._states), "tools_loaded": len(tools)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/tools", response_model=List[MCPToolInfo], summary="List MCP tools")
async def list_tools():
    """Return all currently loaded MCP tools."""
    try:
        manager = _get_shared_manager()
        result = []
        for tool in manager.get_tools():
            result.append(MCPToolInfo(
                name=tool.name,
                description=tool.description or "",
                server=getattr(tool, "server_name", "unknown"),
                category=getattr(tool, "category", "mcp"),
            ))
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
