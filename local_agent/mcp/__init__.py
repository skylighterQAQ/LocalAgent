"""
local_agent.mcp – MCP (Model Context Protocol) integration package.

Public API::

    from local_agent.mcp import MCPManager, MCPConfig, MCPServerConfig, load_mcp_tools

    # Easiest usage: load tools from config/mcp.json and get LangChain tool list
    tools = load_mcp_tools("config/mcp.json")

    # Full control
    manager = MCPManager.from_config_path("config/mcp.json")
    all_tools = manager.load_all()
    manager.stop_all()
"""
from .config import MCPConfig, MCPServerConfig
from .client import MCPClient
from .manager import MCPManager

__all__ = [
    "MCPConfig",
    "MCPServerConfig",
    "MCPClient",
    "MCPManager",
    "load_mcp_tools",
]


def load_mcp_tools(config_path: str = "config/mcp.json"):
    """
    Convenience function: load all MCP tools from *config_path*.

    Returns a list of LangChain BaseTool instances.
    The underlying MCP client connections are left open.
    Call ``MCPManager.stop_all()`` explicitly when done.
    """
    manager = MCPManager.from_config_path(config_path)
    return manager.load_all()
