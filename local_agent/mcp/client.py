"""
MCPClient – connects to a single MCP server (stdio or SSE transport)
and provides list_tools / call_tool operations.

Uses the official `mcp` Python SDK.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

from .config import MCPServerConfig

logger = logging.getLogger(__name__)


def _run_sync(coro):
    """Run an async coroutine synchronously, handling existing event loops."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context (e.g. FastAPI) – use a thread-executor
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class MCPClient:
    """
    Thin wrapper around the MCP Python SDK client.

    Supports both stdio (local subprocess) and SSE (remote HTTP) transports.

    Usage (sync)::

        client = MCPClient(config)
        client.connect()
        tools = client.list_tools()
        result = client.call_tool("tool_name", {"arg": "value"})
        client.close()

    Usage (async)::

        async with MCPClient.async_context(config) as client:
            tools = await client.async_list_tools()
            result = await client.async_call_tool("tool_name", {"arg": "value"})
    """

    def __init__(self, config: MCPServerConfig) -> None:
        self.config = config
        self._session = None          # mcp.ClientSession
        self._stdio_transport = None  # only for stdio mode
        self._connected = False

    # ── Async core ───────────────────────────────────────────────────────

    async def _async_connect(self) -> None:
        """Establish connection to the MCP server."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.sse import sse_client
        except ImportError as exc:
            raise RuntimeError(
                "The `mcp` package is required for MCP support. "
                "Install it with: pip install mcp>=1.0.0"
            ) from exc

        if self.config.transport == "stdio":
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env or None,
            )
            self._stdio_transport = stdio_client(server_params)
            read, write = await self._stdio_transport.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()
        else:
            # SSE mode
            self._sse_transport = sse_client(url=self.config.url)
            read, write = await self._sse_transport.__aenter__()
            self._session = ClientSession(read, write)
            await self._session.__aenter__()

        await self._session.initialize()
        self._connected = True
        logger.info("MCP server '%s' connected (%s)", self.config.name, self.config.transport)

    async def _async_close(self) -> None:
        """Close the connection and clean up resources."""
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None

        if self._stdio_transport:
            try:
                await self._stdio_transport.__aexit__(None, None, None)
            except Exception:
                pass
            self._stdio_transport = None

        if hasattr(self, "_sse_transport") and self._sse_transport:
            try:
                await self._sse_transport.__aexit__(None, None, None)
            except Exception:
                pass

        self._connected = False
        logger.debug("MCP server '%s' disconnected", self.config.name)

    async def async_list_tools(self) -> List[Dict[str, Any]]:
        """Return list of tools as raw dicts (name, description, inputSchema)."""
        if not self._connected:
            raise RuntimeError(f"MCPClient for '{self.config.name}' is not connected")
        response = await self._session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {},
            }
            for t in response.tools
        ]

    async def async_call_tool(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> str:
        """Call an MCP tool and return the result as a string."""
        if not self._connected:
            raise RuntimeError(f"MCPClient for '{self.config.name}' is not connected")
        response = await self._session.call_tool(tool_name, arguments=arguments or {})

        # Flatten content blocks into a single string
        parts: List[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif hasattr(block, "data"):
                parts.append(f"[binary data: {len(block.data)} bytes]")
            else:
                parts.append(str(block))
        return "\n".join(parts)

    # ── Sync wrappers ─────────────────────────────────────────────────────

    def connect(self) -> None:
        """Synchronously connect."""
        _run_sync(self._async_connect())

    def close(self) -> None:
        """Synchronously close."""
        _run_sync(self._async_close())

    def list_tools(self) -> List[Dict[str, Any]]:
        """Synchronously list tools."""
        return _run_sync(self.async_list_tools())

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
        """Synchronously call a tool."""
        return _run_sync(self.async_call_tool(tool_name, arguments))

    # ── Context manager (async) ───────────────────────────────────────────

    @classmethod
    @asynccontextmanager
    async def async_context(cls, config: MCPServerConfig) -> AsyncIterator["MCPClient"]:
        """Async context manager: connect → yield → close."""
        client = cls(config)
        await client._async_connect()
        try:
            yield client
        finally:
            await client._async_close()

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"MCPClient(name={self.config.name!r}, transport={self.config.transport!r}, status={status})"
