"""
MCPToolAdapter – 将 MCP 工具描述符转换为本地 BaseTool 实例。

Tool naming convention:  {server_name}__{mcp_tool_name}
（双下划线分隔 server 前缀与工具名）

改动说明：
  移除对 langchain_core.tools.BaseTool 的依赖，改为继承本地 BaseTool。
  MCPToolWrapper 不再是 Pydantic 模型，改为普通类，字段通过 __init__ 设置。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, create_model

from local_agent.core.tools import BaseTool
from .client import MCPClient
from .config import MCPServerConfig

logger = logging.getLogger(__name__)

_SENTINEL = object()


def _sanitize_tool_name(name: str) -> str:
    """
    将工具名规范化为 ^[a-zA-Z0-9_.-]+$ 格式。
    不合法字符（含空格、中文等）替换为下划线。
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_.\-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_.")
    if not sanitized:
        sanitized = "tool"
    return sanitized


def _json_schema_to_pydantic(
    schema: Dict[str, Any], model_name: str = "InputModel"
) -> Optional[Type[BaseModel]]:
    """
    将 MCP inputSchema（JSON Schema 子集）转为 Pydantic BaseModel 类。
    仅处理常见标量和嵌套对象类型，未知类型回退到 Any。
    失败时返回 None。
    """
    from typing import Any as AnyType

    type_map: Dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))
    field_definitions: Dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        python_type = type_map.get(prop_schema.get("type", ""), AnyType)
        description = prop_schema.get("description", "")
        if prop_name in required_fields:
            field_definitions[prop_name] = (python_type, Field(description=description))
        else:
            field_definitions[prop_name] = (
                Optional[python_type],
                Field(default=None, description=description),
            )

    if not field_definitions:
        # 无具名字段时，接受任意 dict
        field_definitions["root_args"] = (
            Optional[Dict[str, Any]],
            Field(default=None, description="Tool arguments"),
        )

    try:
        return create_model(model_name, **field_definitions)
    except Exception as exc:
        logger.warning("Failed to create pydantic model %s: %s", model_name, exc)
        return None


class MCPToolWrapper(BaseTool):
    """
    将 MCP Server 工具包装为本地 BaseTool 实例。

    执行时创建临时 MCPClient，调用对应工具后关闭连接。
    """

    def __init__(
        self,
        name: str,
        description: str,
        server_name: str,
        mcp_tool_name: str,
        mcp_config: MCPServerConfig,
        args_schema: Optional[Type[BaseModel]] = None,
    ):
        self.name = name
        self.description = description
        self.server_name = server_name
        self.mcp_tool_name = mcp_tool_name
        self.mcp_config = mcp_config
        self.args_schema = args_schema
        self.metadata: Dict[str, Any] = {
            "category": "mcp",
            "requires_confirmation": False,
        }

    # ------------------------------------------------------------------
    # BaseTool 接口实现
    # ------------------------------------------------------------------

    def _run(self, **kwargs: Any) -> str:
        """同步执行：创建临时 MCPClient 调用工具。"""
        if self.mcp_config is None:
            return "[MCPToolWrapper] Error: no server config attached"

        # 过滤掉哨兵值和 Pydantic schema 的占位字段
        args = {
            k: v
            for k, v in kwargs.items()
            if v is not _SENTINEL and v is not None and k != "root_args"
        }

        async def _call():
            async with MCPClient.async_context(self.mcp_config) as client:
                return await client.async_call_tool(self.mcp_tool_name, args)

        try:
            return asyncio.run(_call())
        except Exception as exc:
            logger.error(
                "Error calling MCP tool %s on server %s: %s",
                self.mcp_tool_name,
                self.server_name,
                exc,
                exc_info=True,
            )
            return f"[MCP Error] {exc}"

    async def _arun(self, **kwargs: Any) -> str:
        """异步执行路径。"""
        if self.mcp_config is None:
            return "[MCPToolWrapper] Error: no server config attached"

        args = {
            k: v
            for k, v in kwargs.items()
            if v is not _SENTINEL and v is not None and k != "root_args"
        }

        async with MCPClient.async_context(self.mcp_config) as client:
            return await client.async_call_tool(self.mcp_tool_name, args)

    # ------------------------------------------------------------------
    # Schema 生成（覆盖 BaseTool 默认实现）
    # ------------------------------------------------------------------

    def _get_parameters_schema(self) -> Dict[str, Any]:
        if self.args_schema is not None:
            try:
                return self.args_schema.model_json_schema()
            except Exception:
                pass
        return {"type": "object", "properties": {}}


def build_lc_tools(
    server_config: MCPServerConfig,
    tool_dicts: List[Dict[str, Any]],
) -> List[MCPToolWrapper]:
    """
    根据 server_config 和 MCP list_tools 返回的工具描述列表，
    构建并返回一组 MCPToolWrapper 实例。
    """
    result: List[MCPToolWrapper] = []
    for tool_dict in tool_dicts:
        raw_name = tool_dict["name"]
        qualified_name = _sanitize_tool_name(f"{server_config.name}__{raw_name}")
        description = tool_dict.get("description", "")
        input_schema = tool_dict.get("inputSchema", {})

        # 尝试从 inputSchema 构建 Pydantic args_schema
        try:
            args_schema = _json_schema_to_pydantic(
                input_schema, model_name=f"{qualified_name}_Input"
            )
        except Exception as exc:
            logger.warning(
                "Could not build args_schema for %s: %s – using generic",
                qualified_name,
                exc,
            )
            args_schema = None

        wrapper = MCPToolWrapper(
            name=qualified_name,
            description=description or f"MCP tool '{raw_name}' from server '{server_config.name}'",
            server_name=server_config.name,
            mcp_tool_name=raw_name,
            mcp_config=server_config,
            args_schema=args_schema,
        )
        result.append(wrapper)
        logger.debug("Registered MCP tool: %s", qualified_name)

    return result
