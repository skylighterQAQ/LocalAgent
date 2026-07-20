"""
Tool Base Classes for LocalAgent.

Provides:
  - LocalAgentTool  – 带元数据的工具基类（继承本地 BaseTool）
  - tool_metadata   – 装饰器工厂，为 @tool 函数注入元数据

改动说明：
  移除对 langchain_core.tools 的依赖，改为使用 local_agent.engine.tools。
  公开接口完全不变，现有工具代码无需修改任何逻辑。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from local_agent.core.tools import BaseTool, LocalAgentTool, tool, tool_metadata

__all__ = ["BaseTool", "LocalAgentTool", "tool", "tool_metadata"]
