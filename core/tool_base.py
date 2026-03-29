"""工具基类及其注册"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from langchain_core.tools import BaseTool


class LocalAgentTool(ABC):
    """工具基类"""

    name: str = ""
    description: str = ""
    version: str = "1.0.0"

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Return list of LangChain tools provided by this skill."""
        ...

    def on_load(self) -> None:
        """Called when skill is loaded. Override for initialization."""
        pass

    def on_unload(self) -> None:
        """Called when skill is unloaded. Override for cleanup."""
        pass


class ToolRegistry:
    """ 注册工具 """

    def __init__(self):
        self._tools: Dict[str, LocalAgentTool] = {}

    def register(self, tool: LocalAgentTool) -> None:
        """Register a skill instance."""
        self._tools[tool.name] = tool
        tool.on_load()

    def unregister(self, name: str) -> None:
        """Unregister a skill by name."""
        if name in self._tools:
            self._tools[name].on_unload()
            del self._tools[name]

    def get_all_tools(self) -> List[BaseTool]:
        """Get all tools from all registered tools."""
        tools = []
        for skill in self._tools.values():
            tools.extend(skill.get_tools())
        return tools

    def list_tools(self) -> List[Dict[str, str]]:
        """List all registered tools with metadata."""
        return [
            {"name": s.name, "description": s.description, "version": s.version}
            for s in self._tools.values()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._tools


_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    return _registry
