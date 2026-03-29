"""Skill base class and registry for LocalAgent."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from langchain_core.tools import BaseTool


class OpenClawSkill(ABC):
    """Base class for all LocalAgent skills."""

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


class SkillRegistry:
    """Registry for managing LocalAgent skills."""

    def __init__(self):
        self._skills: Dict[str, OpenClawSkill] = {}

    def register(self, skill: OpenClawSkill) -> None:
        """Register a skill instance."""
        self._skills[skill.name] = skill
        skill.on_load()

    def unregister(self, name: str) -> None:
        """Unregister a skill by name."""
        if name in self._skills:
            self._skills[name].on_unload()
            del self._skills[name]

    def get_all_tools(self) -> List[BaseTool]:
        """Get all tools from all registered skills."""
        tools = []
        for skill in self._skills.values():
            tools.extend(skill.get_tools())
        return tools

    def list_skills(self) -> List[Dict[str, str]]:
        """List all registered skills with metadata."""
        return [
            {"name": s.name, "description": s.description, "version": s.version}
            for s in self._skills.values()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._skills


_registry = SkillRegistry()


def get_registry() -> SkillRegistry:
    return _registry
