"""
Skill Base Classes.

Provides:
  - SkillConfig  – Pydantic model describing a skill's metadata (legacy compat)
  - BaseSkill    – Abstract base class all skills must inherit from
  - JsonSkill    – Skill loaded from a skill.json file (preferred)

Design notes:
  - JsonSkill is the only supported skill type going forward.
  - All skill definitions must be provided as skill.json files.
  - SkillConfig is kept for backward compatibility with any code that references it,
    but new skills should use ParsedSkillConfig directly via JsonSkill.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from local_agent.core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from local_agent.skills.parsed_config import ParsedSkillConfig


class SkillConfig(BaseModel):
    """
    Legacy metadata model – kept for backward compatibility only.
    New skills should use ParsedSkillConfig (via skill.json) directly.
    """

    name: str
    description: str
    version: str = "1.0.0"
    required_tools: List[str] = Field(
        default_factory=list,
        description="Tool names required by this skill (empty = use all registered tools)",
    )
    system_prompt: str = Field(
        default="",
        description="System prompt injected when this skill is active",
    )
    model_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="LLM parameter overrides (e.g. temperature)",
    )
    tags: List[str] = Field(default_factory=list)
    author: str = "LocalAgent"
    steps: List[str] = Field(
        default_factory=list,
        description="High-level execution steps (informational)",
    )
    nested_skills: List[str] = Field(
        default_factory=list,
        description="Names of other skills this skill is allowed to invoke.",
    )


class BaseSkill(ABC):
    """
    Abstract base class for all LocalAgent skills.

    Preferred approach (no skill.py needed)::

        Provide a skill.json file in the skill directory.
        The SkillLoader will parse it and create a JsonSkill instance.

    Example::

        class MySkill(BaseSkill):
            @classmethod
            def get_config(cls) -> SkillConfig:
                return SkillConfig(
                    name="my_skill",
                    description="Does X",
                    required_tools=["fs_read_file"],
                    system_prompt="You are an expert in X...",
                )

        SKILL = MySkill()
    """

    # ── Abstract interface ────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def get_config(cls) -> SkillConfig:
        """Return the immutable SkillConfig for this skill."""
        ...

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.__class__.get_config().name

    @property
    def description(self) -> str:
        return self.__class__.get_config().description

    # ── Public helpers ────────────────────────────────────────────────────

    def get_name(self) -> str:
        return self.__class__.get_config().name

    def get_description(self) -> str:
        return self.__class__.get_config().description

    def get_system_prompt(self) -> str:
        return self.__class__.get_config().system_prompt

    def get_required_tool_names(self) -> List[str]:
        return self.__class__.get_config().required_tools

    def get_tools(self) -> List[BaseTool]:
        """Resolve required tool names to BaseTool instances via the registry."""
        from local_agent.tools.registry import ToolRegistry
        registry = ToolRegistry()
        names = self.get_required_tool_names()
        if not names:
            return registry.get_all()
        return registry.get_for_skill(self.get_name(), names)

    def get_info(self) -> Dict[str, Any]:
        cfg = self.__class__.get_config()
        return {
            "name": cfg.name,
            "description": cfg.description,
            "version": cfg.version,
            "required_tools": cfg.required_tools,
            "tags": cfg.tags,
            "author": cfg.author,
        }


class JsonSkill(BaseSkill):
    """
    A skill loaded from a skill.json file.

    This is the only officially supported skill type.
    Users must provide a skill.json file in the skill directory.

    The skill.json must conform to the ParsedSkillConfig schema
    (see local_agent/skills/parsed_config.py).

    Args:
        parsed_config: A fully-populated ParsedSkillConfig loaded from skill.json.
    """

    def __init__(self, parsed_config: "ParsedSkillConfig") -> None:
        self._parsed_config = parsed_config

    # BaseSkill requires get_config as a classmethod; satisfy the ABC without
    # breaking the instance-level config pattern used here.
    @classmethod
    def get_config(cls) -> SkillConfig:  # type: ignore[override]
        raise NotImplementedError(
            "JsonSkill instances carry their config in parsed_config; "
            "use instance methods (get_name, get_description, …) instead."
        )

    # ── Override all instance helpers to use self._parsed_config ──────────

    @property
    def name(self) -> str:
        return self._parsed_config.skill_name

    @property
    def description(self) -> str:
        return self._parsed_config.description

    def get_name(self) -> str:
        return self._parsed_config.skill_name

    def get_description(self) -> str:
        return self._parsed_config.description

    def get_system_prompt(self) -> str:
        """Return the skill's global prompt (skill_prompt field)."""
        return self._parsed_config.skill_prompt

    def get_required_tool_names(self) -> List[str]:
        return self._parsed_config.available_tools

    def get_tools(self) -> List[BaseTool]:
        from local_agent.tools.registry import ToolRegistry
        registry = ToolRegistry()
        names = self._parsed_config.available_tools
        if not names:
            return registry.get_all()
        return registry.get_for_skill(self._parsed_config.skill_name, names)

    def get_info(self) -> Dict[str, Any]:
        cfg = self._parsed_config
        return {
            "name": cfg.skill_name,
            "description": cfg.description,
            "version": cfg.version,
            "required_tools": cfg.available_tools,
            "tags": [],
            "author": "LocalAgent",
            "steps_count": len(cfg.steps),
        }

    # ── ParsedSkillConfig access ───────────────────────────────────────────

    @property
    def parsed_config(self) -> "ParsedSkillConfig":
        """Return the underlying ParsedSkillConfig (always available for JsonSkill)."""
        return self._parsed_config
