"""
Skill Registry – thread-safe singleton.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from local_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Thread-safe singleton registry for all LocalAgent skills."""

    _instance: Optional["SkillRegistry"] = None
    _class_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "SkillRegistry":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._skills: Dict[str, BaseSkill] = {}
                    obj._lock = threading.Lock()
                    cls._instance = obj
        return cls._instance

    # ── Mutation ──────────────────────────────────────────────────────────

    def register(self, skill: BaseSkill) -> None:
        name = skill.get_name()
        with self._lock:
            self._skills[name] = skill
        logger.debug("Registered skill: %s", name)

    def unregister(self, name: str) -> bool:
        with self._lock:
            if name in self._skills:
                del self._skills[name]
                return True
        return False

    # ── Queries ───────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[BaseSkill]:
        return self._skills.get(name)

    def get_all(self) -> List[BaseSkill]:
        return list(self._skills.values())

    def get_skill_names(self) -> List[str]:
        return list(self._skills.keys())

    def get_all_info(self) -> List[Dict[str, Any]]:
        return [skill.get_info() for skill in self._skills.values()]

    # ── Maintenance ───────────────────────────────────────────────────────

    def clear(self) -> None:
        with self._lock:
            self._skills.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (useful in tests)."""
        with cls._class_lock:
            cls._instance = None
