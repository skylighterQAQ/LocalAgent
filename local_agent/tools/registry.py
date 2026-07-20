"""
Tool Registry – thread-safe singleton.

Supports:
  - Explicit registration (register / register_many)
  - Directory scanning (load_from_directory)
  - Category grouping
  - Metadata query (get_tool_info)
"""
from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from local_agent.core.tools import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Thread-safe singleton central registry for all LocalAgent tools."""

    _instance: Optional["ToolRegistry"] = None
    _class_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._tools: Dict[str, BaseTool] = {}
                    obj._categories: Dict[str, List[str]] = {}
                    obj._lock = threading.Lock()
                    cls._instance = obj
        return cls._instance

    # ── Mutation ──────────────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """Register a single tool. Silently replaces existing tool with same name."""
        name = tool.name
        # category is stored in tool.metadata dict (Pydantic v2 safe),
        # with fallback to direct attribute (for LocalAgentTool subclasses)
        meta = tool.metadata or {}
        category = meta.get("category") or getattr(tool, "category", "general")
        with self._lock:
            self._tools[name] = tool
            self._categories.setdefault(category, [])
            if name not in self._categories[category]:
                self._categories[category].append(name)
        logger.debug("Registered tool: %s (category: %s)", name, category)

    def register_many(self, tools: List[BaseTool]) -> None:
        for t in tools:
            self.register(t)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name; returns True if it existed."""
        with self._lock:
            if name not in self._tools:
                return False
            tool = self._tools.pop(name)
            meta = tool.metadata or {}
            category = meta.get("category") or getattr(tool, "category", "general")
            cat_list = self._categories.get(category, [])
            if name in cat_list:
                cat_list.remove(name)
            return True

    # ── Queries ───────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_all(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get_by_category(self, category: str) -> List[BaseTool]:
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]

    def get_categories(self) -> List[str]:
        return list(self._categories.keys())

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_for_skill(self, skill_name: str, required_tools: List[str]) -> List[BaseTool]:
        """Resolve tool names required by a skill (warns on missing tools)."""
        result = []
        for tool_name in required_tools:
            t = self.get(tool_name)
            if t:
                result.append(t)
            else:
                logger.warning(
                    "Skill '%s' requires tool '%s' which is not registered",
                    skill_name, tool_name,
                )
        return result

    def get_tool_info(self) -> List[Dict[str, Any]]:
        result = []
        for t in self._tools.values():
            meta = t.metadata or {}
            result.append({
                "name": t.name,
                "description": t.description,
                "category": meta.get("category") or getattr(t, "category", "general"),
                "requires_confirmation": meta.get("requires_confirmation") or getattr(t, "requires_confirmation", False),
            })
        return result

    # ── Directory scanning ────────────────────────────────────────────────

    def load_from_directory(self, directory: str) -> int:
        """
        Recursively scan *directory* for ``*.py`` files and load their ``TOOLS``
        list.  Files starting with ``_`` are skipped.

        Returns the number of tools successfully registered.
        """
        loaded = 0
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.debug("Tool directory does not exist: %s", directory)
            return 0

        for py_file in dir_path.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue
            try:
                loaded += self._load_from_file(py_file)
            except Exception as exc:
                logger.warning("Failed to load tools from %s: %s", py_file, exc)

        logger.info("Loaded %d tools from %s", loaded, directory)
        return loaded

    def _load_from_file(self, file_path: Path) -> int:
        spec = importlib.util.spec_from_file_location(
            f"_local_agent_tool_{file_path.stem}", file_path
        )
        if spec is None or spec.loader is None:
            return 0
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        loaded = 0
        if hasattr(module, "TOOLS"):
            for item in module.TOOLS:
                if isinstance(item, BaseTool):
                    self.register(item)
                    loaded += 1
        else:
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                attr = getattr(module, attr_name)
                if isinstance(attr, BaseTool):
                    self.register(attr)
                    loaded += 1
        return loaded

    # ── Maintenance ───────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all registered tools (useful in tests)."""
        with self._lock:
            self._tools.clear()
            self._categories.clear()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful in tests)."""
        with cls._class_lock:
            cls._instance = None
