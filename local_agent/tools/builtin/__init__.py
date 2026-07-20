"""
Built-in Tools Package.

Auto-discovers and loads every ``*.py`` module in this directory whose
public ``TOOLS`` list contains ``BaseTool`` instances.

Usage::

    from local_agent.tools.builtin import load_all_builtin_tools
    loaded = load_all_builtin_tools()          # returns count

Custom tools placed under ``tools/`` (project root) are NOT loaded here –
that is handled by ``LocalAgent.create()``.
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import List

from local_agent.core.tools import BaseTool

logger = logging.getLogger(__name__)

# Explicit load-order list keeps import behaviour predictable.
# Add new built-in modules here (filename without .py).
_BUILTIN_MODULES: List[str] = [
    "filesystem",
    "shell",
    "code_executor",
    "search",
    "browser",
    "data_analysis",
    "system_utils",
    "memory_tools",
    "git_tools",
    "project_tools",
    "computer",
]


def load_all_builtin_tools() -> int:
    """
    Import every built-in tool module and register its TOOLS list.
    Returns the total number of tools registered.
    """
    from local_agent.tools.registry import ToolRegistry
    registry = ToolRegistry()
    total = 0

    for mod_name in _BUILTIN_MODULES:
        full_name = f"local_agent.tools.builtin.{mod_name}"
        try:
            module = importlib.import_module(full_name)
        except Exception as exc:
            logger.warning("Could not import tool module %s: %s", full_name, exc)
            continue

        tools: List[BaseTool] = getattr(module, "TOOLS", [])
        for t in tools:
            if isinstance(t, BaseTool):
                registry.register(t)
                total += 1
            else:
                logger.debug("Skipping non-BaseTool item in %s.TOOLS: %r", mod_name, t)

    logger.info("Loaded %d built-in tools total", total)
    return total
