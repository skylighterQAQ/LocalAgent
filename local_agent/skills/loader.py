"""
Skill Loader – discovers and loads skill directories.

Convention:
  Each skill lives in its own sub-directory with a skill.json file:
    <dir>/<skill_name>/skill.json

  skill.json must be a valid JSON file that matches the ParsedSkillConfig schema.
  See local_agent/skills/parsed_config.py for the full schema definition.

  Example skill.json::

    {
      "skill_name": "my_skill",
      "version": "1.0.0",
      "description": "Does something useful.",
      "overview": "One-line overview for step context injection.",
      "skill_prompt": "Global rules injected into every step...",
      "available_tools": ["fs_read_file", "shell_run"],
      "input_spec": {"task": "string - task description"},
      "output_spec": {"result": "string - final result"},
      "on_input_mismatch": "warn",
      "on_output_mismatch": "warn",
      "steps": [...]
    }

Note:
  SKILL.md and skill.py formats are no longer supported.
  Users must provide skill.json directly.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from local_agent.skills.base import BaseSkill, JsonSkill, SkillConfig
from local_agent.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


class SkillLoader:
    """Loads skills from directories and registers them with SkillRegistry."""

    def __init__(self) -> None:
        self.registry = SkillRegistry()

    # ── Public API ────────────────────────────────────────────────────────

    def load_builtin_skills(self) -> int:
        """Load all skills bundled with LocalAgent and return the count loaded."""
        builtin_dir = Path(__file__).parent / "builtin"
        return self.load_from_directory(str(builtin_dir))

    def load_from_directory(self, directory: str, _depth: int = 0) -> int:
        """
        Scan *directory* for skill sub-directories and load each one.
        Sub-directories starting with ``_`` are skipped.

        Only ``skill.json`` files are supported. SKILL.md and skill.py are ignored.

        Supports nested directories up to 2 levels deep. If a sub-directory does
        not contain a ``skill.json`` file directly, it is treated as a namespace
        directory (e.g., ``sub_skills/``) and its children are scanned recursively.

        Returns the total number of skills registered.
        """
        loaded = 0
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.debug("Skill directory does not exist: %s", directory)
            return 0

        for skill_dir in sorted(dir_path.iterdir()):
            if not skill_dir.is_dir() or skill_dir.name.startswith("_"):
                continue

            skill_json = skill_dir / "skill.json"

            if skill_json.exists():
                try:
                    n = self._load_from_json(skill_json)
                    if n:
                        loaded += n
                        continue
                except Exception as exc:
                    logger.warning(
                        "Failed to load skill from %s (skill.json): %s",
                        skill_dir,
                        exc,
                    )
            elif _depth < 2:
                # Treat as namespace directory (e.g. sub_skills/), recurse into it.
                logger.debug(
                    "No skill.json in %s – treating as namespace dir, recursing (depth=%d)",
                    skill_dir, _depth + 1,
                )
                sub_loaded = self.load_from_directory(str(skill_dir), _depth=_depth + 1)
                loaded += sub_loaded
            else:
                logger.debug(
                    "No skill.json found in %s – skipping (max recursion depth reached)",
                    skill_dir,
                )

        if _depth == 0:
            logger.info("Loaded %d skills from %s", loaded, directory)
        return loaded

    # ── skill.json loader ─────────────────────────────────────────────────

    def _load_from_json(self, json_path: Path) -> int:
        """
        Parse a skill.json file, build a JsonSkill and register it.

        The JSON must conform to the ParsedSkillConfig schema.
        """
        from local_agent.skills.parsed_config import ParsedSkillConfig

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error in %s: %s", json_path, exc)
            return 0

        try:
            parsed_config = ParsedSkillConfig(**data)
        except Exception as exc:
            logger.warning(
                "skill.json at %s does not match ParsedSkillConfig schema: %s",
                json_path,
                exc,
            )
            return 0

        if not parsed_config.skill_name:
            logger.warning("skill.json at %s has no 'skill_name'", json_path)
            return 0

        skill = JsonSkill(parsed_config=parsed_config)
        self.registry.register(skill)
        logger.debug("Loaded skill '%s' from %s", parsed_config.skill_name, json_path)
        return 1
