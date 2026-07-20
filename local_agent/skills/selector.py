"""
Skill Selector – LLM-driven dynamic skill activation.

When a user sends a message without specifying a skill, this module uses
a lightweight LLM call to analyze the query and decide which skill(s) to activate.

This mirrors the Claude Code approach: the model reads skill descriptions,
determines relevance, and returns an ordered activation list.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

# ─── Skill selection prompt template ─────────────────────────────────────────

_SELECTION_PROMPT_TEMPLATE = """\
You are a task analyzer. Given a user query, determine which skill(s) to activate.

Available Skills:
{skill_list}

User Query:
{query}

Instructions:
- Analyze the query and decide which skills are relevant based on their descriptions.
- A skill should only be activated when the query clearly requires its capabilities.
- Multiple skills may be activated if the task spans multiple domains.
- Order skills by execution priority (most relevant first).
- If no skill is needed (e.g., simple questions, general conversation), return an empty list.

IMPORTANT DISAMBIGUATION RULES:
- Use "greenfield_developer" when: creating a NEW project from scratch, building something that doesn't exist yet, starting from an idea/concept.
- Use "existing_project_developer" when: modifying/extending/fixing an EXISTING codebase that already has files and structure.
- If the user says "write", "create", "build", "develop" a new software/app/tool WITHOUT mentioning an existing project, use "greenfield_developer".

Respond ONLY with valid JSON in this exact format (no explanation, no markdown):
{{"skills": ["skill_name_1", "skill_name_2"], "reason": "brief explanation"}}
"""


class SkillSelector:
    """
    Uses an LLM to analyze a user query and select which skills to activate.

    The selector is intentionally lightweight:
    - It makes a single, fast LLM call with a focused prompt.
    - It returns an ordered list of skill names.
    - The calling agent decides how to use the selected skills.

    Usage::

        selector = SkillSelector()
        selected = selector.select_skills(
            query="Write a Python web scraper and save it to files",
            llm_provider=agent.llm_provider,
            available_skills=registry.get_all(),
        )
        # selected == ["code_developer", "web_researcher"]
    """

    def select_skills(
        self,
        query: str,
        llm_provider: Any,
        available_skills: List[Any],
        temperature: float = 0.0,
    ) -> List[str]:
        """
        Analyze *query* and return an ordered list of skill names to activate.

        Args:
            query: The user's raw message.
            llm_provider: The agent's LLM provider (must support .get_llm()).
            available_skills: List of BaseSkill instances to consider.
            temperature: LLM temperature (0.0 for deterministic selection).

        Returns:
            Ordered list of skill names. Empty list means no skill needed.
        """
        if not available_skills:
            return []

        # Filter out sub-skills – they are invoked internally by main skills only
        top_level_skills = []
        for skill in available_skills:
            parsed_config = getattr(skill, "parsed_config", None)
            if parsed_config is not None:
                if getattr(parsed_config, "is_sub_skill", False):
                    continue
            top_level_skills.append(skill)

        if not top_level_skills:
            return []

        skill_list = self._format_skill_list(top_level_skills)
        prompt = _SELECTION_PROMPT_TEMPLATE.format(
            skill_list=skill_list,
            query=query,
        )

        try:
            result = self._call_llm(llm_provider, prompt, temperature)
            selected = self._parse_response(result, top_level_skills)
            logger.info(
                "SkillSelector: query=%r → selected=%r",
                query[:80],
                selected,
            )
            return selected
        except Exception as exc:
            logger.warning(
                "SkillSelector failed (non-fatal, falling back to no skill): %s", exc
            )
            return []

    # ── Private helpers ───────────────────────────────────────────────────────

    def _format_skill_list(self, skills: List[Any]) -> str:
        """Format skill info into a compact description for the prompt."""
        lines = []
        for skill in skills:
            name = skill.get_name()
            desc = skill.get_description()
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def _call_llm(self, llm_provider: Any, prompt: str, temperature: float) -> str:
        """
        Make a synchronous LLM call and return the raw text response.

        Tries to use the provider's LLM with a simple messages-based call.
        Falls back gracefully if the provider interface varies.
        
        For thinking models (qwen3, qwq, deepseek-r1 etc.), thinking mode is
        explicitly disabled to ensure clean JSON output without <think> tags.
        """
        from local_agent.core.messages import HumanMessage, SystemMessage

        # 对 thinking 模型（qwen3/qwq/deepseek-r1 等）禁用 thinking 模式，
        # 确保 SkillSelector 输出纯净 JSON，不含 <think>...</think> 标签。
        llm_kwargs: dict = {"temperature": temperature}
        try:
            from local_agent.llm.clients.ollama import _is_thinking_model
            model_name = getattr(llm_provider, "model", "") or ""
            if model_name and _is_thinking_model(model_name):
                llm_kwargs["disable_thinking"] = True
                logger.debug(
                    "SkillSelector: disabling thinking mode for model '%s' to get clean JSON",
                    model_name,
                )
        except ImportError:
            pass

        llm = llm_provider.get_llm(**llm_kwargs)

        messages = [
            SystemMessage(content="You are a precise task analyzer. Always respond with valid JSON only."),
            HumanMessage(content=prompt),
        ]

        # All LocalAgent LLM clients implement ``invoke``.  Do not mask a
        # connection/API failure with a legacy ``predict`` call that these
        # clients intentionally do not provide.
        response = llm.invoke(messages)
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def _parse_response(
        self,
        response: str,
        available_skills: List[Any],
    ) -> List[str]:
        """
        Parse the LLM JSON response and return valid skill names.

        Gracefully handles malformed responses by:
        1. Trying direct JSON parse.
        2. Extracting JSON from markdown code blocks.
        3. Returning empty list on failure.
        """
        valid_names = {skill.get_name() for skill in available_skills}

        # Try to find JSON object in response
        text = response.strip()

        # Remove markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = re.sub(r"```\s*$", "", text).strip()

        # Extract first JSON object if there's extra text
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("SkillSelector: Could not parse JSON response: %s | raw=%r", exc, response[:200])
            return []

        if not isinstance(data, dict):
            logger.warning("SkillSelector: Response is not a dict: %r", data)
            return []

        skills_raw = data.get("skills", [])
        if not isinstance(skills_raw, list):
            return []

        # Filter to only valid, registered skill names (preserves LLM order)
        selected = [name for name in skills_raw if name in valid_names]

        reason = data.get("reason", "")
        if reason:
            logger.debug("SkillSelector reason: %s", reason)

        return selected
