"""
local_agent.skills.skill_tool
==============================
SkillTool – 将一个 skill 封装为工具，实现 skill 嵌套调用。

当父 skill（如 web_researcher）需要将子任务委托给另一个 skill（如 url_accessor）
时，LLM 可以通过调用 `invoke_skill` 工具来触发该嵌套 skill 的执行。

工作原理：
  1. SkillTool 被注册为普通工具（name="invoke_skill"），LLM 可直接 tool_call 调用
  2. _run() 方法接收 skill_name 和 task，从 SkillRegistry 获取目标 skill
  3. JsonSkill 总是携带 ParsedSkillConfig，通过 SkillExecutor 步骤隔离执行
  4. 子引擎同步执行任务，返回最终文本结果给父 Agent

嵌套限制：
  - 子引擎的最大迭代次数默认为 10（防止深层嵌套造成无限循环）
  - 子引擎不允许再次调用 invoke_skill（防止无限递归）
  - 子引擎使用与父引擎相同的 LLM 实例
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from local_agent.core.tools import BaseTool

logger = logging.getLogger(__name__)


class SkillTool(BaseTool):
    """
    将指定 skill 作为工具调用，实现 skill 嵌套。

    暴露给 LLM 的工具名称：``invoke_skill``

    参数：
        skill_name : 要调用的 skill 名称（必须已在 SkillRegistry 中注册）
        task       : 要交给该 skill 完成的具体任务描述

    返回：
        子 skill 执行后的最终文本结果。
    """

    name: str = "invoke_skill"
    description: str = (
        "Invoke a specific skill to handle a sub-task. "
        "Use this when you need the capabilities of another skill to complete part of the current task. "
        "The sub-skill will execute with its own system prompt and tools, then return the result. "
        "IMPORTANT: When invoking 'url_accessor', you MUST pass the target URL in the 'url' parameter "
        "(not embedded in 'task'). Example: invoke_skill(skill_name='url_accessor', "
        "task='extract restaurant info', url='https://example.com/page', query='good BBQ restaurants'). "
        "Parameters: skill_name (str) - the name of the skill; "
        "task (str) - description of what to accomplish; "
        "url (str, optional) - the target URL for url_accessor or similar skills; "
        "query (str, optional) - the research question for content extraction."
    )

    def __init__(
        self,
        llm: Any,
        allowed_skills: Optional[List[str]] = None,
        max_iterations: int = 10,
        debug_hooks: Optional[dict] = None,
        allow_sub_skills: bool = False,
        blocked_skills: Optional[List[str]] = None,
        current_skill_name: Optional[str] = None,
    ) -> None:
        """
        Args:
            llm              : 父 Agent 的 LLM 实例（子引擎复用）
            allowed_skills   : 允许调用的 skill 名称列表（None 表示允许所有）
            max_iterations   : 子 ReActEngine 的最大迭代次数
            debug_hooks      : 调试钩子字典（before_llm / after_llm / after_tool），
                               由 graph.make_debug_hooks() 生成，传入后 skill 内部
                               的 LLM / 工具调用均会产生 debug 日志。
            allow_sub_skills : 是否允许调用 is_sub_skill=True 的 skill。
                               默认 False（用户交互场景禁止直接调用）；
                               SkillExecutor 内部注入时设为 True，以支持 step 通过
                               invoke_skill 调用 sub_skill（如 module_developer）。
            blocked_skills   : 明确禁止调用的 skill 名称列表（优先于 allowed_skills）。
                               主要用于防止 skill 在 AGENT 步骤中递归调用自身。
                               例如：greenfield_developer 的 step_2_5 注入时会将
                               'greenfield_developer' 加入此列表，防止 LLM 错误地
                               递归调用自身而非 module_developer。
            current_skill_name: 当前执行的父 skill 名称，用于检查 sub_skill 的
                               allowed_parent_skills 白名单。
        """
        self._llm = llm
        self._allowed_skills = set(allowed_skills) if allowed_skills else None
        self._max_iterations = max_iterations
        self._debug_hooks: dict = debug_hooks or {}
        self._allow_sub_skills = allow_sub_skills
        self._blocked_skills: set[str] = set(blocked_skills) if blocked_skills else set()
        self._current_skill_name: Optional[str] = current_skill_name
        self.metadata: dict = {"category": "skill"}

    def _run(self, skill_name: str, task: str = "", url: Optional[str] = None, query: Optional[str] = None, **kwargs: Any) -> str:  # type: ignore[override]
        """
        执行嵌套 skill 调用。

        所有 skill 均为 JsonSkill，已在加载时携带 ParsedSkillConfig，
        直接通过 SkillExecutor 步骤隔离执行。

        Args:
            skill_name : 目标 skill 名称
            task       : 子任务描述
            url        : （可选）目标 URL，传入后会以结构化方式调用 skill（如 url_accessor）
            query      : （可选）研究问题，与 url 一起传入时作为额外字段

        Returns:
            子 skill 执行的最终文本输出。
        """
        # 校验必填参数
        if not skill_name:
            return "[Skill Error] 'skill_name' is required but was not provided."
        if not task and not url:
            return (
                "[Skill Error] Either 'task' or 'url' must be provided. "
                "Please supply a task description via the 'task' parameter."
            )
        # ── 0. 禁止调用列表检查（优先于 allowed_skills，防止递归调用）────────
        if skill_name in self._blocked_skills:
            logger.warning(
                "SkillTool._run: skill '%s' is in blocked_skills, rejecting call to prevent recursion. "
                "blocked_skills=%s",
                skill_name, self._blocked_skills,
            )
            return (
                f"[Skill Error] Skill '{skill_name}' is blocked in this context to prevent recursive invocation. "
                + "If you are trying to generate a module design document, use 'module_developer' instead."
            )
        # ── 1. 验证 skill 是否被允许调用 ───────────────────────────────────
        if self._allowed_skills is not None and skill_name not in self._allowed_skills:
            allowed_list = ", ".join(sorted(self._allowed_skills))
            return (
                f"[Skill Error] Skill '{skill_name}' is not in the allowed nested skills list. "
                f"Allowed: {allowed_list}"
            )

        # ── 2. 从注册表获取目标 skill ────────────────────────────────────────
        try:
            from local_agent.skills.registry import SkillRegistry
            registry = SkillRegistry()
            skill = registry.get(skill_name)
        except Exception as exc:
            return f"[Skill Error] Failed to access SkillRegistry: {exc}"

        if skill is None:
            return (
                f"[Skill Error] Skill '{skill_name}' not found in registry. "
                f"Available skills: {', '.join(s.get_name() for s in registry.get_all())}"
            )

        # ── 3. 获取 ParsedSkillConfig（JsonSkill 总是有值）─────────────────
        parsed_config = getattr(skill, "parsed_config", None)
        if parsed_config is None:
            return (
                f"[Skill Error] Skill '{skill_name}' does not have a ParsedSkillConfig. "
                f"Please ensure the skill is defined in skill.json format."
            )

        # ── 3b. 访问控制：sub-skill 不允许通过 invoke_skill 工具直接调用 ─────
        # sub-skills 只能由主 skill 的 StepType.SKILL 步骤通过内部路径调用，
        # 或由 SkillExecutor 内部以 allow_sub_skills=True 注入的 SkillTool 调用。
        # 用户交互场景下，LLM 若尝试直接 invoke sub-skill，在此拦截并返回错误。
        if getattr(parsed_config, "is_sub_skill", False) and not self._allow_sub_skills:
            return (
                f"[Skill Error] Skill '{skill_name}' is an internal sub-skill and cannot be "
                f"invoked directly via the invoke_skill tool. It is designed to be orchestrated "
                f"internally by 'greenfield_developer' or 'existing_project_developer'."
            )

        # ── 3c. 父 skill 白名单检查：检查 sub_skill 的 allowed_parent_skills ──
        # 若 sub_skill 配置了 allowed_parent_skills，则只有白名单中的父 skill 才能调用。
        if getattr(parsed_config, "is_sub_skill", False) and self._allow_sub_skills:
            allowed_parents = getattr(parsed_config, "allowed_parent_skills", None)
            if allowed_parents is not None and self._current_skill_name not in allowed_parents:
                return (
                    f"[Skill Error] Skill '{skill_name}' can only be invoked by "
                    f"{allowed_parents}. Current parent skill: '{self._current_skill_name}'."
                )

        # ── 4. 若调用方提供了 url，走结构化路径（如 url_accessor）────────────
        if url:
            structured_input: dict = {"url": url, "query": query or task}
            result = self._run_structured(skill_name, structured_input)
            # 在结果中注入 URL 标记，便于父 Agent 历史压缩时识别
            return self._inject_url_marker(result, url)

        # ── 5. 否则尝试从 task 字符串中解析 URL（兼容旧调用方式）────────────
        extracted_url = self._extract_url_from_task(task)
        if extracted_url and self._skill_requires_url(parsed_config):
            structured_input = {"url": extracted_url, "query": task}
            logger.info(
                "SkillTool: extracted url '%s' from task string for skill '%s'",
                extracted_url, skill_name,
            )
            result = self._run_structured(skill_name, structured_input)
            # 在结果中注入 URL 标记，便于父 Agent 历史压缩时识别
            return self._inject_url_marker(result, extracted_url)

        # ── 6. 无 URL：走通用 task 路径 ─────────────────────────────────────
        return self._run_with_executor(skill_name, task, parsed_config)

    def _run_structured(self, skill_name: str, structured_input: dict) -> str:
        """
        以结构化字典调用嵌套 skill，保留各输入字段（如 url、query）。

        与 _run() 的区别：直接将 structured_input 作为 initial_input 传给 SkillExecutor，
        不再包裹成 {"task": ..., "query": ...}，避免子 skill 因缺少字段而校验失败。

        Args:
            skill_name      : 目标 skill 名称
            structured_input: 包含子 skill 所需字段的字典

        Returns:
            子 skill 执行的最终文本输出。
        """
        # ── 0. 禁止调用列表检查（防止递归）──────────────────────────────────
        if skill_name in self._blocked_skills:
            logger.warning(
                "SkillTool._run_structured: skill '%s' is in blocked_skills, rejecting call. "
                "blocked_skills=%s",
                skill_name, self._blocked_skills,
            )
            return (
                f"[Skill Error] Skill '{skill_name}' is blocked in this context to prevent recursive invocation. "
                + "If you are trying to generate a module design document, use 'module_developer' instead."
            )
        if self._allowed_skills is not None and skill_name not in self._allowed_skills:
            allowed_list = ", ".join(sorted(self._allowed_skills))
            return (
                f"[Skill Error] Skill '{skill_name}' is not in the allowed nested skills list. "
                f"Allowed: {allowed_list}"
            )

        try:
            from local_agent.skills.registry import SkillRegistry
            registry = SkillRegistry()
            skill = registry.get(skill_name)
        except Exception as exc:
            return f"[Skill Error] Failed to access SkillRegistry: {exc}"

        if skill is None:
            return (
                f"[Skill Error] Skill '{skill_name}' not found in registry. "
                f"Available skills: {', '.join(s.get_name() for s in registry.get_all())}"
            )

        parsed_config = getattr(skill, "parsed_config", None)
        if parsed_config is None:
            return (
                f"[Skill Error] Skill '{skill_name}' does not have a ParsedSkillConfig. "
                f"Please ensure the skill is defined in skill.json format."
            )

        # ── 父 skill 白名单检查（structured 路径）──────────────────────────────
        if getattr(parsed_config, "is_sub_skill", False) and self._allow_sub_skills:
            allowed_parents = getattr(parsed_config, "allowed_parent_skills", None)
            if allowed_parents is not None and self._current_skill_name not in allowed_parents:
                return (
                    f"[Skill Error] Skill '{skill_name}' can only be invoked by "
                    f"{allowed_parents}. Current parent skill: '{self._current_skill_name}'."
                )

        try:
            from local_agent.skills.executor import SkillExecutor
            from local_agent.tools.registry import ToolRegistry

            tool_registry = ToolRegistry()
            executor = SkillExecutor(
                tool_registry=tool_registry,
                llm=self._llm,
                debug_hooks=self._debug_hooks,
            )

            logger.info(
                "SkillTool: invoking '%s' via SkillExecutor (structured, %d steps)",
                skill_name, len(parsed_config.steps),
            )

            result = executor.execute(
                parsed_config=parsed_config,
                initial_input=structured_input,
                global_context=str(structured_input),
            )

            # url_accessor specific: extract content + status + title if available
            final = self._extract_final_output(skill_name, result, parsed_config)

            if final:
                logger.info(
                    "SkillTool: '%s' structured call completed, response length=%d",
                    skill_name, len(final),
                )
                return f"[Result from {skill_name} skill]\n{final}"
            return f"[Skill Error] Skill '{skill_name}' produced no output."

        except Exception as exc:
            logger.error(
                "SkillTool: SkillExecutor for '%s' (structured) raised exception: %s",
                skill_name, exc, exc_info=True,
            )
            return f"[Skill Error] Skill '{skill_name}' structured execution failed: {exc}"

    def _run_with_executor(self, skill_name: str, task: str, parsed_config: Any) -> str:
        """使用 SkillExecutor 步骤隔离执行。"""
        try:
            from local_agent.skills.executor import SkillExecutor
            from local_agent.tools.registry import ToolRegistry

            tool_registry = ToolRegistry()
            executor = SkillExecutor(
                tool_registry=tool_registry,
                llm=self._llm,
                debug_hooks=self._debug_hooks,
            )

            logger.info(
                "SkillTool: invoking '%s' via SkillExecutor (%d steps)",
                skill_name, len(parsed_config.steps),
            )

            result = executor.execute(
                parsed_config=parsed_config,
                initial_input={"task": task, "query": task},
                global_context=task,
            )

            # 提取最终输出：优先取 output_spec 声明字段，再取 "result" key，最后取任意字符串
            final = self._extract_final_output(skill_name, result, parsed_config)

            if final:
                logger.info(
                    "SkillTool: '%s' completed, response length=%d",
                    skill_name, len(final),
                )
                return f"[Result from {skill_name} skill]\n{final}"
            return f"[Skill Error] Skill '{skill_name}' produced no output."

        except Exception as exc:
            logger.error(
                "SkillTool: SkillExecutor for '%s' raised exception: %s",
                skill_name, exc, exc_info=True,
            )
            return f"[Skill Error] Skill '{skill_name}' execution failed: {exc}"

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _inject_url_marker(result: str, url: str) -> str:
        """
        在 invoke_skill 返回结果的开头注入 URL 和状态标记，
        便于父 Agent 的 _compress_invoke_skill_history 函数识别已访问的 URL。

        格式：
          [URL: https://example.com]
          [Status: 成功]
          <原始结果内容>

        若结果已包含 [URL: ...] 标记，则不重复注入。
        """
        if url and "[URL:" not in result:
            # 判断状态
            if "[Skill Error]" in result:
                status = "失败"
            else:
                status = "成功"
            marker = f"[URL: {url}]\n[Status: {status}]\n"
            return marker + result
        return result

    @staticmethod
    def _extract_final_output(skill_name: str, result: dict, parsed_config: Any) -> str:
        """
        从 SkillExecutor 返回的上下文变量中提取最终输出文本。

        优先级：
          1. 对于 url_accessor：组合 title + content + status 字段（如果可用）
          2. 取 output_spec 中声明的所有已填充字段，拼接成完整输出
          3. 取 "result" key
          4. 取任意非空字符串值（逆序遍历，取最近写入的）
        """
        # ── 策略 1：url_accessor 专有字段 ─────────────────────────────────
        if skill_name == "url_accessor":
            content = result.get("content", "")
            title = result.get("title", "")
            status = result.get("status", "")
            parts = []
            if title:
                parts.append(f"标题: {title}")
            if content:
                parts.append(content)
            if status:
                parts.append(f"[状态: {status}]")
            if parts:
                return "\n".join(parts)
            # 若专有字段都为空，fallthrough 到通用策略

        # ── 策略 2：取 output_spec 中声明的所有已填充字段，拼接输出 ────────
        output_spec = getattr(parsed_config, "output_spec", None)
        if isinstance(output_spec, dict):
            field_parts = []
            for field in output_spec:
                if field in result and result[field] and isinstance(result[field], str):
                    field_parts.append(str(result[field]))
            if field_parts:
                return "\n".join(field_parts)

        # ── 策略 3：取 "result" key ───────────────────────────────────────
        if "result" in result and result["result"]:
            return str(result["result"])

        # ── 策略 4：逆序取任意字符串 ─────────────────────────────────────
        for v in reversed(list(result.values())):
            if isinstance(v, str) and v:
                return v

        return ""

    @staticmethod
    def _extract_url_from_task(task: str) -> Optional[str]:
        """从 task 字符串中提取第一个 http/https URL。"""
        import re
        match = re.search(r'https?://[^\s\'"，。）\]）]+', task)
        return match.group(0) if match else None

    @staticmethod
    def _skill_requires_url(parsed_config: Any) -> bool:
        """判断 skill 的 input_spec 是否包含 url 字段（表示需要 URL 输入）。"""
        input_spec = getattr(parsed_config, "input_spec", None)
        if isinstance(input_spec, dict):
            return "url" in input_spec
        return False

    def get_schema(self) -> dict:
        """返回 OpenAI function-calling 格式的工具定义。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": "The name of the skill to invoke (e.g., 'url_accessor')",
                        },
                        "task": {
                            "type": "string",
                            "description": (
                                "A clear, detailed description of the task for the sub-skill. "
                                "Include all necessary context (e.g., what information to extract)."
                            ),
                        },
                        "url": {
                            "type": "string",
                            "description": (
                                "Optional. When invoking url_accessor or any URL-based skill, "
                                "provide the target URL here directly (e.g., 'https://example.com/article'). "
                                "This ensures the skill receives a properly structured input."
                            ),
                        },
                        "query": {
                            "type": "string",
                            "description": (
                                "Optional. The research question or extraction focus. "
                                "Used alongside 'url' to guide content extraction."
                            ),
                        },
                    },
                    "required": ["skill_name"],
                },
            },
        }

    def _get_parameters_schema(self) -> dict:
        return self.get_schema()["function"]["parameters"]
