"""
Main LocalAgent class – orchestrates LLM, tools, skills, and memory.
"""
from __future__ import annotations

import logging
import os
import uuid
from urllib.parse import urlparse
from typing import Any, Dict, Iterator, List, Optional

from local_agent.core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from local_agent.core.config import get_settings
from local_agent.core.graph import create_agent_graph
from local_agent.core.react import ToolExecutionError
from local_agent.core.state import AgentState
from local_agent.llm.providers.ollama import OllamaProvider
from local_agent.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class LocalAgent:
    """
    Top-level agent facade.

    Typical usage::

        agent = LocalAgent.create(model="qwen2.5:7b")
        print(agent.chat("List my home directory"))

        for token in agent.stream("Summarise README.md"):
            print(token, end="", flush=True)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        skill: Optional[str] = None,
        tools: Optional[List[Any]] = None,
        system_prompt: Optional[str] = None,
        user_id: str = "default",
        session_id: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.settings = get_settings()

        # Windows machines frequently have a system proxy configured.  httpx
        # honours that proxy by default, which can make a local Ollama server
        # appear unavailable even though it is listening on port 11434.
        # Keep this limited to the standard local Ollama endpoint; a custom
        # remote OLLAMA_BASE_URL must remain untouched.
        if provider == "ollama" or (
            provider is None and self.settings.llm_provider == "ollama"
        ):
            self._prepare_windows_ollama_environment()
        
        # Smart provider selection if not specified
        if provider is None:
            provider = self._auto_select_provider()
        
        self.provider_type = provider
        
        # Set default model based on provider
        if self.provider_type == "openai":
            self.model = model or self.settings.openai_default_model
        elif self.provider_type == "wanqing":
            self.model = model or self.settings.wanqing_default_model
        elif self.provider_type == "claude_code":
            self.model = model or self.settings.claude_code_default_model
        else:
            self.model = model or self.settings.ollama_default_model
            
        # None means "auto-select via LLM"; explicit string means user pinned a skill
        self._user_pinned_skill = skill
        self.active_skill = skill
        self.custom_tools = tools or []
        # Set transiently while executing a TaskPlanner step that delegates to
        # a skill.  It is deliberately not user-configurable state.
        self._plan_nested_skill: Optional[str] = None
        self.system_prompt = system_prompt
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())

        # Initialize the appropriate LLM provider
        if self.provider_type == "openai":
            from local_agent.llm.providers.openai import OpenAIProvider
            self.llm_provider = OpenAIProvider(model=self.model)
        elif self.provider_type == "wanqing":
            from local_agent.llm.providers.wanqing import WanqingProvider
            self.llm_provider = WanqingProvider(model=self.model)
        elif self.provider_type == "claude_code":
            from local_agent.llm.providers.claude_code import ClaudeCodeProvider
            self.llm_provider = ClaudeCodeProvider(model=self.model)
        else:
            self.llm_provider = OllamaProvider(model=self.model)
            
        self.tool_registry = ToolRegistry()
        self._graph = None
        self._conversation_history: List[BaseMessage] = []

        logger.info(
            "LocalAgent initialised (provider=%s, model=%s, skill=%s, session=%s)",
            self.provider_type, self.model, self.active_skill, self.session_id,
        )

    def _prepare_windows_ollama_environment(self) -> None:
        """Make the standard local Ollama endpoint bypass Windows proxies."""
        if os.name != "nt":
            return

        parsed_url = urlparse(self.settings.ollama_base_url)
        if parsed_url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            return

        base_url = "http://127.0.0.1:11434"
        self.settings.ollama_base_url = base_url
        os.environ["OLLAMA_BASE_URL"] = base_url

        bypass_hosts = ("localhost", "127.0.0.1")
        existing = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
        entries = [item.strip() for item in existing.split(",") if item.strip()]
        entries_lower = {item.lower() for item in entries}
        entries.extend(host for host in bypass_hosts if host not in entries_lower)
        no_proxy = ",".join(entries)
        # Some tools read the lowercase spelling, while Windows users usually
        # configure the uppercase spelling.  Keep both consistent.
        os.environ["NO_PROXY"] = no_proxy
        os.environ["no_proxy"] = no_proxy

        logger.debug(
            "Prepared local Ollama environment: OLLAMA_BASE_URL=%s, NO_PROXY=%s",
            base_url,
            no_proxy,
        )

    def _auto_select_provider(self) -> str:
        """
        Automatically select an available LLM provider.
        
        Priority:
          1. Ollama (local, preferred if available)
          2. Wanqing (if API key is configured)
          3. OpenAI (if API key is configured)
          4. Claude Code (if API key is configured)
          5. Fallback to config default
        
        Returns:
            Provider name: "ollama", "openai", "wanqing", or "claude_code"
        
        Raises:
            RuntimeError: If no provider is available
        """
        # An explicitly configured provider is strict.  Falling back to a
        # cloud provider when a local Ollama service is unavailable is both
        # surprising and can send requests outside the user's machine.
        if self.settings.llm_provider:
            if self.settings.llm_provider == "ollama":
                if self._check_ollama_available():
                    logger.info("Using configured provider: ollama")
                    return "ollama"
                raise RuntimeError(
                    "Configured provider 'ollama' is unavailable at "
                    f"{self.settings.ollama_base_url}. Start Ollama with "
                    "'ollama serve' and verify it with 'ollama list'."
                )
            elif self.settings.llm_provider == "openai":
                if self._check_openai_available():
                    logger.info("Using configured provider: openai")
                    return "openai"
                raise RuntimeError("Configured provider 'openai' has no API key configured.")
            elif self.settings.llm_provider == "wanqing":
                if self._check_wanqing_available():
                    logger.info("Using configured provider: wanqing")
                    return "wanqing"
                raise RuntimeError("Configured provider 'wanqing' has no API key configured.")
            elif self.settings.llm_provider == "claude_code":
                if self._check_claude_code_available():
                    logger.info("Using configured provider: claude_code")
                    return "claude_code"
                raise RuntimeError("Configured provider 'claude_code' has no API key configured.")
            raise RuntimeError(
                f"Unsupported configured LLM provider: {self.settings.llm_provider!r}."
            )
        
        # Fallback: try Ollama first (local), then Wanqing, then OpenAI, then Claude Code
        if self._check_ollama_available():
            logger.info("Auto-selected provider: ollama (local)")
            return "ollama"
        
        if self._check_wanqing_available():
            logger.info("Auto-selected provider: wanqing (API key configured)")
            return "wanqing"
        
        if self._check_openai_available():
            logger.info("Auto-selected provider: openai (API key configured)")
            return "openai"
        
        if self._check_claude_code_available():
            logger.info("Auto-selected provider: claude_code (API key configured)")
            return "claude_code"
        
        # No provider available
        raise RuntimeError(
            "No LLM provider available. Please:\n"
            "  1. Start Ollama: ollama serve\n"
            "  2. OR set OPENAI_API_KEY or WANQING_API_KEY environment variable\n"
            "  3. OR set CLAUDE_CODE_API_KEY environment variable\n"
            "  4. OR configure provider in config.yaml"
        )
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available and responding."""
        try:
            provider = OllamaProvider()
            return provider.check_connection()
        except Exception:
            return False
    
    def _check_openai_available(self) -> bool:
        """Check if OpenAI is configured (has API key)."""
        return bool(self.settings.openai_api_key and self.settings.openai_api_key.strip())

    def _check_wanqing_available(self) -> bool:
        """Check if Wanqing is configured (has API key)."""
        return bool(self.settings.wanqing_api_key and self.settings.wanqing_api_key.strip())

    def _check_claude_code_available(self) -> bool:
        """Check if Claude Code is configured (has API key)."""
        return bool(self.settings.claude_code_api_key and self.settings.claude_code_api_key.strip())

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _resolve_active_skill(self, message: str) -> Optional[str]:
        """
        Determine which skill to activate for the given message.

        - If the user pinned a skill (via --skill or /skill), always use it.
        - Otherwise, use SkillSelector to automatically pick the best skill via LLM.
        - Returns None when no skill is needed (default agent behaviour).
        """
        # User explicitly pinned a skill — honour it without LLM overhead
        if self._user_pinned_skill is not None:
            return self._user_pinned_skill

        # Auto-select via LLM
        try:
            from local_agent.skills.selector import SkillSelector
            from local_agent.skills.registry import SkillRegistry

            available = SkillRegistry().get_all()
            if not available:
                return None

            # Short explicit search requests are a reliable web-research match.
            # This avoids relying on a small local model to emit perfect JSON.
            top_level_names = {
                skill.get_name()
                for skill in available
                if not getattr(getattr(skill, "parsed_config", None), "is_sub_skill", False)
            }
            research_terms = (
                "搜索", "搜一下", "找一下", "查找", "查询", "调研",
                "检索", "资料", "最新", "推荐", "research", "search",
            )
            if "web_researcher" in top_level_names and any(
                term in message.lower() for term in research_terms
            ):
                logger.info("Selected web_researcher using deterministic research-query match")
                return "web_researcher"

            selector = SkillSelector()
            selected = selector.select_skills(
                query=message,
                llm_provider=self.llm_provider,
                available_skills=available,
            )
            return selected[0] if selected else None
        except Exception as exc:
            logger.warning("Skill auto-selection failed (non-fatal): %s", exc)
            return None

    @staticmethod
    def _has_available_top_level_skills() -> bool:
        """Return whether any user-invokable skill is registered."""
        try:
            from local_agent.skills.registry import SkillRegistry
            return any(
                not getattr(getattr(skill, "parsed_config", None), "is_sub_skill", False)
                for skill in SkillRegistry().get_all()
            )
        except Exception:
            return False

    def _build_graph(self):
        """Compile the LangGraph ReAct graph (call once per skill/config change)."""
        all_tools = self.tool_registry.get_all() + self.custom_tools
        system_prompt = self.system_prompt

        if self.active_skill and system_prompt is None:
            from local_agent.skills.registry import SkillRegistry
            skill = SkillRegistry().get(self.active_skill)
            if skill:
                system_prompt = skill.get_system_prompt()
                skill_tools = skill.get_tools()
                if skill_tools:
                    # When a skill specifies required_tools, use those PLUS all
                    # filesystem / shell / code tools so the agent can always
                    # write files and execute code.
                    # This prevents the case where a skill's curated tool list
                    # accidentally omits fs_write_file / fs_create_dir / shell_run.
                    required_names = {t.name for t in skill_tools}
                    _ALWAYS_INCLUDE_PATTERNS = [
                        "fs_", "shell_", "code_execute_", "code_run_",
                        "project_scaffold", "project_tree",
                        "project_run_command", "project_install_deps",
                        "code_lint_multi", "code_format",
                        "code_run_tests", "repl_eval", "git_",
                    ]
                    extra = [
                        t for t in (self.tool_registry.get_all() + self.custom_tools)
                        if t.name not in required_names
                        and any(t.name.startswith(p) for p in _ALWAYS_INCLUDE_PATTERNS)
                    ]
                    all_tools = skill_tools + extra

                # ── 注入 SkillTool（支持 nested skill 调用）─────────────────
                try:
                    # JsonSkill: parsed_config 直接提供 nested_skills（通过 steps 中 nested_skill 字段）
                    # 从 ParsedSkillConfig 的 steps 中收集所有 nested_skill 引用
                    parsed_cfg = getattr(skill, "parsed_config", None)
                    if parsed_cfg is not None and hasattr(parsed_cfg, "steps"):
                        nested_skills_list = list({
                            step.nested_skill
                            for step in parsed_cfg.steps
                            if getattr(step, "nested_skill", None)
                        })
                    else:
                        nested_skills_list = []
                except Exception:
                    nested_skills_list = []

                if nested_skills_list:
                    from local_agent.skills.skill_tool import SkillTool
                    from local_agent.core.graph import make_debug_hooks
                    llm_raw = self.llm_provider.get_llm(temperature=0.1)
                    skill_tool = SkillTool(
                        llm=llm_raw,
                        allowed_skills=nested_skills_list,
                        max_iterations=10,
                        debug_hooks=make_debug_hooks(),  # 传递调试钩子
                    )
                    # 避免重复注入
                    if not any(t.name == "invoke_skill" for t in all_tools):
                        all_tools = all_tools + [skill_tool]
                    logger.info(
                        "Injected invoke_skill tool for skill '%s' (allowed nested: %s)",
                        self.active_skill, nested_skills_list,
                    )

        # TaskPlanner steps may delegate to a skill even when the overall task
        # has no active parent skill.  Skills are not registered as ordinary
        # tools, so expose the SkillTool adapter explicitly for this step.
        if self._plan_nested_skill:
            from local_agent.skills.skill_tool import SkillTool
            from local_agent.core.graph import make_debug_hooks
            # This is a delegation step, not a general ReAct step.  Restrict
            # its surface to invoke_skill so a weak model cannot fall back to
            # search_web repeatedly instead of opening the supplied URLs.
            all_tools = [SkillTool(
                    llm=self.llm_provider.get_llm(temperature=0.1),
                    allowed_skills=[self._plan_nested_skill],
                    allow_sub_skills=True,
                    max_iterations=10,
                    debug_hooks=make_debug_hooks(),
                )]
            logger.info(
                "Injected invoke_skill tool for TaskPlanner step (allowed nested: %s)",
                self._plan_nested_skill,
            )

        if not all_tools:
            logger.warning("No tools registered – agent will run without tools.")

        llm = self.llm_provider.get_llm(temperature=0.1)
        self._graph = create_agent_graph(llm, all_tools, system_prompt=system_prompt, provider=self.provider_type)

    def _make_initial_state(self, memory_context: str = "") -> AgentState:
        """Build a fresh AgentState snapshot from current conversation history."""
        return AgentState(
            messages=list(self._conversation_history),
            active_skill=self.active_skill,
            tool_results=[],
            memory_context=memory_context,
            user_id=self.user_id,
            session_id=self.session_id,
            iteration_count=0,
            error=None,
        )

    def _ensure_graph(self) -> None:
        if self._graph is None:
            self._build_graph()

    @staticmethod
    def _extract_final_text(messages: List[BaseMessage]) -> str:
        """Return the last non-tool-call AIMessage content."""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                return msg.content or ""
        return messages[-1].content if messages and hasattr(messages[-1], "content") else ""

    # ──────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────

    def _run_with_plan(
        self,
        message: str,
        memory_context: str = "",
    ) -> List[BaseMessage]:
        """
        任务规划驱动的执行核心：
          1. 生成 TaskPlan（步骤列表 + 每步是否激活 skill）
          2. 按步骤顺序执行，每步的 LLM 输入包含：
             - 用户原始请求
             - 整体规划概要
             - 前序步骤结果
             - 当前步骤描述
          3. 收集每步输出，传递给下一步

        Returns:
            最终 messages 列表（用于提取最终回复）
        """
        from local_agent.core.planner import TaskPlanner, TaskPlan, StepResult
        from local_agent.core.debug import should_print_debug

        llm = self.llm_provider.get_llm(temperature=0.1)

        # ── 获取可用 skill 信息 ────────────────────────────────────────────
        skill_descriptions: Dict[str, str] = {}
        available_skills: List[str] = []
        try:
            from local_agent.skills.registry import SkillRegistry
            skills = SkillRegistry().get_all()
            for s in skills:
                name = s.get_name()
                desc = s.get_description()
                available_skills.append(name)
                skill_descriptions[name] = desc
        except Exception:
            pass

        # ── 生成任务规划 ──────────────────────────────────────────────────
        planner = TaskPlanner(llm=llm)
        plan: TaskPlan = planner.plan(
            user_request=message,
            available_skills=available_skills,
            skill_descriptions=skill_descriptions,
        )

        if should_print_debug():
            from local_agent.core.debug import print_task_plan
            print_task_plan(plan)
        logger.info("Task plan: %d steps", len(plan.steps))

        # ── 按步骤执行 ────────────────────────────────────────────────────
        previous_results: List[StepResult] = []
        final_messages: List[BaseMessage] = []

        for step in plan.steps:
            if should_print_debug():
                from local_agent.core.debug import print_task_step
                print_task_step(
                    step_id=str(step.step_id),
                    title=step.title,
                    skill=step.skill,
                )

            # 构建本步骤的上下文 prompt
            step_context = self._build_step_context(
                user_request=message,
                plan=plan,
                previous_results=previous_results,
                current_step=step,
            )

            # A TaskPlanner nested-skill step needs the invoke_skill adapter.
            # Rebuild the graph on every change so it cannot leak into an
            # unrelated step.
            self._plan_nested_skill = step.nested_skill if step.type == "skill" else None

            # 切换 skill
            step_active_skill = None if step.type == "skill" else step.skill
            if step_active_skill != self.active_skill:
                self.active_skill = step_active_skill
                self._graph = None
            else:
                self._graph = None

            self._ensure_graph()
            assert self._graph is not None

            # 用带规划上下文的消息执行本步骤
            step_messages = [HumanMessage(content=step_context)]
            state = {
                "messages": step_messages,
                "memory_context": memory_context,
                "iteration_count": 0,
            }

            try:
                result_state = self._graph.invoke(
                    state,
                    config={"recursion_limit": self.settings.recursion_limit},
                )
                step_msgs: List[BaseMessage] = result_state.get("messages", [])
                step_output = self._extract_final_text(step_msgs)
                step_output = self._append_search_urls(step_output, step_msgs)
                final_messages = step_msgs  # 保留最后一步的完整消息

                previous_results.append(StepResult(
                    step_id=step.step_id,
                    title=step.title,
                    status="success",
                    output=step_output,
                ))

                if should_print_debug():
                    from local_agent.core.debug import _get_console
                    preview = step_output[:200] + "..." if len(step_output) > 200 else step_output
                    _get_console().print(f"[green]✓ Step {step.step_id} Done:[/green] {preview}")

            except Exception as exc:
                logger.error("Step %s failed: %s", step.step_id, exc)
                previous_results.append(StepResult(
                    step_id=step.step_id,
                    title=step.title,
                    status="error",
                    output="",
                    error=str(exc),
                ))
                # 步骤失败：继续下一步，不中断整个流程

        return final_messages

    def _build_step_context(
        self,
        user_request: str,
        plan: Any,
        previous_results: List[Any],
        current_step: Any,
    ) -> str:
        """
        构建单步执行时注入给 LLM 的上下文 prompt。

        结构：
          [用户请求]
          <原始请求>

          [任务规划]
          <规划摘要>

          [前序步骤结果]
          <前序步骤输出>（如有）

          [当前步骤]
          <步骤描述>
        """
        parts = [
            f"[用户请求]\n{user_request}",
            f"[任务规划]\n{plan.to_summary()}",
        ]

        if previous_results:
            prev_lines = ["[前序步骤结果]"]
            for res in previous_results:
                status_icon = "✓" if res.status == "success" else "✗"
                prev_lines.append(f"{status_icon} {res.step_id}. {res.title}")
                if res.output:
                    output_preview = res.output[:500] + "..." if len(res.output) > 500 else res.output
                    prev_lines.append(f"   输出: {output_preview}")
                elif res.error:
                    prev_lines.append(f"   错误: {res.error}")
            parts.append("\n".join(prev_lines))

        skill_hint = f"（使用 skill: {current_step.skill}）" if current_step.skill else ""
        nested_instruction = ""
        if current_step.type == "skill" and current_step.nested_skill:
            nested_instruction = (
                f"\n执行约束: 使用 invoke_skill 调用 {current_step.nested_skill}。"
                "若前序结果中包含多个 URL，逐个调用；不要重新搜索替代 URL 访问。"
            )
        parts.append(
            f"[当前步骤: {current_step.step_id}]{skill_hint}\n"
            f"目标: {current_step.title}\n"
            f"说明: {current_step.description}{nested_instruction}"
        )

        return "\n\n".join(parts)

    @staticmethod
    def _append_search_urls(output: str, messages: List[BaseMessage]) -> str:
        """Carry search URLs across TaskPlanner steps without dumping full HTML."""
        import re

        urls: List[str] = []
        seen = set()
        pattern = re.compile(r"https?://[^\s,\"'<>\]\)]+")
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            for url in pattern.findall(msg.content or ""):
                url = url.rstrip(".,;)")
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
        if not urls:
            return output
        url_block = "\n".join(f"- {url}" for url in urls[:5])
        return f"{output}\n\n[可供后续访问的 URL]\n{url_block}".strip()

    def _run_skill_executor(self, skill_name: str, message: str) -> str:
        """
        通过 SkillExecutor 步骤化执行一个 JSON skill。

        当 skill 具有 parsed_config（即 skill.json 定义的多步骤 skill）时，
        使用 SkillExecutor 按步骤隔离执行，而不是直接走 ReActEngine。

        这是 code_developer 等复杂 skill 的正确执行路径，确保：
          Planning → Architecture → Coding → Testing → Review → ... 等步骤按序执行。

        Args:
            skill_name: skill 名称
            message:    用户原始请求（作为 task 输入）

        Returns:
            skill 执行的最终文本结果
        """
        from local_agent.skills.registry import SkillRegistry
        from local_agent.skills.executor import SkillExecutor
        from local_agent.core.debug import should_print_debug, print_skill_activation

        registry = SkillRegistry()
        skill_obj = registry.get(skill_name)
        if skill_obj is None:
            raise RuntimeError(f"Skill '{skill_name}' not found in registry")

        parsed_config = getattr(skill_obj, "parsed_config", None)
        if parsed_config is None:
            raise RuntimeError(
                f"Skill '{skill_name}' has no parsed_config. "
                "Only JSON skills (skill.json) support SkillExecutor."
            )

        if should_print_debug():
            print_skill_activation(skill_name)

        llm = self.llm_provider.get_llm(temperature=0.1)
        from local_agent.core.graph import make_debug_hooks
        executor = SkillExecutor(
            tool_registry=self.tool_registry,
            llm=llm,
            debug_hooks=make_debug_hooks(),
        )

        logger.info(
            "Running skill '%s' via SkillExecutor (%d steps)",
            skill_name, len(parsed_config.steps),
        )

        result_ctx = executor.execute(
            parsed_config=parsed_config,
            initial_input={"task": message, "query": message},
            global_context=message,
        )

        # 提取最终结果。JSON skill 已在顶层 output_spec 声明用户可见的
        # 最终字段（例如 web_researcher 的 report）；不能把执行上下文中的
        # 搜索关键词、原始工具结果等中间值拼进终端回复。
        final_result = self._select_skill_final_output(parsed_config, result_ctx)
        if not final_result:
            # 仅在 skill 未声明或未产出任何顶层输出时才保留旧兜底行为。
            parts = [str(v) for k, v in result_ctx.items() if v and k != "task"]
            final_result = "\n\n".join(parts)

        return final_result

    @staticmethod
    def _select_skill_final_output(parsed_config: Any, result_ctx: Dict[str, Any]) -> str:
        """Select the user-facing output declared by a JSON skill."""
        for field_name in getattr(parsed_config, "output_spec", {}) or {}:
            value = result_ctx.get(field_name)
            if value:
                return str(value)
        return str(result_ctx.get("result", "") or "")

    def chat(self, message: str, memory_context: str = "") -> str:
        """Send *message* and return the complete response string.

        执行逻辑（优先级由高到低）：
          1. 若用户已 pin skill，直接使用该 skill 执行。
          2. 尝试 SkillSelector 匹配 skill，若匹配到则：
             a. 若 skill 有 parsed_config（JSON skill）→ 走 SkillExecutor 步骤化执行
             b. 否则走普通 ReActEngine（注入 system_prompt）
          3. 若无 skill 匹配，调用 TaskPlanner 生成规划后按步骤执行。
        """
        from local_agent.core.debug import (
            print_debug_separator,
            print_model_input,
            print_skill_activation,
            should_print_debug,
        )

        self._conversation_history.append(HumanMessage(content=message))

        try:
            # ── Step 1: Skill 优先匹配 ─────────────────────────────────────
            resolved_skill = self._resolve_active_skill(message)

            # Planning is the last resort: it is only enabled when the system
            # has no usable top-level skills at all.  If skills exist but none
            # matches, run the default agent directly rather than generating a
            # plan that can bypass the registered skill architecture.
            if resolved_skill is not None or self._has_available_top_level_skills():
                if resolved_skill is None:
                    logger.info("Skills are available but none matched; using default ReAct agent without TaskPlanner")
                    self.active_skill = None
                    self._graph = None
                # 匹配到 skill，先判断是否走 SkillExecutor
                if resolved_skill != self.active_skill:
                    self.active_skill = resolved_skill
                    self._graph = None

                logger.info("Skill matched: %s — checking execution path", self.active_skill)

                # ── 判断是否为 JSON skill（有 parsed_config）────────────────
                from local_agent.skills.registry import SkillRegistry
                _active_skill_name: str = self.active_skill or ""
                skill_obj = SkillRegistry().get(_active_skill_name)
                parsed_config = getattr(skill_obj, "parsed_config", None) if skill_obj else None

                if parsed_config is not None:
                    # JSON skill → SkillExecutor 步骤化执行
                    _step_count = len(parsed_config.steps) if hasattr(parsed_config, "steps") else 0
                    logger.info(
                        "Skill '%s' has parsed_config (%d steps) → using SkillExecutor",
                        _active_skill_name, _step_count,
                    )
                    response = self._run_skill_executor(_active_skill_name, message)

                    if should_print_debug():
                        print_debug_separator()

                    return response
                else:
                    # 普通 skill → ReActEngine（注入 system_prompt）
                    if should_print_debug():
                        print_skill_activation(self.active_skill)

                    logger.info("Skill '%s' has no parsed_config → using ReActEngine", self.active_skill)
                    self._ensure_graph()
                    assert self._graph is not None

                    result = self._graph.invoke(
                        dict(self._make_initial_state(memory_context)),
                        config={"recursion_limit": self.settings.recursion_limit},
                    )
                    final_messages_raw = result.get("messages", [])
                    final_messages = list(final_messages_raw) if final_messages_raw else []

                    response = self._extract_final_text(final_messages)

                    if should_print_debug():
                        print_debug_separator()

                    # Update conversation history
                    self._conversation_history.extend(
                        final_messages[len(self._conversation_history):]
                    )
                    return response

            else:
                # ── Step 2: 无任何可用 skill，使用任务规划驱动执行 ─────────
                logger.info("No top-level skill available — falling back to TaskPlanner")
                self.active_skill = None
                self._graph = None

                final_messages = self._run_with_plan(message, memory_context)

                if not final_messages:
                    # 最终 fallback：直接执行（无 skill，无规划）
                    self._ensure_graph()
                    assert self._graph is not None
                    result = self._graph.invoke(
                        dict(self._make_initial_state(memory_context)),
                        config={"recursion_limit": self.settings.recursion_limit},
                    )
                    final_messages_raw = result.get("messages", [])
                    final_messages = list(final_messages_raw) if final_messages_raw else []

                response = self._extract_final_text(final_messages)

                if should_print_debug():
                    print_debug_separator()

                # Update conversation history
                self._conversation_history.extend(
                    final_messages[len(self._conversation_history):]
                )
                return response

        except ToolExecutionError:
            raise
        except Exception as exc:
            logger.error("Error during agent execution: %s", exc, exc_info=True)
            return f"Error: {exc}"

    def stream(self, message: str, memory_context: str = "") -> Iterator[str]:
        """
        Send *message* and yield response tokens as they arrive (token-level streaming).

        执行逻辑（优先级由高到低）：
          1. 若用户已 pin skill 或 SkillSelector 匹配到 skill，且 skill 有 parsed_config：
             走 SkillExecutor 步骤化执行（同步执行，逐步 yield 进度）
          2. 若匹配到 skill 但无 parsed_config：走普通 ReActEngine 流式执行
          3. 若无 skill 匹配，调用 TaskPlanner 生成规划后按步骤流式执行。
        规划阶段同步执行（规划本身不流式），规划完成后流式执行各步骤。
        """
        from local_agent.core.debug import (
            print_debug_separator,
            print_model_input,
            print_skill_activation,
            should_print_debug,
        )
        from local_agent.core.planner import TaskPlanner, TaskPlan, StepResult

        self._conversation_history.append(HumanMessage(content=message))

        try:
            # ── Step 1: Skill 优先匹配 ─────────────────────────────────────
            resolved_skill = self._resolve_active_skill(message)

            # With registered skills, an unmatched query uses the default
            # ReAct path. TaskPlanner is reserved for a completely skillless
            # runtime.
            if resolved_skill is not None or self._has_available_top_level_skills():
                if resolved_skill is None:
                    logger.info("Skills are available but none matched; streaming with default ReAct agent without TaskPlanner")
                    self.active_skill = None
                    self._graph = None
                # 匹配到 skill，先判断是否走 SkillExecutor
                if resolved_skill != self.active_skill:
                    self.active_skill = resolved_skill
                    self._graph = None

                logger.info("Skill matched: %s — checking execution path (stream)", self.active_skill)

                # ── 判断是否为 JSON skill（有 parsed_config）────────────────
                from local_agent.skills.registry import SkillRegistry
                _active_skill_name2: str = self.active_skill or ""
                skill_obj2 = SkillRegistry().get(_active_skill_name2)
                parsed_config = getattr(skill_obj2, "parsed_config", None) if skill_obj2 else None

                if parsed_config is not None:
                    # JSON skill → SkillExecutor 步骤化执行（同步，逐步 yield 进度）
                    _step_count2 = len(parsed_config.steps) if hasattr(parsed_config, "steps") else 0
                    logger.info(
                        "Skill '%s' has parsed_config (%d steps) → using SkillExecutor (stream)",
                        _active_skill_name2, _step_count2,
                    )
                    yield f"\n[Skill: {_active_skill_name2}] Starting step-by-step execution...\n"

                    try:
                        result_text = self._run_skill_executor(_active_skill_name2, message)
                        yield result_text
                    except Exception as exc:
                        logger.error("SkillExecutor stream failed: %s", exc, exc_info=True)
                        yield f"\n\nError in skill execution: {exc}\n"

                    if should_print_debug():
                        print_debug_separator()
                    return

                else:
                    # 普通 skill → ReActEngine 流式执行（注入 system_prompt）
                    if should_print_debug():
                        print_skill_activation(self.active_skill)

                    logger.info("Skill '%s' has no parsed_config → using ReActEngine (stream)", self.active_skill)
                    self._ensure_graph()
                    assert self._graph is not None

                    final_messages_list: List[BaseMessage] = []
                    announced_tool_ids: set[str] = set()
                    state = {
                        "messages": list(self._conversation_history),
                        "memory_context": memory_context,
                        "iteration_count": 0,
                    }

                    try:
                        for event in self._graph.stream(
                            state,
                            config={"recursion_limit": self.settings.recursion_limit},
                            stream_mode=["messages", "values"],
                        ):
                            mode, data = event
                            if mode == "messages":
                                chunk, _metadata = data
                                if isinstance(chunk, AIMessageChunk):
                                    for tc in (chunk.tool_calls or []):
                                        tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "") or tc.get("name", "")
                                        if tc_id and tc_id not in announced_tool_ids:
                                            announced_tool_ids.add(tc_id)
                                            tool_name = tc.name if hasattr(tc, "name") else tc.get("name", "unknown")
                                            from local_agent.core.debug import print_tool_announce
                                            print_tool_announce(tool_name)
                                    token: str = chunk.content or ""
                                    if token and not chunk.tool_calls:
                                        yield token
                                elif isinstance(chunk, ToolMessage):
                                    from local_agent.core.debug import print_tool_completed
                                    print_tool_completed()
                            elif mode == "values":
                                final_messages_list = data.get("messages", [])

                        if final_messages_list:
                            self._conversation_history = final_messages_list

                    except Exception as exc:
                        logger.error("Skill stream failed: %s", exc, exc_info=True)
                        yield f"\n\nError: {exc}\n"

                    if should_print_debug():
                        print_debug_separator()
                    return

            # ── Step 2: 无任何可用 skill，使用任务规划驱动流式执行 ────────
            logger.info("No top-level skill available — falling back to TaskPlanner (stream)")
            self.active_skill = None
            self._graph = None

            llm = self.llm_provider.get_llm(temperature=0.1)

            # ── 获取可用 skill 信息 ────────────────────────────────────────
            skill_descriptions: Dict[str, str] = {}
            available_skills: List[str] = []
            try:
                from local_agent.skills.registry import SkillRegistry
                skills = SkillRegistry().get_all()
                for s in skills:
                    name = s.get_name()
                    desc = s.get_description()
                    available_skills.append(name)
                    skill_descriptions[name] = desc
            except Exception:
                pass

            # ── 同步生成任务规划 ──────────────────────────────────────────
            planner = TaskPlanner(llm=llm)
            plan: TaskPlan = planner.plan(
                user_request=message,
                available_skills=available_skills,
                skill_descriptions=skill_descriptions,
            )

            from local_agent.core.debug import print_task_plan
            print_task_plan(plan)

            # ── 按步骤流式执行 ────────────────────────────────────────────
            previous_results: List[StepResult] = []
            final_messages_list: List[BaseMessage] = []
            announced_tool_ids: set[str] = set()

            for step in plan.steps:
                from local_agent.core.debug import print_task_step
                print_task_step(
                    step_id=str(step.step_id),
                    title=step.title,
                    skill=step.skill,
                )

                # 构建本步骤上下文 prompt
                step_context = self._build_step_context(
                    user_request=message,
                    plan=plan,
                    previous_results=previous_results,
                    current_step=step,
                )

                # 嵌套 skill 通过 invoke_skill 执行，不切换为顶层 skill。
                self._plan_nested_skill = step.nested_skill if step.type == "skill" else None
                step_active_skill = None if step.type == "skill" else step.skill
                if step_active_skill != self.active_skill:
                    self.active_skill = step_active_skill
                    self._graph = None
                else:
                    self._graph = None

                self._ensure_graph()
                assert self._graph is not None

                step_messages = [HumanMessage(content=step_context)]
                state = {
                    "messages": step_messages,
                    "memory_context": memory_context,
                    "iteration_count": 0,
                }

                step_output_tokens: List[str] = []
                try:
                    for event in self._graph.stream(
                        state,
                        config={"recursion_limit": self.settings.recursion_limit},
                        stream_mode=["messages", "values"],
                    ):
                        mode, data = event

                        if mode == "messages":
                            chunk, _metadata = data
                            if isinstance(chunk, AIMessageChunk):
                                for tc in (chunk.tool_calls or []):
                                    tc_id = tc.id if hasattr(tc, "id") else tc.get("id", "") or tc.get("name", "")
                                    if tc_id and tc_id not in announced_tool_ids:
                                        announced_tool_ids.add(tc_id)
                                        tool_name = tc.name if hasattr(tc, "name") else tc.get("name", "unknown")
                                        from local_agent.core.debug import print_tool_announce
                                        print_tool_announce(tool_name)

                                token: str = chunk.content or ""
                                if token and not chunk.tool_calls:
                                    step_output_tokens.append(token)
                                    yield token

                            elif isinstance(chunk, ToolMessage):
                                from local_agent.core.debug import print_tool_completed
                                print_tool_completed()

                        elif mode == "values":
                            final_messages_list = data.get("messages", [])

                    step_output: str = "".join(step_output_tokens)
                    step_output = self._append_search_urls(step_output, final_messages_list)
                    previous_results.append(StepResult(
                        step_id=step.step_id,
                        title=step.title,
                        status="success",
                        output=step_output,
                    ))

                except Exception as exc:
                    logger.error("Step %s streaming failed: %s", step.step_id, exc)
                    previous_results.append(StepResult(
                        step_id=step.step_id,
                        title=step.title,
                        status="error",
                        error=str(exc),
                    ))
                    yield f"\n[Step {step.step_id} Error: {exc}]\n"

            # Update conversation history
            if final_messages_list:
                self._conversation_history = final_messages_list

            if should_print_debug():
                print_debug_separator()

        except ToolExecutionError:
            raise
        except Exception as exc:
            logger.error("Error during streaming: %s", exc, exc_info=True)
            import traceback
            traceback.print_exc()
            yield f"\n\nError: {exc}\n"

    def reset_conversation(self) -> None:
        """Discard the current conversation history."""
        self._conversation_history = []
        logger.debug("Conversation history reset")

    def get_conversation_history(self) -> List[BaseMessage]:
        return list(self._conversation_history)

    def validate_model(self, model_name: str, provider: Optional[str] = None) -> tuple[bool, str]:
        """
        Check whether *model_name* is usable before switching to it.

        First checks whether the model is in the configured models list from
        config.yaml. If the model is in the list, it is accepted immediately
        without a live network test (fast path).

        For models NOT in the configured list:
          - Ollama: checks the model exists in the Ollama server's local list.
          - OpenAI / Wanqing / Claude Code: rejected (must be in config.yaml).

        Returns:
            (ok, error_message) – ok is True when the model is valid.
        """
        target_provider = provider or self.provider_type

        # Fast path: model is in the user-configured list → accept immediately
        configured = self.settings.get_configured_models(target_provider)
        if model_name in configured:
            return True, ""

        # Ollama fallback: check live server list
        if target_provider == "ollama":
            available = self.llm_provider.list_models()
            if model_name in available:
                return True, ""
            available_str = ", ".join(configured) if configured else "(none configured)"
            return False, (
                f"Model '{model_name}' is not in your configured models list. "
                f"Configured models: {available_str}. "
                f"Add it under ollama.models in config.yaml, or run: ollama pull {model_name}"
            )

        # Cloud providers: must be in config.yaml models list
        configured_str = ", ".join(configured) if configured else "(none configured)"
        return False, (
            f"Model '{model_name}' is not in your configured models list for '{target_provider}'. "
            f"Configured models: {configured_str}. "
            f"Add it under {target_provider}.models in config.yaml to enable it."
        )

    def set_skill(self, skill_name: Optional[str]) -> None:
        """Switch the active skill and invalidate the compiled graph.
        
        Calling this pins the skill (user override), bypassing auto-selection.
        Pass None to re-enable auto-selection for subsequent messages.
        """
        self._user_pinned_skill = skill_name
        self.active_skill = skill_name
        self._graph = None
        logger.info("Active skill pinned to: %s", skill_name)

    def get_available_tools(self) -> List[Dict[str, Any]]:
        return self.tool_registry.get_tool_info()

    # ──────────────────────────────────────────────────────────────────────
    # Factory
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        model: Optional[str] = None,
        skill: Optional[str] = None,
        provider: Optional[str] = None,
        load_all_tools: bool = True,
        extra_tool_dirs: Optional[List[str]] = None,
        extra_skill_dirs: Optional[List[str]] = None,
        load_mcp: bool = True,
    ) -> "LocalAgent":
        """
        Convenience factory: load built-in tools + skills + MCP tools, then return an agent.

        Args:
            model: Model name (default: from settings based on provider).
            skill: Skill name to activate.
            provider: LLM provider ("ollama" or "openai", default: from settings).
            load_all_tools: Load built-in tools (default True).
            extra_tool_dirs: Additional directories to scan for custom tools.
            extra_skill_dirs: Additional directories to scan for custom skills.
            load_mcp: Load tools from config/mcp.json (default True).
        """
        from local_agent.skills.loader import SkillLoader
        from local_agent.tools.builtin import load_all_builtin_tools
        from local_agent.tools.registry import ToolRegistry

        settings = get_settings()

        if load_all_tools:
            load_all_builtin_tools()

        # Load extra tools from user-defined directories
        if extra_tool_dirs:
            reg = ToolRegistry()
            for d in extra_tool_dirs:
                reg.load_from_directory(d)

        # Load built-in skills + extra skill directories
        loader = SkillLoader()
        loader.load_builtin_skills()
        for d in (extra_skill_dirs or []):
            loader.load_from_directory(d)

        # Load MCP tools (stdio / SSE servers configured in config/mcp.json)
        if load_mcp and settings.mcp_enabled:
            try:
                from local_agent.mcp import MCPManager
                manager = MCPManager.from_config_path(settings.mcp_config_path)
                mcp_tools = manager.load_all()
                if mcp_tools:
                    reg = ToolRegistry()
                    for tool in mcp_tools:
                        reg.register(tool)
                    logger.info("Loaded %d MCP tool(s) into ToolRegistry", len(mcp_tools))
            except Exception as exc:
                logger.warning("MCP tool loading failed (non-fatal): %s", exc)

        return cls(model=model, skill=skill, provider=provider)
