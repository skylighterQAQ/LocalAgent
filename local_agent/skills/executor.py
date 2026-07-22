"""
local_agent.skills.executor
=============================
SkillExecutor – 基于 ParsedSkillConfig 的步骤隔离执行器。

核心设计原则（解决 LLM 输入过多问题）：
  每次 LLM 调用的消息结构：
    SystemMessage:
      [全局背景 - global_context]          ← 用户原始任务等基础背景（最精简）
      [Skill 背景 - skill_overview]        ← skill 的 overview（1-2 句）
      [Skill 公共 prompt - skill_prompt]   ← skill 的全局行为规范（每步均注入）
      [当前步骤指令]                        ← 当前 StepSpec 的 description
      [当前步骤专属 prompt - step_prompt]  ← 仅本步骤相关的约束和期望
      [期望输入规范]                        ← input_spec 说明
      [期望输出规范]                        ← output_spec 说明
      [输入输出校验失败处理策略]            ← validation policy

    HumanMessage:
      [上一步骤的输出结果]             ← 仅上一步 output_mapping 的结果值，不含上一步输入细节
      [当前步骤的输入]                 ← 从 input_mapping 提取的本步所需变量

  关键约束：
    1. 上一步的输入细节不传，只传上一步的输出结果
    2. 每次全新构建消息列表，不复用/追加历史消息
    3. AGENT 类型步骤内部的工具调用轮次消息只在本步骤执行期间累积，
       步骤结束后只保留最终输出传给下一步

步骤类型路由：
  TOOL  → 直接调用工具，不启动 LLM
  LLM   → 单次 LLM 调用，消息按上述格式全新构建
  AGENT → 子 ReActEngine，只注入本步骤的 tools + system_prompt，初始消息按上述格式构建
  SKILL → 调用 SkillTool（嵌套 skill）
"""
from __future__ import annotations

import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from local_agent.core.subagent.context import ExecutionContext
from local_agent.skills.parsed_config import ParsedSkillConfig, StepSpec, StepType

logger = logging.getLogger(__name__)


def _print_step_fatal(step_id: str, exc: Exception) -> None:
    """Print a prominent error to stderr when a skill step fatally fails."""
    msg = f"Skill step [{step_id}] failed: {exc}"
    print(
        f"\n\033[1;31m[Step Fatal Error]\033[0m {msg}\n\033[33mExiting skill execution.\033[0m\n",
        file=sys.stderr,
    )


def _print_validation_warning(step_id: str, kind: str, missing: List[str]) -> None:
    """Print a validation warning to stderr."""
    print(
        f"\n\033[1;33m[Validation Warning]\033[0m Step [{step_id}] "
        f"{kind} 缺少期望字段: {missing}\n",
        file=sys.stderr,
    )


def _looks_like_file_path(value: str) -> bool:
    """
    判断一个字符串是否看起来像一个文件路径（而非普通值）。

    条件：
    - 字符串以 '/' 开头（绝对路径）或以 './' '../' 开头（相对路径）
    - 字符串中包含 '/' 且以常见文档扩展名结尾（.md, .txt, .json, .yaml, .py 等）
    - 长度合理（不是一个极长的 JSON 字符串）
    """
    import os  # noqa: PLC0415
    if not value:
        return False
    # 超长字符串（> 500 chars）不可能是路径
    if len(value) > 500:
        return False
    # 含换行符的不是路径
    if '\n' in value or '\r' in value:
        return False
    # 必须包含 '/' 才可能是路径
    if '/' not in value:
        return False
    # 绝对路径或相对路径前缀
    if value.startswith('/') or value.startswith('./') or value.startswith('../'):
        # 还需要确认是已存在的文件（避免将路径格式的描述字符串也读取）
        return os.path.isfile(value)
    return False


class SkillExecutor:
    """
    按 ParsedSkillConfig 中的步骤隔离执行一个 skill。

    每步 LLM 调用只接收：
      - 全局背景（用户原始 query 等）
      - skill overview（1-2 句）
      - skill 公共 prompt（全局行为规范）
      - 当前步骤指令
      - 当前步骤专属 prompt
      - 期望输入输出规范
      - 上一步骤的输出结果
      - 当前步骤的输入变量

    不传递：
      - 完整对话历史
      - 上一步骤的输入细节
      - 其他步骤的 step_prompt
    """

    def __init__(self, tool_registry: Any, llm: Any, debug_hooks: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            tool_registry: ToolRegistry 实例，用于解析工具名到工具对象
            llm:           LLM 实例（支持 .invoke(messages)）
            debug_hooks:   调试钩子字典（before_llm / after_llm / after_tool），
                           由 graph.make_debug_hooks() 生成，传入后 LLM 和工具
                           调用均会产生与主 agent 一致的 debug 日志输出。
        """
        self._tool_registry = tool_registry
        self._llm = llm
        self._debug_hooks: Dict[str, Any] = debug_hooks or {}
        self._current_skill_name: str = "unknown"

    # ── 公开 API ──────────────────────────────────────────────────────────────

    def execute(
        self,
        parsed_config: ParsedSkillConfig,
        initial_input: Dict[str, Any],
        global_context: str = "",
    ) -> Dict[str, Any]:
        """
        按步骤顺序执行 skill，返回最终上下文变量。

        Args:
            parsed_config:  经 SkillParser 解析的结构化配置
            initial_input:  初始输入变量（如 {"query": "..."}）
            global_context: 精简的全局背景字符串（用户原始任务等），注入每步的 SystemMessage

        Returns:
            执行结束时的上下文变量字典（包含所有步骤的 output_mapping 写入的变量）
        """
        from local_agent.core.debug import print_skill_step_start

        ctx = ExecutionContext(initial_data=initial_input)
        prev_step_output: Optional[Dict[str, Any]] = None  # 上一步骤的输出
        skill_name = getattr(parsed_config, "skill_name", "unknown")
        # 在步骤执行期间将 skill_name 暂存到实例上，供 _run_skill_step 使用
        self._current_skill_name = skill_name
        for step_spec in parsed_config.steps:
            logger.info(
                "SkillExecutor: executing step [%s] %s (type=%s)",
                step_spec.id, step_spec.name, step_spec.type,
            )

            # 1. 从上下文提取本步骤所需输入（仅 input_mapping 声明的变量）
            step_input = self._resolve_step_input(step_spec, ctx)

            # 1b. 对于 TOOL 步骤，自动推断辅助变量（如 target_file_dir）
            if step_spec.type == StepType.TOOL:
                # 若 tool_params 中引用了 {target_file_dir}，但 step_input 中没有或值为 None，
                # 自动从 target_file_path 推断（取父目录）。
                # 注意：必须检查值是否为 None（不能只检查 key 是否存在），因为 input_mapping 可能
                # 声明了 target_file_dir 但上下文中该变量尚未赋值（值为 None）。
                if step_input.get("target_file_dir") is None and step_input.get("target_file_path"):
                    import os as _os
                    fp = str(step_input["target_file_path"])
                    step_input["target_file_dir"] = _os.path.dirname(fp) or "."

            # ── debug: 打印步骤开始（类型无关，统一在此打印）──────────────────
            print_skill_step_start(
                skill_name=skill_name,
                step_id=step_spec.id,
                step_name=step_spec.name,
                step_type=step_spec.type.value if hasattr(step_spec.type, "value") else str(step_spec.type),
                step_input=step_input,
            )

            # 2. 校验输入
            self._validate_input(step_spec, step_input)

            # module_developer's design step is normally an LLM generation
            # step, but the implementation pass should deterministically reuse
            # the design document produced by the earlier design-only pass.
            precomputed_result: Optional[Dict[str, Any]] = None
            if (
                skill_name == "module_developer"
                and step_spec.id == "step_3"
                and step_spec.type == StepType.LLM
                and step_input.get("module_name")
            ):
                design_path = f"design/{step_input['module_name']}_design.md"
                read_tool = self._tool_registry.get("fs_read_file")
                if read_tool is not None:
                    raw_design = read_tool._run(path=design_path)
                    if isinstance(raw_design, str) and not raw_design.lstrip().lower().startswith("error"):
                        precomputed_result = {
                            "result": raw_design,
                            "module_design": design_path,
                            "module_design_content": raw_design,
                            "module_design_reused": True,
                        }
                        logger.info(
                            "SkillExecutor: reusing existing module design '%s' (%d chars)",
                            design_path,
                            len(raw_design),
                        )

            # 3. 构建本次 LLM 调用的消息（全新构建，不累积）
            # TOOL 步骤不需要 LLM，跳过消息构建；
            # AGENT/LLM 步骤注入 skill_prompt，SKILL 步骤不需要
            is_tool_step = step_spec.type == StepType.TOOL
            include_skill_prompt = not is_tool_step and step_spec.type not in (StepType.SKILL,)
            messages = self._build_messages(
                global_context=global_context,
                skill_overview=parsed_config.to_prompt_overview(),
                skill_prompt=parsed_config.skill_prompt,
                step_spec=step_spec,
                prev_step_output=prev_step_output,
                current_step_input=step_input,
                include_skill_prompt=include_skill_prompt,
            ) if not is_tool_step else []

            # 4. 按步骤类型路由执行
            # ── design_only 框架层强制跳过：若上下文中 design_only=true，直接跳过 SKILL 步骤 ──
            # 二次校验：同时检查 task 字段中是否真的包含 design_only 关键词，
            # 防止 step_1 LLM 幻觉将 design_only 误设为 true 导致代码实现被跳过。
            _task_str = str(ctx.get("task") or "")
            _task_has_design_only = "design_only" in _task_str or "执行模式" in _task_str
            if step_spec.type == StepType.SKILL and ctx.get("design_only") in ("true", True) and _task_has_design_only:
                logger.info(
                    "SkillExecutor: step [%s] skipped (design_only=true, SKILL step bypassed at framework level)",
                    step_spec.id,
                )
                result = {"result": "design_only 模式：跳过代码实现，设计文档已生成"}
                self._validate_output(step_spec, result)
                for ctx_key, result_key in step_spec.output_mapping.items():
                    if result_key in result:
                        ctx.set(ctx_key, result[result_key])
                prev_step_output = result
                continue

            try:
                result = precomputed_result or self._dispatch(step_spec, step_input, messages)
            except Exception as exc:
                result = self._handle_failure(step_spec, exc, ctx)
                if result is None:
                    # on_failure=raise 时已在 _handle_failure 中重新抛出
                    # 其他情况（skip/continue）result 为 {}
                    result = {}

            # 5. 校验输出
            self._validate_output(step_spec, result)

            # 6. 将本步骤输出按 output_mapping 写回上下文
            for ctx_key, result_key in step_spec.output_mapping.items():
                if result_key in result:
                    ctx.set(ctx_key, result[result_key])
                    logger.debug(
                        "SkillExecutor: step [%s] wrote ctx['%s'] = result['%s']",
                        step_spec.id, ctx_key, result_key,
                    )

            # 7. 记录本步骤输出（供下一步使用），不含本步骤输入
            prev_step_output = result

        return ctx.get_all()

    # ── 消息构建 ──────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        global_context: str,
        skill_overview: str,
        skill_prompt: str,
        step_spec: StepSpec,
        prev_step_output: Optional[Dict[str, Any]],
        current_step_input: Dict[str, Any],
        include_skill_prompt: bool = True,
        # 最近截断控制：限制各部分的最大字符数，防止输入过长导致模型超时
        max_skill_prompt_chars: int = 3000,
        max_step_prompt_chars: int = 4000,
        max_prev_output_chars: int = 2000,
        max_file_inject_chars: int = 8000,
    ) -> List[Any]:
        """
        全新构建本次 LLM 调用的消息列表。

        消息结构（已优化，最小化输入 token）：
          SystemMessage:
            [全局背景]                   ← 用户原始 task（精简）
            [Skill 背景]                 ← skill_name + overview（1行）
            [Skill 公共 prompt]          ← 仅 AGENT/LLM 步骤注入，且可跳过重复注入
            [当前步骤指令]               ← 当前步骤 name + description
            [当前步骤专属 prompt]        ← step_prompt（聚焦本步骤操作）
            [期望输入规范]
            [期望输出规范]
            [校验失败处理策略]
          HumanMessage:
            [上一步输出]
            [本步输入]
            [文件内容自动注入]           ← 输入字段中如有文件路径，自动读取

        注意：include_skill_prompt=False 时跳过 skill_prompt 注入，
        用于 TOOL 步骤（无需给 LLM 任何输入）。
        """
        from local_agent.core.messages import HumanMessage, SystemMessage

        # ── SystemMessage ────────────────────────────────────────────────────
        system_parts: List[str] = []

        if global_context:
            system_parts.append(f"[背景]\n{global_context}")

        if skill_overview:
            system_parts.append(f"[Skill]\n{skill_overview}")

        # Skill 公共 prompt（仅在需要时注入，避免每步重复传大块全局规范）
        if include_skill_prompt and skill_prompt:
            # 截断过长的 skill_prompt，防止 context 爆炸导致模型超时
            truncated_skill_prompt = skill_prompt
            if len(skill_prompt) > max_skill_prompt_chars:
                truncated_skill_prompt = skill_prompt[:max_skill_prompt_chars] + f"\n...(截断，原长 {len(skill_prompt)} 字)"
                logger.debug(
                    "_build_messages: skill_prompt truncated from %d to %d chars",
                    len(skill_prompt), max_skill_prompt_chars,
                )
            system_parts.append(f"[Skill 规范]\n{truncated_skill_prompt}")

        # 当前步骤指令
        step_instruction = f"[当前步骤: {step_spec.name}]\n{step_spec.description}"
        system_parts.append(step_instruction)

        # 步骤专属 prompt（优先 step_prompt，其次 system_prompt）
        effective_step_prompt = step_spec.get_effective_step_prompt()
        if effective_step_prompt:
            # 截断过长的步骤 prompt，防止 context 爆炸
            truncated_step_prompt = effective_step_prompt
            if len(effective_step_prompt) > max_step_prompt_chars:
                truncated_step_prompt = effective_step_prompt[:max_step_prompt_chars] + f"\n...(截断，原长 {len(effective_step_prompt)} 字)"
                logger.debug(
                    "_build_messages: step_prompt truncated from %d to %d chars",
                    len(effective_step_prompt), max_step_prompt_chars,
                )
            system_parts.append(f"[步骤指令]\n{truncated_step_prompt}")

        # 期望输入规范（新增）
        if step_spec.input_spec:
            spec_lines = ["[期望输入]"]
            for field, desc in step_spec.input_spec.items():
                spec_lines.append(f"  - {field}: {desc}")
            system_parts.append("\n".join(spec_lines))

        # 期望输出规范（新增）
        if step_spec.output_spec:
            spec_lines = ["[期望输出]"]
            for field, desc in step_spec.output_spec.items():
                spec_lines.append(f"  - {field}: {desc}")
            system_parts.append("\n".join(spec_lines))

        # 校验失败处理策略提示（新增）
        validation_hints = []
        if step_spec.input_validation_policy != "ignore":
            validation_hints.append(
                f"输入校验失败时: {step_spec.input_validation_policy}"
                "（如输入不满足期望，请说明缺少什么信息）"
            )
        if step_spec.output_validation_policy != "ignore":
            validation_hints.append(
                f"输出校验失败时: {step_spec.output_validation_policy}"
                "（请确保输出包含所有期望字段）"
            )
        if validation_hints:
            system_parts.append("[校验策略]\n" + "\n".join(validation_hints))

        system_content = "\n\n".join(system_parts)

        # ── HumanMessage ────────────────────────────────────────────────────
        human_parts: List[str] = []

        if prev_step_output is not None:
            prev_text = _format_value(prev_step_output)
            if prev_text:
                # 截断上一步输出，防止大型工具结果撑爆 context
                if len(prev_text) > max_prev_output_chars:
                    prev_text = prev_text[:max_prev_output_chars] + f"\n...(截断，原长 {len(prev_text)} 字)"
                    logger.debug(
                        "_build_messages: prev_step_output truncated to %d chars",
                        max_prev_output_chars,
                    )
                human_parts.append(f"[上一步骤的输出]\n{prev_text}")

        if current_step_input:
            input_text = _format_value(current_step_input)
            human_parts.append(f"[本步骤输入]\n{input_text}")

        # ── 文件路径自动注入 ──────────────────────────────────────────────────
        # 如果 current_step_input 中某个字段值是一个可读的文件路径，自动读取文件内容
        # 并以 "[<field_name> 文件内容]:\n<content>" 的形式附加到 HumanMessage 中。
        # 这样可以在 LLM 步骤（type: llm）中直接提供文件内容，无需大模型再调工具读文件。
        file_content_parts: List[str] = []
        for field_name, field_value in current_step_input.items():
            if isinstance(field_value, str) and _looks_like_file_path(field_value):
                try:
                    with open(field_value, "r", encoding="utf-8") as _fp:
                        file_content = _fp.read()
                    # 截断自动注入的文件内容，防止大文件撑爆 context
                    truncated_file_content = file_content
                    if len(file_content) > max_file_inject_chars:
                        truncated_file_content = file_content[:max_file_inject_chars] + f"\n...(文件内容已截断，原长 {len(file_content)} 字)"
                        logger.debug(
                            "_build_messages: file content truncated for field '%s' "
                            "(path=%s, original %d chars, truncated to %d chars)",
                            field_name, field_value, len(file_content), max_file_inject_chars,
                        )
                    file_content_parts.append(
                        f"[{field_name} 文件内容]\n{truncated_file_content}"
                    )
                    logger.debug(
                        "_build_messages: auto-injected file content for field '%s' "
                        "(path=%s, %d chars)",
                        field_name, field_value, len(truncated_file_content),
                    )
                except Exception as _exc:
                    logger.debug(
                        "_build_messages: skipping auto-inject for field '%s' (path=%s): %s",
                        field_name, field_value, _exc,
                    )
        for part in file_content_parts:
            human_parts.append(part)

        human_content = "\n\n".join(human_parts) if human_parts else "请执行当前步骤。"

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=human_content),
        ]

    # ── 校验逻辑 ──────────────────────────────────────────────────────────────

    def _validate_input(self, step_spec: StepSpec, step_input: Dict[str, Any]) -> None:
        """
        校验步骤输入是否满足 input_spec 的期望字段。

        根据 input_validation_policy 决定处理方式：
          - raise: 抛出 ValueError
          - warn:  打印警告，继续执行
          - ignore: 静默继续
        """
        if not step_spec.input_spec:
            return  # 无期望规格，不校验

        policy = step_spec.input_validation_policy or "warn"
        if policy == "ignore":
            return

        missing = [
            field for field in step_spec.input_spec
            if field not in step_input or step_input[field] is None
        ]

        if not missing:
            return

        if policy == "raise":
            raise ValueError(
                f"Step [{step_spec.id}] input validation failed: "
                f"missing required fields {missing}"
            )
        else:  # warn
            _print_validation_warning(step_spec.id, "输入", missing)
            logger.warning(
                "SkillExecutor: step [%s] input missing fields: %s",
                step_spec.id, missing,
            )

    def _validate_output(self, step_spec: StepSpec, result: Dict[str, Any]) -> None:
        """
        校验步骤输出是否包含 output_spec 期望的字段。

        根据 output_validation_policy 决定处理方式。

        支持两种字段匹配方式：
          1. 直接匹配：output_spec 字段名作为 result dict 的 key（LLM/AGENT 步骤）
          2. 通过 output_mapping 间接匹配：output_spec 字段名是上下文变量名（ctx_key），
             对应的 result_key 通过 output_mapping 查找（SKILL 步骤）
        """
        if not step_spec.output_spec:
            return

        policy = step_spec.output_validation_policy or "warn"
        if policy == "ignore":
            return

        # 反向映射：ctx_key → result_key（用于 SKILL 步骤的间接字段匹配）
        ctx_to_result_key = {
            ctx_k: res_k
            for ctx_k, res_k in step_spec.output_mapping.items()
        }

        missing = []
        for field in step_spec.output_spec:
            # 直接匹配
            if field in result and result[field] is not None:
                continue
            # 通过 output_mapping 间接匹配（output_spec 字段名是 ctx_key）
            mapped_result_key = ctx_to_result_key.get(field)
            if mapped_result_key and result.get(mapped_result_key) is not None:
                continue
            missing.append(field)

        if not missing:
            return

        if policy == "raise":
            raise ValueError(
                f"Step [{step_spec.id}] output validation failed: "
                f"missing expected output fields {missing}"
            )
        else:  # warn
            _print_validation_warning(step_spec.id, "输出", missing)
            logger.warning(
                "SkillExecutor: step [%s] output missing fields: %s",
                step_spec.id, missing,
            )

    # ── 步骤类型路由 ──────────────────────────────────────────────────────────

    def _dispatch(
        self,
        step_spec: StepSpec,
        step_input: Dict[str, Any],
        messages: List[Any],
    ) -> Dict[str, Any]:
        """根据步骤类型路由到对应执行器。"""
        if step_spec.type == StepType.TOOL:
            return self._run_tool_step(step_spec, step_input)
        elif step_spec.type == StepType.LLM:
            return self._run_llm_step(step_spec, messages)
        elif step_spec.type == StepType.AGENT:
            return self._run_agent_step(step_spec, messages)
        elif step_spec.type == StepType.SKILL:
            return self._run_skill_step(step_spec, step_input)
        else:
            raise ValueError(f"Unknown step type: {step_spec.type}")

    def _run_tool_step(
        self, step_spec: StepSpec, step_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        TOOL 步骤：完全绕过 LLM，直接按顺序调用 tools 列表中的工具。

        支持多工具按序调用，参数来自 tool_params 模板（支持 {变量名} 占位符替换）。
        tool_params 为空时回退到直接用 step_input 作为每个工具的参数。

        tool_params 格式（与 tools 列表对应，逐一匹配）：
          [
            {"path": "{module_dir}"},                          # → tools[0] 的参数
            {"path": "{design_path}", "content": "# stub"},   # → tools[1] 的参数
          ]

        每次工具调用的结果会累积，最终以 result=<所有结果汇总> 返回。
        """
        import re as _re

        if not step_spec.tools:
            logger.warning(
                "SkillExecutor: TOOL step [%s] has no tools configured", step_spec.id
            )
            return {}

        from local_agent.core.debug import print_tool_step_call

        all_results: List[str] = []
        failures: List[str] = []

        for tool_idx, tool_name in enumerate(step_spec.tools):
            tool = self._tool_registry.get(tool_name)
            if tool is None:
                logger.warning(
                    "SkillExecutor: TOOL step [%s] tool '%s' not found, skipping",
                    step_spec.id, tool_name,
                )
                all_results.append(f"[Tool '{tool_name}' not found, skipped]")
                failures.append(f"tool '{tool_name}' is not registered")
                continue

            # 确定本次调用的参数
            if step_spec.tool_params and tool_idx < len(step_spec.tool_params):
                # 使用 tool_params 模板，替换 {变量名} 占位符
                raw_params = step_spec.tool_params[tool_idx]
                call_params: Dict[str, Any] = {}
                for param_key, param_val in raw_params.items():
                    if isinstance(param_val, str):
                        # 替换所有 {变量名} 占位符
                        def _replace(m: Any) -> str:
                            var_name = m.group(1)
                            val = step_input.get(var_name)
                            # None 时返回空字符串，避免占位符被字面量写入文件
                            return str(val) if val is not None else ""
                        replaced_val = _re.sub(r'\{(\w+)\}', _replace, param_val)
                        call_params[param_key] = replaced_val
                    else:
                        call_params[param_key] = param_val

                # ── path 参数合法性修复 ──────────────────────────────────────
                # 若 path 参数包含换行符或工具结果格式特征字符串，则该值是误传的上一步
                # 工具输出（如 "[fs_create_dir] [工具结果: ...]"），跳过此次调用以避免
                # 创建出以工具结果文本为名的异常目录/文件。
                _TOOL_RESULT_MARKERS = ("[工具结果:", "[Tool Result:", "状态: success", "状态: error")
                if "path" in call_params:
                    path_val = str(call_params["path"])
                    has_newline = "\n" in path_val or "\r" in path_val
                    has_tool_marker = any(m in path_val for m in _TOOL_RESULT_MARKERS)
                    if has_newline or has_tool_marker:
                        logger.warning(
                            "SkillExecutor: TOOL step [%s] tool '%s' path param looks invalid "
                            "(contains newline or tool result marker), skipping call. "
                            "path prefix: %s",
                            step_spec.id, tool_name, repr(path_val[:80])
                        )
                        all_results.append(
                            f"[{tool_name}] [跳过] path 参数包含非法内容（工具结果或换行符），已忽略此调用"
                        )
                        failures.append(
                            f"tool '{tool_name}' received an invalid path value"
                        )
                        continue

                # ── content 参数 JSON 解包 ────────────────────────────────────
                # 若 content 参数以 '{' 开头，疑似 LLM 输出了 JSON 包装而非纯内容，
                # 尝试解析 JSON 并提取真正的内容字段（如 plan_content / requirements_content
                # / module_design_content / file_content）。
                if "content" in call_params:
                    content_val = call_params["content"]
                    if isinstance(content_val, str) and content_val.lstrip().startswith("{"):
                        try:
                            parsed_content_obj = json.loads(content_val)
                            if isinstance(parsed_content_obj, dict):
                                # 优先按常见内容字段名提取
                                _CONTENT_KEYS = [
                                    "plan_content", "requirements_content",
                                    "module_design_content", "file_content",
                                    "content", "body", "text",
                                ]
                                for _ck in _CONTENT_KEYS:
                                    if _ck in parsed_content_obj and isinstance(parsed_content_obj[_ck], str):
                                        logger.info(
                                            "SkillExecutor: TOOL step [%s] tool '%s' content param "
                                            "was JSON-wrapped; extracted field '%s' (%d chars)",
                                            step_spec.id, tool_name, _ck, len(parsed_content_obj[_ck])
                                        )
                                        call_params["content"] = parsed_content_obj[_ck]
                                        break
                                else:
                                    # 没有已知字段，但只有一个 string 类型的值，直接取它
                                    str_values = [v for v in parsed_content_obj.values() if isinstance(v, str)]
                                    if len(str_values) == 1:
                                        call_params["content"] = str_values[0]
                                        logger.info(
                                            "SkillExecutor: TOOL step [%s] tool '%s' content param "
                                            "was JSON-wrapped; extracted sole string value (%d chars)",
                                            step_spec.id, tool_name, len(str_values[0])
                                        )
                        except (json.JSONDecodeError, ValueError):
                            # 不是合法 JSON，继续使用原始值
                            pass

                    # Weak models occasionally wrap a complete source file in
                    # one Markdown fence despite the prompt. Strip only a
                    # single outer fence; embedded fences remain untouched.
                    content_val = call_params.get("content")
                    path_val = str(call_params.get("path", ""))
                    if (
                        tool_name == "fs_write_file"
                        and isinstance(content_val, str)
                        and path_val.lower().endswith(
                            (".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".java", ".rs")
                        )
                    ):
                        stripped = content_val.strip()
                        if stripped.startswith("```") and stripped.endswith("```"):
                            first_newline = stripped.find("\n")
                            if first_newline >= 0:
                                call_params["content"] = stripped[first_newline + 1:-3].rstrip() + "\n"

                    if self._current_skill_name == "code_file_developer" and step_spec.id == "step_3_write":
                        generated_content = str(call_params.get("content") or "")
                        if not generated_content.strip():
                            failures.append("generated file content is empty")
                            continue
                        if "TODO: 待实现" in generated_content:
                            failures.append("generated file content is still a TODO placeholder")
                            continue
            else:
                # 回退：直接用 step_input 作为参数
                call_params = dict(step_input)

            logger.debug(
                "SkillExecutor: TOOL step [%s] calling tool[%d]='%s' with %s",
                step_spec.id, tool_idx, tool_name, list(call_params.keys()),
            )

            try:
                raw_result = tool._run(**call_params)
            except Exception as exc:
                logger.warning(
                    "SkillExecutor: TOOL step [%s] tool '%s' failed: %s",
                    step_spec.id, tool_name, exc,
                )
                all_results.append(f"[Tool '{tool_name}' error: {exc}]")
                failures.append(f"tool '{tool_name}' raised: {exc}")
                continue

            # 解析工具结果
            parsed = self._parse_and_log_tool_result(tool_name, raw_result)
            output_text = parsed.to_llm_context() if parsed else str(raw_result)
            all_results.append(f"[{tool_name}] {output_text}")
            if parsed is not None and parsed.status == "error":
                failures.append(f"tool '{tool_name}' failed: {parsed.summary}")
            elif str(raw_result).lstrip().lower().startswith("error"):
                failures.append(f"tool '{tool_name}' failed: {raw_result}")

            # ── debug: 打印工具调用输入输出 ──────────────────────────────────
            print_tool_step_call(
                step_id=step_spec.id,
                step_name=step_spec.name,
                tool_name=tool_name,
                tool_input=call_params,
                tool_output=output_text,
            )

        if failures:
            raise RuntimeError(
                f"TOOL step [{step_spec.id}] did not complete: " + "; ".join(failures)
            )

        combined = "\n".join(all_results)
        return {"result": combined}

    def _run_llm_step(
        self, step_spec: StepSpec, messages: List[Any]
    ) -> Dict[str, Any]:
        """
        LLM 步骤：单次 LLM 调用，消息已全新构建（不含历史）。

        若步骤定义了 output_spec，则尝试从 LLM 输出文本中解析出对应字段，
        解析结果与 {"result": content} 合并返回，使后续输出校验可以正确找到期望字段。

        支持的输出格式（按优先级）：
          1. JSON 对象：{"key": "value", ...}
          2. key=value 键值对（每行或逗号分隔）：url_type=static, access_strategy=fetch_url
          3. 多行 key: multiline value 文本块（适合 content/title/status 字段）
          4. 解析失败：仅返回 {"result": content}，校验失败与否取决于 output_validation_policy

        对于 content 字段，若未被解析为独立字段，则将整个 LLM 输出设为 content 的值，
        以保证完整内容不丢失。
        """
        logger.debug(
            "SkillExecutor: LLM step [%s] calling LLM with %d messages",
            step_spec.id, len(messages),
        )
        # ── debug hooks: before_llm ─────────────────────────────────────
        # 在 before_llm hook 的参数里带上 step 上下文，供 debug 输出使用
        before_llm = self._debug_hooks.get("before_llm")
        if before_llm:
            try:
                before_llm(
                    messages=messages,
                    iteration=0,
                    messages_for_llm=messages,
                    step_context=f"[{step_spec.id}] {step_spec.name}",
                )
            except Exception:
                pass
        import time as _time
        _t_start = _time.time()
        response = self._llm.invoke(messages)
        _elapsed = _time.time() - _t_start
        logger.info(
            "[LLM step] [%s] done in %.1fs content_len=%d tool_calls=%d",
            step_spec.id, _elapsed,
            len(response.content) if hasattr(response, "content") and response.content else 0,
            len(response.tool_calls) if hasattr(response, "tool_calls") and response.tool_calls else 0,
        )
        # ── debug hooks: after_llm ──────────────────────────────────────
        after_llm = self._debug_hooks.get("after_llm")
        if after_llm:
            try:
                after_llm(
                    message=response,
                    step_context=f"[{step_spec.id}] {step_spec.name}",
                )
            except Exception:
                pass
        content = response.content if hasattr(response, "content") else str(response)
        result: Dict[str, Any] = {"result": content}

        # 若 output_spec 定义了期望字段，尝试从文本中提取
        if step_spec.output_spec:
            extracted = self._parse_llm_output(content, list(step_spec.output_spec.keys()))
            if extracted:
                result.update(extracted)

            # ── 内容类字段通用 fallback ────────────────────────────────────────────
            # 当字段提取失败时，对内容类字段（plan_content / requirements_content /
            # 任意含 content/plan/requirement/body/text 的字段名，或 output_spec 只有
            # 一个字段）直接用完整 LLM 输出填充，确保后续步骤拿到有效内容而非 None。
            for field in step_spec.output_spec:
                if result.get(field):  # 已经被正确填充，跳过
                    continue
                field_lower = field.lower()
                is_content_like = (
                    "content" in field_lower
                    or "plan" in field_lower
                    or "requirement" in field_lower
                    or "body" in field_lower
                    or "text" in field_lower
                    or len(step_spec.output_spec) == 1  # 只有一个期望字段时无条件 fallback
                )
                if is_content_like and content:
                    result[field] = content
                    logger.debug(
                        "_run_llm_step: step [%s] field '%s' extraction failed, "
                        "falling back to full LLM output (%d chars)",
                        step_spec.id, field, len(content),
                    )

        return result

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        """
        用括号计数法从文本中提取第一个完整的 JSON 对象（支持嵌套 / 含换行）。

        不使用正则，按字符逐一扫描：
          - 遇到 '{' 时 depth+1
          - 遇到 '}' 时 depth-1
          - depth 降到 0 时截取 [start, end] 子串并返回

        字符串字面量内的括号会被跳过（处理 \" 转义）。
        若未找到任何 JSON 对象，返回 None。
        """
        start = text.find('{')
        if start == -1:
            return None
        depth = 0
        in_string = False
        i = start
        while i < len(text):
            ch = text[i]
            if in_string:
                if ch == '\\':
                    i += 2  # 跳过转义字符
                    continue
                if ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
            i += 1
        return None

    @staticmethod
    def _normalize_json_newlines(raw: str) -> str:
        """
        将 JSON 字符串值内的真实换行符替换为合法的 JSON 转义序列。

        LLM 有时返回字符串值含真实 newline 的 JSON，导致 json.loads 报
        "Invalid control character" 错误。此方法把字符串字面量内的：
          \\n  ->  \\\\n
          \\r  ->  \\\\r
        字符串外部（key 名等）的换行不处理；跳过已有的 \\\\ 转义序列。
        """
        result: List[str] = []
        in_string = False
        i = 0
        while i < len(raw):
            ch = raw[i]
            if in_string:
                if ch == '\\':
                    result.append(raw[i:i + 2])
                    i += 2
                    continue
                elif ch == '"':
                    in_string = False
                elif ch == '\n':
                    result.append('\\n')
                    i += 1
                    continue
                elif ch == '\r':
                    result.append('\\r')
                    i += 1
                    continue
            else:
                if ch == '"':
                    in_string = True
            result.append(ch)
            i += 1
        return ''.join(result)

    @staticmethod
    def _parse_llm_output(content: str, expected_keys: List[str]) -> Dict[str, Any]:
        """
        从 LLM 输出文本中解析结构化字段。

        尝试顺序：
          1. 整体 JSON 解析
          2a. 提取 ```json ... ``` 代码块
          2b. 用括号计数法提取第一个完整 JSON 对象（支持嵌套 / 含换行 / 大段内容）
          3. key=value 键值对解析（支持逗号或换行分隔）
          4. "key: value" 冒号格式
          5. 多行文本块解析（key: multiline content）

        只提取 expected_keys 中声明的字段，其余忽略。
        对 content/text/body 类字段，尝试提取长文本（不限定单行）。
        """
        import re

        parsed: Dict[str, Any] = {}

        # ── 尝试 1: 整体 JSON 解析 ────────────────────────────────────────────
        stripped = content.strip()
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                for k in expected_keys:
                    if k in obj:
                        parsed[k] = obj[k]
                if parsed:
                    return parsed
        except (json.JSONDecodeError, ValueError):
            pass

        # ── 尝试 2a: 提取 ```json ... ``` 代码块 ─────────────────────────────
        json_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', content, re.DOTALL)
        if json_block_match:
            try:
                obj = json.loads(json_block_match.group(1))
                if isinstance(obj, dict):
                    for k in expected_keys:
                        if k in obj:
                            parsed[k] = obj[k]
                    if parsed:
                        return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # ── 尝试 2b: 括号计数法提取第一个完整 JSON 对象（支持嵌套/换行/大段内容）──
        raw_json_str = SkillExecutor._extract_first_json_object(content)
        if raw_json_str:
            # 先尝试原始字符串，再尝试 newline 规范化后的版本
            for candidate in (raw_json_str, SkillExecutor._normalize_json_newlines(raw_json_str)):
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        for k in expected_keys:
                            if k in obj:
                                parsed[k] = obj[k]
                        if parsed:
                            return parsed
                except (json.JSONDecodeError, ValueError):
                    pass

        # ── 尝试 3: key=value 键值对解析 ──────────────────────────────────────
        # 例如: "url_type=static, access_strategy=fetch_url"
        #   或: "url_type=static\naccess_strategy=fetch_url"
        # 注意：module_list 的值是 JSON 数组（含方括号、引号等特殊字符），
        #   不能用 [^\s,;\n]+ 截断 — 改用贪婪匹配，提取 "=" 后直到行末的全部内容。
        for k in expected_keys:
            # 首先尝试提取 JSON 数组值（[ ... ] 格式，跨越特殊字符）
            pattern_json = rf'(?:^|[\s,;]){re.escape(k)}\s*=\s*(\[.*?\])'
            m_json = re.search(pattern_json, content, re.IGNORECASE | re.DOTALL)
            if m_json:
                parsed[k] = m_json.group(1).strip()
                continue
            # 再尝试普通 key=value（值到行末）
            pattern = rf'(?:^|[\s,;]){re.escape(k)}\s*=\s*(.+?)(?:\n|$)'
            m = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if m:
                parsed[k] = m.group(1).strip().strip('"\'')

        if parsed:
            return parsed

        # ── 尝试 4: 多行文本块解析（key: multiline content，直到下一个 key 或末尾） ──
        # 适用于 content/title/status 等字段名，其值可能是多行文本
        lines = content.splitlines()
        # 构建 key_pattern → start_line index 的映射
        key_pattern_map: Dict[str, int] = {}
        for i, line in enumerate(lines):
            for k in expected_keys:
                if re.match(rf'^{re.escape(k)}\s*:', line, re.IGNORECASE):
                    key_pattern_map[k] = i
                    break

        if key_pattern_map:
            sorted_keys_by_line = sorted(key_pattern_map.items(), key=lambda x: x[1])
            for idx, (k, start_idx) in enumerate(sorted_keys_by_line):
                # 下一个 key 的起始行（或文件末尾）
                end_idx = sorted_keys_by_line[idx + 1][1] if idx + 1 < len(sorted_keys_by_line) else len(lines)
                # 提取首行的 value 部分
                first_line = lines[start_idx]
                colon_pos = first_line.find(":")
                first_value = first_line[colon_pos + 1:].strip() if colon_pos >= 0 else ""
                # 收集多行
                extra_lines = [l.rstrip() for l in lines[start_idx + 1:end_idx] if l.strip()]
                all_parts = ([first_value] if first_value else []) + extra_lines
                if all_parts:
                    parsed[k] = "\n".join(all_parts)
            if parsed:
                return parsed

        # ── 尝试 5: 单行 "key: value" 冒号格式 (fallback) ────────────────────
        for k in expected_keys:
            pattern = rf'(?:^|[\s,;]){re.escape(k)}\s*:\s*([^\n,;]+)'
            m = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
            if m:
                parsed[k] = m.group(1).strip().strip('"\'')

        return parsed

    def _run_agent_step(
        self, step_spec: StepSpec, messages: List[Any]
    ) -> Dict[str, Any]:
        """
        AGENT 步骤：创建子 ReActEngine，只注入本步骤的 tools + system_prompt。
        初始消息按上述格式全新构建（由 _build_messages 已完成），
        子引擎内部工具调用轮次在本步内累积，步骤结束后只保留最终文本输出。
        """
        from local_agent.core.react import ReActEngine
        from local_agent.core.messages import AIMessage

        # 解析本步骤可用的工具
        step_tools = self._resolve_tools(step_spec.tools)

        # 子引擎的 system_prompt 只用本步骤的（已包含在 messages[0].content 中）
        # 为避免重复，子引擎 system_prompt 置空，完整指令通过 messages 传入
        #
        # tool_call_retry 说明：
        #   对于「只读」类步骤（无 fs_write_file 等写工具），工具调用链中的 retry 引导
        #   只会在弱模型读完文件后不输出结果时再 push 一次，不会造成循环读取。
        #   真正造成 step_2 循环读取的原因是：_build_minimal_messages_for_llm 每轮只传
        #   最后一条 ToolMessage，模型每轮拿到相同的文件内容，不知道自己已经读过了，
        #   于是继续调用相同工具。解决方案是在 AGENT step 子引擎中禁用 sliding window
        #   的「只传最后一条 ToolMessage」逻辑，改为保留所有 ToolMessage（完整历史模式），
        #   这样模型可以看到自己在上一轮已经读了文件、并且产生了输出，从而真正结束。
        #
        #   与此同时，max_tool_retry 设为 1，避免弱模型因首次没有输出格式化结果而被
        #   反复 push 调用工具（它只需要被 push 一次，第二次能输出 module_list= 即可）。
        has_write_tool = any(
            t.name in ("fs_write_file",) for t in step_tools if hasattr(t, "name")
        )
        # 写文件步骤（code generation）：给更多迭代次数和重试机会，关闭滑动窗口以保留完整历史，
        # 这样模型在分块写入时能看到自己已经写了骨架，继续填充实现细节。
        # 只读步骤（无写工具）：关闭滑动窗口同样有益，模型能看到已读取的文件内容，
        # 避免重复读同一文件触发 loop detection。
        sub_engine = ReActEngine(
            llm=self._llm,
            tools=step_tools,
            system_prompt="",       # 不额外注入 system_prompt，已在 messages 中
            max_iterations=15 if has_write_tool else 8,   # 写代码步骤给更多轮次
            tool_call_retry=True,
            max_tool_retry=2 if has_write_tool else 1,    # 写代码步骤允许更多次重试引导
            max_tool_result_length=0,       # 不截断工具结果，全量输出
            enable_tool_result_summarization=False,  # 不对工具结果做 LLM 总结压缩
            message_sliding_window=0,       # 关闭滑动窗口，保留完整历史，防止重复读文件
            debug_hooks=self._debug_hooks,  # 传递调试钩子，确保 skill 内部产生 debug 日志
        )

        logger.debug(
            "SkillExecutor: AGENT step [%s] starting sub-engine with %d tools (streaming mode)",
            step_spec.id, len(step_tools),
        )

        state = {"messages": messages}
        # 使用流式调用代替同步 invoke，避免模型生成大量内容时触发 ReadTimeout。
        # 流式模式下每收到一个 token 就会重置 read timeout 计时器，不会因为
        # 单次响应太长而超时（只要模型在持续输出，read timeout 就不会触发）。
        result_messages: List[Any] = []

        # ── 提前退出检测辅助函数 ──────────────────────────────────────────────
        # 当 step 的 output_spec 中有路径类输出字段（如 design_plan），
        # 且工具历史中关键文件都已写成功时，子引擎可以提前退出，无需等到 max_iterations。
        def _check_early_exit(msgs: List[Any]) -> Optional[str]:
            """
            扫描消息历史中的 ToolMessage，若已写成功所有关键文件，返回合成的 final_text。
            仅用于有 design_plan 类 output_spec 的步骤。
            """
            if not step_spec.output_spec:
                return None
            # 仅在期望输出包含 "design_plan" 时启用（针对 greenfield_developer step_1）
            has_design_plan_spec = any("design_plan" in k or "plan" in k.lower()
                                       for k in step_spec.output_spec)
            if not has_design_plan_spec:
                return None

            import re as _re2
            plan_path: Optional[str] = None
            req_path: Optional[str] = None
            for _m in msgs:
                if not (hasattr(_m, "content") and isinstance(_m.content, str)):
                    continue
                _cl = _m.content.lower()
                if "successfully wrote" not in _cl and "successfully appended" not in _cl:
                    continue
                _pm = _re2.search(r'(?:wrote file|created|appended to file):\s*(\S+)', _m.content)
                if not _pm:
                    continue
                _fp = _pm.group(1)
                if "plan.md" in _fp:
                    plan_path = _fp
                elif "requirements" in _fp.lower():
                    req_path = _fp
            if plan_path and req_path:
                logger.info(
                    "SkillExecutor: AGENT step [%s] early exit: "
                    "both plan.md (%s) and requirements.md (%s) written successfully",
                    step_spec.id, plan_path, req_path,
                )
                return f"design_plan={plan_path}"
            return None

        try:
            for event_type, event_data in sub_engine.stream(state):
                if event_type == "values":
                    # 每次 values 快照即更新 result_messages
                    result_messages = event_data.get("messages", result_messages)
                    # 提前退出检测：两个关键文件都已写成功则无需继续
                    early_exit_text = _check_early_exit(result_messages)
                    if early_exit_text:
                        logger.info(
                            "SkillExecutor: AGENT step [%s] early exit triggered, "
                            "stopping sub-engine stream",
                            step_spec.id,
                        )
                        # 注入一条合成的 AIMessage，让后续 final_text 提取直接拿到
                        from local_agent.core.messages import AIMessage as _AIMsg
                        result_messages = list(result_messages) + [_AIMsg(content=early_exit_text)]
                        break  # 终止 stream 消费
                # messages 事件（chunks/tool messages）不需要处理，仅消费流
        except Exception as _stream_exc:
            # 流式调用失败时，回退到 invoke（保持兼容性）
            logger.warning(
                "SkillExecutor: AGENT step [%s] stream failed (%s), falling back to invoke",
                step_spec.id, _stream_exc,
            )
            result_state = sub_engine.invoke(state)
            result_messages = result_state.get("messages", [])
            # invoke 完成后也做一次提前退出检测（兜底）
            early_exit_text = _check_early_exit(result_messages)
            if early_exit_text and not any(
                isinstance(m, AIMessage) and m.content == early_exit_text
                for m in result_messages
            ):
                result_messages = list(result_messages) + [AIMessage(content=early_exit_text)]

        # 提取子引擎最终的文本输出（最后一条无 tool_calls 的 AIMessage）
        _STUCK_LOOP_PREFIX = "I detected that I'm stuck in a loop"
        final_text = ""
        for msg in reversed(result_messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                candidate = msg.content or ""
                # 跳过 loop detection 自动注入的错误消息和 "encountered error" 类消息
                if candidate and not candidate.startswith("I encountered an error") and not candidate.startswith(_STUCK_LOOP_PREFIX):
                    final_text = candidate
                    break

        # ── 兜底 1：若 final_text 为空（agent 最后一次是工具调用，未输出文字汇报），
        #    尝试从消息历史中找任意非空的 AIMessage content（包括带 error 的，总比空好）
        #    但要跳过 "stuck in a loop" 类的系统自动注入错误消息（这是 loop detection 产生的，
        #    不是真正的任务输出，不能当作有效结果）
        if not final_text:
            for msg in reversed(result_messages):
                if isinstance(msg, AIMessage):
                    candidate = msg.content or ""
                    if candidate and not candidate.startswith(_STUCK_LOOP_PREFIX):
                        final_text = candidate
                        logger.debug(
                            "SkillExecutor: AGENT step [%s] final_text was empty, "
                            "falling back to last AIMessage content",
                            step_spec.id,
                        )
                        break

        # ── 兜底 2：若仍为空，扫描所有 ToolMessage，优先提取 plan.md 路径作为 design_plan，
        #    不能只看最后一条（最后写入的通常是 requirements.md，不是 plan.md）。
        #    典型场景：step_1 写文件后因 max_iterations 退出，未输出路径汇报。
        if not final_text and step_spec.output_spec:
            import re as _re
            # 1. 第一优先级：从全部 ToolMessage 中找 plan.md 的成功写入路径
            plan_path_found = None
            req_path_found = None
            for msg in result_messages:  # 正向遍历，确保找到最后一次写入结果
                if not (hasattr(msg, "content") and isinstance(msg.content, str)):
                    continue
                content_lower = msg.content.lower()
                if "successfully wrote" not in content_lower and "success" not in content_lower:
                    continue
                path_match = _re.search(r'(?:wrote file|created):\s*(\S+)', msg.content)
                if not path_match:
                    continue
                found_path = path_match.group(1)
                if "plan.md" in found_path:
                    plan_path_found = found_path
                elif "requirements" in found_path.lower():
                    req_path_found = found_path

            # 若 plan.md 和 requirements.md 都找到，合成标准格式文字汇报
            if plan_path_found and req_path_found:
                final_text = f"design_plan={plan_path_found}"
                logger.info(
                    "SkillExecutor: AGENT step [%s] synthesized design_plan='%s' "
                    "from ToolMessages (both plan.md and requirements.md written successfully)",
                    step_spec.id, plan_path_found,
                )
            elif plan_path_found:
                # 只找到 plan.md 也足够——requirements.md 路径可能在其他消息里
                final_text = f"design_plan={plan_path_found}"
                logger.info(
                    "SkillExecutor: AGENT step [%s] extracted plan.md path '%s' "
                    "from ToolMessages as final_text fallback",
                    step_spec.id, plan_path_found,
                )
            elif req_path_found:
                # 只有 requirements.md 路径——不能直接用，记录警告后继续走兜底 3
                logger.warning(
                    "SkillExecutor: AGENT step [%s] found only requirements path '%s', "
                    "plan.md not found in ToolMessages; will try rescue LLM",
                    step_spec.id, req_path_found,
                )

        # ── 兜底 2.5：若工具历史中存在至少一次成功写入但 plan.md/requirements.md 路径均未提取到，
        #    尝试从 output_spec 字段名推断哪些字段可以从写入记录中直接填充。
        #    （此段为通用兜底，与 plan.md 特定逻辑无关，防止未来其他步骤出现同类问题）
        if not final_text and step_spec.output_spec:
            import re as _re
            # 收集所有成功写入的文件路径
            written_paths = []
            for msg in result_messages:
                if not (hasattr(msg, "content") and isinstance(msg.content, str)):
                    continue
                if "successfully wrote" not in msg.content.lower():
                    continue
                m = _re.search(r'(?:wrote file|created):\s*(\S+)', msg.content)
                if m:
                    written_paths.append(m.group(1))

            if written_paths:
                # 对每个 output_spec 字段，尝试找名字最匹配的路径
                for field in step_spec.output_spec:
                    # 若 final_text 已经设置了该字段的内容则跳过
                    if final_text and f"{field}=" in final_text:
                        continue
                    field_lower = field.lower()
                    # 简单启发式：字段名含 "plan" → 找含 plan.md 的路径
                    for p in written_paths:
                        p_lower = p.lower()
                        if ("plan" in field_lower and "plan" in p_lower) or \
                           ("design" in field_lower and "plan" in p_lower):
                            final_text = final_text or f"{field}={p}"
                            logger.info(
                                "SkillExecutor: AGENT step [%s] heuristic: "
                                "mapped field '%s' → path '%s'",
                                step_spec.id, field, p,
                            )
                            break

        # ── 兜底 3（救援 LLM）：若 final_text 仍为空，且本步骤有 output_spec，且有 ToolMessage 内容，
        #    说明子引擎工具调用成功（文件已读），但模型因循环检测被强制退出而未产生文本输出。
        #    此时用一次简单的 LLM 调用，将文件内容 + 期望输出格式一起喂给模型，让其直接输出结构化结果。
        #    典型场景：step_2 读取 plan.md 后因循环被中断，module_list 始终为空。
        if not final_text and step_spec.output_spec:
            last_tool_content = ""
            from local_agent.core.messages import ToolMessage as _TM
            for msg in reversed(result_messages):
                if isinstance(msg, _TM) and isinstance(msg.content, str) and msg.content:
                    last_tool_content = msg.content
                    break
            if last_tool_content:
                logger.info(
                    "SkillExecutor: AGENT step [%s] triggering rescue LLM call "
                    "(sub-engine exited without text output, last ToolMessage has content)",
                    step_spec.id,
                )
                try:
                    rescue_text = self._rescue_llm_call(step_spec, messages, last_tool_content)
                    if rescue_text:
                        final_text = rescue_text
                        logger.info(
                            "SkillExecutor: AGENT step [%s] rescue LLM call succeeded, "
                            "final_text has %d chars",
                            step_spec.id, len(final_text),
                        )
                except Exception as exc:
                    logger.warning(
                        "SkillExecutor: AGENT step [%s] rescue LLM call failed: %s",
                        step_spec.id, exc,
                    )

        result: Dict[str, Any] = {"result": final_text}

        # 若步骤定义了 output_spec，尝试从最终文本中解析结构化字段
        # （与 _run_llm_step 保持一致，使输出校验能正确找到期望字段）
        if step_spec.output_spec:
            if final_text:
                extracted = self._parse_llm_output(final_text, list(step_spec.output_spec.keys()))
                if extracted:
                    result.update(extracted)
            # 对于 output_spec 中任何尚未被设置的字段，全部用 final_text 兜底，
            # 确保校验阶段能找到字段（哪怕内容只是未解析的原始文本）。
            for field in step_spec.output_spec:
                if not result.get(field) and final_text:
                    result[field] = final_text

        return result

    def _rescue_llm_call(
        self,
        step_spec: StepSpec,
        original_messages: List[Any],
        last_tool_content: str,
    ) -> str:
        """
        救援 LLM 调用：当 AGENT 步骤因循环检测退出而未产生文本输出时，
        用一次简单的 LLM 调用从已读取的工具结果（如文件内容）中直接提取结构化输出。

        使用与本步骤相同的 SystemMessage（含 step_prompt + output_spec 说明），
        将最后一条 ToolMessage 内容作为 HumanMessage 喂给 LLM，让其直接输出结果。
        """
        from local_agent.core.messages import HumanMessage, SystemMessage

        # 提取原始 SystemMessage（含 step_prompt）
        sys_msg = None
        for m in original_messages:
            if isinstance(m, SystemMessage):
                sys_msg = m
                break

        if sys_msg is None:
            return ""

        # 构建期望输出格式提示
        output_keys = list(step_spec.output_spec.keys()) if step_spec.output_spec else []
        output_hint = ""
        if output_keys:
            key_example = output_keys[0]
            output_hint = (
                f"\n\n【重要】你已经读取了上方的文件内容。"
                f"现在请直接输出结构化结果，格式如下（以 {key_example}= 开头）：\n"
                f"{key_example}=<提取的值>"
            )

        rescue_human = HumanMessage(content=(
            f"[已读取的文件内容]\n{last_tool_content}"
            f"{output_hint}"
        ))

        rescue_messages = [sys_msg, rescue_human]

        logger.debug(
            "SkillExecutor: _rescue_llm_call for step [%s], "
            "tool_content_len=%d, output_keys=%s",
            step_spec.id, len(last_tool_content), output_keys,
        )

        # ── debug hook: before_llm ────────────────────────────────────────────
        before_llm = self._debug_hooks.get("before_llm")
        if before_llm:
            try:
                before_llm(
                    messages=rescue_messages,
                    messages_for_llm=rescue_messages,
                    step_context=f"[{step_spec.id}] {step_spec.name} (rescue)",
                )
            except Exception:
                pass

        response = self._llm.invoke(rescue_messages)
        content = response.content if hasattr(response, "content") else str(response)

        # ── debug hook: after_llm ─────────────────────────────────────────────
        after_llm = self._debug_hooks.get("after_llm")
        if after_llm:
            try:
                after_llm(
                    message=response,
                    step_context=f"[{step_spec.id}] {step_spec.name} (rescue)",
                )
            except Exception:
                pass

        return content or ""

    def _run_skill_step(
        self, step_spec: StepSpec, step_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        SKILL 步骤：通过 SkillTool 调用嵌套 skill。

        支持两种模式：
        1. step_input 包含 url 字段（单 URL）→ 直接以结构化字典调用子 skill
        2. step_input 包含 filtered_urls 字段（URL 列表）→ 循环逐个以 url 调用，汇总结果
        3. 其他情况 → task 字符串 fallback

        调用前后通过 PromptContextRegistry（invocation_id）记录嵌套上下文，
        便于 debug print 模式中追踪 skill 调用链路。
        """
        if not step_spec.nested_skill:
            raise ValueError(
                f"SKILL step [{step_spec.id}] has no nested_skill configured"
            )

        def with_mapped_outputs(result: Dict[str, Any]) -> Dict[str, Any]:
            """Bridge the nested skill's standard ``result`` to step outputs."""
            payload = result.get("result", "")
            for result_key in step_spec.output_mapping.values():
                result.setdefault(result_key, payload)
            return result

        from local_agent.skills.skill_tool import SkillTool
        from local_agent.core.prompt_registry import (
            generate_invocation_id,
            save_prompt_context,
            retrieve_prompt_context,
        )
        from local_agent.core.debug import (
            print_prompt_context_save,
            print_prompt_context_restore,
            print_skill_invocation_input,
            print_skill_invocation_output,
        )

        # ── debug: 打印 skill 调用输入 ─────────────────────────────────────────
        parent_skill_name = getattr(self, "_current_skill_name", "unknown")
        print_skill_invocation_input(
            parent_skill=parent_skill_name,
            step_id=step_spec.id,
            nested_skill=step_spec.nested_skill,
            step_input=step_input,
        )

        # ── 生成调用 ID，存入注册表 ────────────────────────────────────────────
        invocation_id = generate_invocation_id()
        task_desc = str(step_input)
        # SKILL 步骤不维护完整的 messages 列表，存入空列表作为占位
        save_prompt_context(
            invocation_id=invocation_id,
            messages=[],
            skill_name=step_spec.nested_skill,
            task=task_desc,
        )
        print_prompt_context_save(
            invocation_id=invocation_id,
            skill_name=step_spec.nested_skill,
            task=task_desc,
            messages_count=0,  # SkillExecutor 步骤无 LLM messages 列表
        )
        logger.debug(
            "_run_skill_step: saved prompt context for nested skill '%s' "
            "(invocation_id=%s, step_id=%s)",
            step_spec.nested_skill, invocation_id, step_spec.id,
        )

        query = step_input.get("query", "")

        try:
            # ── 从注册表直接获取子 skill（绕过 SkillTool 的访问控制） ────────────
            from local_agent.skills.registry import SkillRegistry
            from local_agent.tools.registry import ToolRegistry

            registry = SkillRegistry()
            nested_skill = registry.get(step_spec.nested_skill)
            nested_parsed_config = getattr(nested_skill, "parsed_config", None) if nested_skill else None

            # Deterministic file batching for module_developer. The design
            # contract uses one ``### module/path.ext`` section per file.
            # Parse those sections in the framework and invoke the file skill
            # exactly once per declared file, in document order.
            module_design_content = step_input.get("module_design_content")
            if module_design_content is not None and step_spec.nested_skill == "code_file_developer":
                if nested_parsed_config is None:
                    raise ValueError("code_file_developer has no parsed config")
                design_text = str(module_design_content)
                file_specs: List[Tuple[str, str]] = []
                # Only scan the declared file-list section. This supports
                # headings, numbered/bulleted lists, bold/backtick paths, and
                # a filename heading followed by a separate 路径 field.
                file_list_match = re.search(
                    r"^##\s*2[.、]?\s*文件清单\s*$([\s\S]*?)(?=^##\s*3[.、]?|\Z)",
                    design_text,
                    re.MULTILINE,
                )
                if file_list_match:
                    file_list_text = file_list_match.group(1)
                else:
                    file_list_text = ""

                module_match = re.search(r"design/([A-Za-z0-9_.-]+)_design\.md", design_text)
                module_name = module_match.group(1) if module_match else ""
                extension_pattern = (
                    r"(?:py|pyi|js|jsx|ts|tsx|go|java|rs|md|json|ya?ml|toml|ini|cfg|txt|env)"
                )
                candidate_pattern = re.compile(
                    rf"(?<![A-Za-z0-9_.-])(?P<path>"
                    rf"(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.{extension_pattern})"
                )
                candidates: List[Tuple[str, int]] = []
                seen_paths: set[str] = set()
                for match in candidate_pattern.finditer(file_list_text):
                    file_path = match.group("path")
                    if "/" not in file_path:
                        if not module_name:
                            continue
                        file_path = f"{module_name}/{file_path}"
                    if module_name and not file_path.startswith(f"{module_name}/"):
                        continue
                    if file_path in seen_paths:
                        continue
                    seen_paths.add(file_path)
                    candidates.append((file_path, match.start()))

                for index, (file_path, start) in enumerate(candidates):
                    end = candidates[index + 1][1] if index + 1 < len(candidates) else len(file_list_text)
                    # Include the list marker/heading immediately before the
                    # path while excluding other files' sections.
                    section_start = max(
                        file_list_text.rfind("\n", 0, start),
                        file_list_text.rfind("###", 0, start),
                    )
                    section = file_list_text[max(0, section_start):end].strip()
                    file_specs.append((file_path, section))

                if not file_specs:
                    raise ValueError(
                        "module design contains no parseable paths in its '## 2. 文件清单' section"
                    )

                tool_registry = ToolRegistry()
                completed_files: List[str] = []
                failed_files: List[str] = []
                for file_path, section in file_specs:
                    file_task = f"文件路径：{file_path}\n{section}"
                    try:
                        sub_executor = SkillExecutor(
                            tool_registry=tool_registry,
                            llm=self._llm,
                            debug_hooks=self._debug_hooks,
                        )
                        sub_result = sub_executor.execute(
                            parsed_config=nested_parsed_config,
                            initial_input={"task": file_task},
                            global_context=file_task,
                        )
                        final_text = SkillTool._extract_final_output(  # type: ignore[arg-type]
                            step_spec.nested_skill, sub_result, nested_parsed_config
                        )
                        if not final_text or "[Skill Error]" in final_text:
                            raise RuntimeError(final_text or "nested skill returned no result")
                        completed_files.append(file_path)
                    except Exception as exc:  # continue remaining files, fail batch afterwards
                        logger.exception(
                            "_run_skill_step: file '%s' failed in batch", file_path
                        )
                        failed_files.append(f"{file_path}: {exc}")

                if failed_files:
                    raise RuntimeError(
                        "File batch incomplete "
                        f"({len(completed_files)}/{len(file_specs)} succeeded): "
                        + "; ".join(failed_files)
                    )

                result = {"result": "\n".join(f"{path}: success" for path in completed_files)}
                print_skill_invocation_output(
                    parent_skill=parent_skill_name,
                    step_id=step_spec.id,
                    nested_skill=step_spec.nested_skill,
                    result=result,
                )
                return with_mapped_outputs(result)

            # Deterministic module batching. Module orchestration must not rely
            # on an LLM remembering to emit one invoke_skill call per item.
            module_list_raw = step_input.get("module_list")
            if module_list_raw is not None:
                if nested_parsed_config is None:
                    raise ValueError(
                        f"Nested skill '{step_spec.nested_skill}' has no parsed config"
                    )

                if isinstance(module_list_raw, list):
                    modules = module_list_raw
                elif isinstance(module_list_raw, str):
                    try:
                        modules = json.loads(module_list_raw.strip())
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"module_list is not valid JSON: {exc}") from exc
                else:
                    raise ValueError("module_list must be a JSON array or list")

                if not isinstance(modules, list) or not modules:
                    raise ValueError("module_list must contain at least one module")

                tool_registry = ToolRegistry()
                completed: List[str] = []
                failed: List[str] = []
                result_lines: List[str] = []

                for index, module in enumerate(modules):
                    if not isinstance(module, dict):
                        failed.append(f"item {index + 1}: expected object")
                        continue
                    module_name = str(module.get("module_name") or "").strip()
                    if not module_name:
                        failed.append(f"item {index + 1}: missing module_name")
                        continue

                    module_task = (
                        f"模块名：{module_name}\n"
                        f"编程语言：{module.get('language', '')}\n"
                        f"职责：{module.get('description', '')}\n"
                        f"对外接口：{module.get('interfaces', '')}\n"
                        f"依赖模块：{module.get('dependencies', '无')}\n"
                        f"技术选型：{module.get('tech_stack', '')}\n"
                        f"设计文档路径：design/{module_name}_design.md"
                    )
                    if step_spec.batch_task_suffix:
                        module_task += "\n" + step_spec.batch_task_suffix.strip()

                    try:
                        sub_executor = SkillExecutor(
                            tool_registry=tool_registry,
                            llm=self._llm,
                            debug_hooks=self._debug_hooks,
                        )
                        sub_result = sub_executor.execute(
                            parsed_config=nested_parsed_config,
                            initial_input={"task": module_task},
                            global_context=module_task,
                        )
                        final_text = SkillTool._extract_final_output(  # type: ignore[arg-type]
                            step_spec.nested_skill, sub_result, nested_parsed_config
                        )
                        if not final_text or "[Skill Error]" in final_text:
                            raise RuntimeError(final_text or "nested skill returned no result")
                        completed.append(module_name)
                        result_lines.append(f"{module_name}: success")
                    except Exception as exc:  # continue remaining modules, fail batch afterwards
                        logger.exception(
                            "_run_skill_step: module '%s' failed in batch", module_name
                        )
                        failed.append(f"{module_name}: {exc}")
                        result_lines.append(f"{module_name}: failed")

                if failed:
                    raise RuntimeError(
                        "Module batch incomplete "
                        f"({len(completed)}/{len(modules)} succeeded): " + "; ".join(failed)
                    )

                result = {"result": "\n".join(result_lines)}
                print_skill_invocation_output(
                    parent_skill=parent_skill_name,
                    step_id=step_spec.id,
                    nested_skill=step_spec.nested_skill,
                    result=result,
                )
                return with_mapped_outputs(result)

            # ── 模式 2：filtered_urls 列表 → 逐个调用 ────────────────────────────
            filtered_urls_raw = step_input.get("filtered_urls")
            if filtered_urls_raw is not None:
                # 解析 URL 列表：可能是 JSON 字符串、列表或逗号分隔字符串
                urls: List[str] = []
                if isinstance(filtered_urls_raw, list):
                    urls = [str(u) for u in filtered_urls_raw if u]
                elif isinstance(filtered_urls_raw, str):
                    stripped = filtered_urls_raw.strip()
                    if stripped.startswith("["):
                        try:
                            urls = json.loads(stripped)
                        except Exception:
                            # fallback：按换行或逗号分割
                            urls = [u.strip().strip('"\'') for u in stripped.strip("[]").replace("\n", ",").split(",") if u.strip().strip('"\'')]
                    else:
                        urls = [u.strip() for u in stripped.splitlines() if u.strip()]
                        if not urls:
                            urls = [u.strip() for u in stripped.split(",") if u.strip()]

                if urls:
                    page_contents_parts: List[str] = []
                    nested_skill_name: str = step_spec.nested_skill  # already checked above

                    skill_tool = SkillTool(llm=self._llm, debug_hooks=self._debug_hooks)

                    def _fetch_url(url: str) -> Tuple[str, str]:
                        child_input: Dict[str, Any] = {"url": url}
                        if query:
                            child_input["query"] = str(query)
                        part = skill_tool._run_structured(  # type: ignore[protected-access]
                            skill_name=nested_skill_name,
                            structured_input=child_input,
                        )
                        return url, part

                    max_workers = min(len(urls), 5)
                    results_map: Dict[str, str] = {}
                    with ThreadPoolExecutor(max_workers=max_workers) as executor_pool:
                        future_map = {executor_pool.submit(_fetch_url, url): url for url in urls}
                        for future in as_completed(future_map):
                            try:
                                fetched_url, part = future.result()
                                results_map[fetched_url] = part
                            except Exception as exc:  # pylint: disable=broad-except
                                failed_url = future_map[future]
                                logger.warning(
                                    "_run_skill_step: parallel fetch failed for url '%s': %s",
                                    failed_url, exc,
                                )
                                results_map[failed_url] = f"[Skill Error] Failed to fetch: {exc}"

                    # 按原始顺序拼接结果
                    for url in urls:
                        if url in results_map:
                            page_contents_parts.append(f"=== URL: {url} ===\n{results_map[url]}")

                    result = {"result": "\n\n".join(page_contents_parts)}
                    # ── debug: 打印 skill 调用输出（URL 列表模式） ──────────────
                    print_skill_invocation_output(
                        parent_skill=parent_skill_name,
                        step_id=step_spec.id,
                        nested_skill=step_spec.nested_skill,
                        result=result,
                    )
                    return with_mapped_outputs(result)

            # ── 若获取到 ParsedSkillConfig，直接用 SkillExecutor 执行（绕过 SkillTool 访问控制）──
            if nested_parsed_config is not None:
                tool_registry = ToolRegistry()
                sub_executor = SkillExecutor(
                    tool_registry=tool_registry,
                    llm=self._llm,
                    debug_hooks=self._debug_hooks,  # 传递调试钩子
                )
                global_ctx = _format_value(step_input)

                logger.info(
                    "_run_skill_step: directly executing nested skill '%s' via SkillExecutor (%d steps)",
                    step_spec.nested_skill, len(nested_parsed_config.steps),
                )

                # ── 模式 1：step_input 包含 url → 结构化输入 ──────────────────────
                if "url" in step_input and step_input["url"]:
                    sub_result = sub_executor.execute(
                        parsed_config=nested_parsed_config,
                        initial_input=step_input,
                        global_context=global_ctx,
                    )
                else:
                    # ── 模式 3：通用 task 路径 ──────────────────────────────────────
                    task_str = step_input.get("task") or global_ctx
                    sub_result = sub_executor.execute(
                        parsed_config=nested_parsed_config,
                        initial_input={"task": task_str, "query": task_str, **step_input},
                        global_context=task_str,
                    )

                # 提取最终输出文本（SkillTool 已在函数头部导入）
                final_text = SkillTool._extract_final_output(  # type: ignore[arg-type]
                    step_spec.nested_skill, sub_result, nested_parsed_config
                )
                skill_result = {"result": final_text or str(sub_result)}
                # ── debug: 打印 skill 调用输出（ParsedSkillConfig 路径） ─────────
                print_skill_invocation_output(
                    parent_skill=parent_skill_name,
                    step_id=step_spec.id,
                    nested_skill=step_spec.nested_skill,
                    result=skill_result,
                )
                return with_mapped_outputs(skill_result)

            # ── Fallback：子 skill 未在注册表中，走 SkillTool（非 sub_skill 时） ────
            skill_tool = SkillTool(llm=self._llm, debug_hooks=self._debug_hooks)

            # ── 模式 1：step_input 包含 url → 单次结构化调用 ──────────────────────
            if "url" in step_input and step_input["url"]:
                raw_result = skill_tool._run_structured(
                    skill_name=step_spec.nested_skill,
                    structured_input=step_input,
                )
                fallback_result = {"result": raw_result}
                print_skill_invocation_output(
                    parent_skill=parent_skill_name,
                    step_id=step_spec.id,
                    nested_skill=step_spec.nested_skill,
                    result=fallback_result,
                )
                return with_mapped_outputs(fallback_result)

            # ── 模式 3：fallback → task 字符串 ────────────────────────────────────
            task = _format_value(step_input)
            raw_result = skill_tool._run(
                skill_name=step_spec.nested_skill,
                task=task,
            )
            fallback_result2 = {"result": raw_result}
            print_skill_invocation_output(
                parent_skill=parent_skill_name,
                step_id=step_spec.id,
                nested_skill=step_spec.nested_skill,
                result=fallback_result2,
            )
            return with_mapped_outputs(fallback_result2)

        finally:
            # ── 调用完成（无论成功/异常）：从注册表取回并清除条目 ──────────────
            entry = retrieve_prompt_context(invocation_id)
            if entry:
                result_repr = ""
                print_prompt_context_restore(
                    invocation_id=invocation_id,
                    skill_name=entry.skill_name,
                    result_len=len(result_repr),
                )
                logger.debug(
                    "_run_skill_step: retrieved prompt context for '%s' "
                    "(invocation_id=%s)",
                    entry.skill_name, invocation_id,
                )

    # ── 工具结果解析 ──────────────────────────────────────────────────────────

    def _parse_and_log_tool_result(self, tool_name: str, raw_result: Any) -> Any:
        """
        解析工具结果并打印到终端（用于 TOOL 类型步骤）。
        
        Returns:
            ParsedToolResult 或 None（解析失败时）
        """
        try:
            from local_agent.core.tool_result_parser import ToolResultParser
            parser = ToolResultParser()
            raw_str = raw_result if isinstance(raw_result, str) else str(raw_result)
            parsed = parser.parse(tool_name, raw_str)
            # 打印解析结果到终端
            print(parsed.format_for_display())
            return parsed
        except Exception as exc:
            logger.debug("SkillExecutor: tool result parsing failed: %s", exc)
            return None

    # ── 错误处理 ──────────────────────────────────────────────────────────────

    def _handle_failure(
        self,
        step_spec: StepSpec,
        exc: Exception,
        ctx: ExecutionContext,
    ) -> Optional[Dict[str, Any]]:
        """
        根据 on_failure 策略处理步骤失败。

        Returns:
            {} 表示跳过/继续，None 不会出现（raise 时直接抛出）
        """
        strategy = step_spec.on_failure or "raise"
        logger.error(
            "SkillExecutor: step [%s] failed: %s | strategy=%s",
            step_spec.id, exc, strategy,
        )

        if strategy == "raise":
            _print_step_fatal(step_spec.id, exc)
            raise exc
        elif strategy == "skip":
            # skip: log the failure but do NOT propagate — return empty dict so
            # the skill continues with the next step.
            _print_step_fatal(step_spec.id, exc)
            return {}
        elif strategy.startswith("retry:"):
            # retry 在 BaseStep.run 中处理，这里作为 fallback
            _print_step_fatal(step_spec.id, exc)
            raise exc
        elif strategy.startswith("fallback:"):
            target = strategy.split(":", 1)[1]
            logger.warning(
                "SkillExecutor: fallback to step '%s' not yet supported; treating as fatal [%s]",
                target, step_spec.id,
            )
            _print_step_fatal(step_spec.id, exc)
            raise exc
        else:
            _print_step_fatal(step_spec.id, exc)
            raise exc

    # ── 工具辅助 ──────────────────────────────────────────────────────────────

    def _resolve_step_input(
        self, step_spec: StepSpec, ctx: ExecutionContext
    ) -> Dict[str, Any]:
        """从上下文按 input_mapping 提取本步骤所需变量。"""
        return {
            local_k: ctx.get(ctx_k)
            for local_k, ctx_k in step_spec.input_mapping.items()
        }

    def _resolve_tools(self, tool_names: List[str]) -> List[Any]:
        """将工具名列表解析为工具对象列表。

        对于 'invoke_skill' 特殊处理：SkillTool 需要 LLM 实例，不在普通 ToolRegistry 中，
        此处按需实例化并注入。
        """
        tools = []
        invoke_skill_requested = False
        for name in tool_names:
            if name == "invoke_skill":
                invoke_skill_requested = True
                continue
            t = self._tool_registry.get(name)
            if t is not None:
                tools.append(t)
            else:
                logger.warning(
                    "SkillExecutor: tool '%s' not found in registry, skipping", name
                )

        # ── 注入 invoke_skill（SkillTool）──────────────────────────────────
        # SkillTool 不在 ToolRegistry 中，需单独实例化（需要 LLM 实例）
        if invoke_skill_requested:
            # 检查是否已有 invoke_skill（避免重复注入）
            if not any(getattr(t, "name", None) == "invoke_skill" for t in tools):
                try:
                    from local_agent.skills.skill_tool import SkillTool
                    skill_tool = SkillTool(
                        llm=self._llm,
                        debug_hooks=self._debug_hooks,
                        allow_sub_skills=True,  # SkillExecutor 内部注入，允许调用 sub_skill
                        blocked_skills=[self._current_skill_name] if self._current_skill_name not in ("unknown", "") else None,  # 防止 skill 递归调用自身
                        current_skill_name=self._current_skill_name,  # 传入父 skill 名称，用于 allowed_parent_skills 检查
                    )
                    tools.append(skill_tool)
                    logger.debug(
                        "SkillExecutor: injected invoke_skill tool for step"
                    )
                except Exception as exc:
                    logger.warning(
                        "SkillExecutor: failed to inject invoke_skill tool: %s", exc
                    )

        return tools


# ── 格式化工具 ────────────────────────────────────────────────────────────────


def _format_value(value: Any, max_len: int = 0) -> str:
    """
    将任意值格式化为可读字符串，用于 LLM 消息正文。
    max_len > 0 时截断并添加提示；默认 0 表示不截断，全量输出。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            text = str(value)
    else:
        text = str(value)

    if max_len > 0 and len(text) > max_len:
        text = text[:max_len] + f"\n...[内容过长，已截断至 {max_len} 字符]"
    return text
