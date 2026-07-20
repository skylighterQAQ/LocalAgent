"""
local_agent.core.planner
=========================
TaskPlanner – 任务规划器，在执行用户请求前先进行结构化规划。

规划流程：
  1. 接收用户请求 + 可用 skill 列表
  2. 调用 LLM 生成 TaskPlan（步骤列表，每步是否需要 skill）
  3. TaskPlan 作为后续每步执行的 prompt 上下文

TaskPlan 格式（与 skill.json 对齐）：
  - 包含若干 TaskStep（步骤 id、标题、描述、类型、是否激活 skill、可用工具、输入/输出规格等）
  - 整体策略说明
  - 整体可用工具列表
  - 整体输入/输出规格

使用示例::

    planner = TaskPlanner(llm)
    plan = planner.plan(
        user_request="搜索最新的 Python 异步教程并保存到文件",
        available_skills=["web_researcher", "file_manager"],
    )
    # plan.steps[0].skill == "web_researcher"
    # plan.steps[1].skill == "file_manager"
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── 数据模型 ─────────────────────────────────────────────────────────────────

class TaskStep(BaseModel):
    """
    单个规划步骤，格式与 skill.json 中的 steps 对齐。
    """
    step_id: str = Field(description="步骤唯一标识，如 'step_1'")
    title: str = Field(description="步骤标题（人读，用于展示和日志）")
    description: str = Field(description="本步骤具体要做什么")
    type: str = Field(
        default="agent",
        description="步骤类型：'llm'（纯推理）/ 'agent'（ReAct循环+工具）/ 'skill'（调用子skill）",
    )
    skill: Optional[str] = Field(
        default=None,
        description="激活的 skill 名称，None 表示直接使用 ReAct 循环",
    )
    tools: List[str] = Field(
        default_factory=list,
        description="本步骤可使用的工具名称列表（对应 skill.json steps[].tools）",
    )
    nested_skill: Optional[str] = Field(
        default=None,
        description="嵌套调用的 sub-skill 名称（type='skill' 时使用）",
    )
    input_spec: Dict[str, str] = Field(
        default_factory=dict,
        description="本步骤输入规格 {字段名: 类型描述}",
    )
    output_spec: Dict[str, str] = Field(
        default_factory=dict,
        description="本步骤输出规格 {字段名: 类型描述}",
    )
    reason: str = Field(
        default="",
        description="选择或不选择 skill 的原因（LLM 推理依据）",
    )


class TaskPlan(BaseModel):
    """
    完整任务规划结果，格式与 skill.json 顶层结构对齐。
    """
    user_request: str = Field(description="用户原始请求")
    overall_strategy: str = Field(
        default="",
        description="整体执行策略的一句话说明",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="整个任务规划涉及的所有工具列表（对应 skill.json available_tools）",
    )
    input_spec: Dict[str, str] = Field(
        default_factory=dict,
        description="任务整体输入规格",
    )
    output_spec: Dict[str, str] = Field(
        default_factory=dict,
        description="任务整体输出规格",
    )
    steps: List[TaskStep] = Field(
        default_factory=list,
        description="有序步骤列表",
    )

    def to_summary(self) -> str:
        """生成给 LLM 注入的规划摘要字符串。"""
        lines = [
            f"[任务规划] 共 {len(self.steps)} 步",
            f"策略: {self.overall_strategy}",
            "",
        ]
        for step in self.steps:
            skill_info = f" (skill: {step.skill})" if step.skill else " (直接执行)"
            type_info = f" [type:{step.type}]"
            tools_info = f" tools:{step.tools}" if step.tools else ""
            lines.append(f"  {step.step_id}. {step.title}{skill_info}{type_info}{tools_info}")
            lines.append(f"     {step.description}")
        return "\n".join(lines)

    def get_step(self, step_id: str) -> Optional[TaskStep]:
        """按 id 查找步骤。"""
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None


class StepResult(BaseModel):
    """单步执行结果，传递给下一步骤。"""
    step_id: str
    title: str
    status: str = "success"  # "success" | "error" | "skipped"
    output: str = Field(default="", description="本步骤的文本输出（给下步 LLM 用）")
    error: Optional[str] = None


# ─── 规划 Prompt ──────────────────────────────────────────────────────────────

_PLANNER_SYSTEM_PROMPT = """\
你是一个任务规划器。你的职责是：分析用户请求，将其拆解为有序执行步骤，为每步判断是否需要激活专用 skill，并指定每步可用工具和输入输出规格。
只输出纯 JSON，不要任何 Markdown 代码块，不要解释文字。
输出格式必须与 skill.json 的结构完全一致。
"""

_PLANNER_USER_PROMPT_TEMPLATE = """\
## 用户请求
{user_request}

## 可用 Skill 列表
{skill_list}

## 可用工具列表（参考）
常见工具：search_web, fetch_url, fs_read_file, fs_write_file, fs_list_dir, shell_run,
code_execute_python, code_execute_shell, browser_get_text, memory_save 等。

## 输出要求
请分析用户请求，将其拆解为 1-5 个有序执行步骤，输出以下 JSON 格式（与 skill.json 完全一致）：

{{
  "overall_strategy": "整体执行策略的一句话说明",
  "available_tools": ["所有步骤涉及的工具名称列表"],
  "input_spec": {{
    "query": "string - 用户原始请求"
  }},
  "output_spec": {{
    "result": "string - 最终输出结果"
  }},
  "steps": [
    {{
      "step_id": "step_1",
      "title": "步骤标题（简短，方便展示）",
      "description": "本步骤具体做什么（50字以内）",
      "type": "agent",
      "skill": "skill名称 或 null（直接执行不需要 skill）",
      "tools": ["本步骤可用的工具名称列表"],
      "nested_skill": null,
      "input_spec": {{"field_name": "type - 描述"}},
      "output_spec": {{"field_name": "type - 描述"}},
      "reason": "为什么选/不选 skill（简短说明）"
    }}
  ]
}}

## 规划原则
1. 步骤要具体可执行，不要泛泛而谈
2. skill 只在明确需要其能力时才激活（如需要搜索网页用 web_researcher，需要代码开发用 code_developer）
3. 简单问答、计算等不需要激活任何 skill（skill 为 null）
4. 步骤数量适中：简单任务 1-2 步，复杂任务 3-5 步，不要过度拆分
5. 只使用可用 Skill 列表中存在的 skill 名称
6. type 字段含义：
   - "llm": 纯 LLM 推理，不调用工具
   - "agent": ReAct 循环，可调用 tools 中指定的工具
   - "skill": 调用嵌套 sub-skill（nested_skill 字段指定）
7. tools 字段：列出本步骤实际需要调用的工具；type="llm" 时为空列表
8. Skill 不是普通工具：绝不能把可用 Skill 名称（例如 url_accessor）写入 tools。
   若某一步需要调用一个 Skill，必须使用 type="skill"，并同时设置
   skill 和 nested_skill 为该 Skill 名称，tools 设为 []。特别是：网页搜索后
   读取搜索结果 URL 时，使用 nested_skill="url_accessor"，而不是 tools=["url_accessor"]。
"""


# ─── TaskPlanner 类 ───────────────────────────────────────────────────────────

class TaskPlanner:
    """
    任务规划器：将用户请求分解为结构化的执行步骤。
    输出格式与 skill.json 完全一致。

    Args:
        llm: 任何支持 .invoke(messages) 的 LLM 实例
    """

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    def plan(
        self,
        user_request: str,
        available_skills: Optional[List[str]] = None,
        skill_descriptions: Optional[dict] = None,
    ) -> TaskPlan:
        """
        为用户请求生成结构化执行计划（格式与 skill.json 对齐）。

        Args:
            user_request:       用户原始请求文本
            available_skills:   可用 skill 名称列表
            skill_descriptions: skill 名称 → 简介 映射（可选，用于更准确的选择）

        Returns:
            TaskPlan 实例（失败时返回单步 fallback 计划）
        """
        skills = available_skills or []
        descriptions = skill_descriptions or {}

        # 构建 skill 列表文本
        if skills:
            skill_lines = []
            for name in skills:
                desc = descriptions.get(name, "")
                if desc:
                    skill_lines.append(f"- {name}: {desc}")
                else:
                    skill_lines.append(f"- {name}")
            skill_list_text = "\n".join(skill_lines)
        else:
            skill_list_text = "（无可用 skill，所有步骤直接执行）"

        prompt = _PLANNER_USER_PROMPT_TEMPLATE.format(
            user_request=user_request,
            skill_list=skill_list_text,
        )

        try:
            from local_agent.core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self._llm.invoke(messages)
            raw = response.content if hasattr(response, "content") else str(response)
            plan = self._parse_plan(raw, user_request, skills)
            logger.info(
                "TaskPlanner: planned %d steps for request=%r",
                len(plan.steps),
                user_request[:60],
            )
            return plan
        except Exception as exc:
            logger.warning(
                "TaskPlanner: planning failed (%s), falling back to single-step plan",
                exc,
            )
            # Fallback：单步直接执行
            return TaskPlan(
                user_request=user_request,
                overall_strategy="直接执行（规划失败）",
                available_tools=[],
                input_spec={"query": "string - 用户原始请求"},
                output_spec={"result": "string - 执行结果"},
                steps=[
                    TaskStep(
                        step_id="step_1",
                        title="执行用户请求",
                        description=user_request[:100],
                        type="agent",
                        skill=None,
                        tools=[],
                        nested_skill=None,
                        input_spec={"query": "string - 用户原始请求"},
                        output_spec={"result": "string - 执行结果"},
                        reason="规划失败，降级为直接执行",
                    )
                ],
            )

    # ── 解析 ─────────────────────────────────────────────────────────────────

    def _parse_plan(
        self,
        raw: str,
        user_request: str,
        available_skills: Optional[List[str]] = None,
    ) -> TaskPlan:
        """解析 LLM 返回的 JSON 为 TaskPlan（格式与 skill.json 对齐）。"""
        text = raw.strip()

        # 去掉 markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text).strip()
        text = re.sub(r"```\s*$", "", text).strip()

        # 提取第一个 JSON 对象
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            text = json_match.group(0)

        data = json.loads(text)

        steps = []
        for i, s in enumerate(data.get("steps", [])):
            tools = s.get("tools") or []
            skill = s.get("skill") or None
            nested_skill = s.get("nested_skill") or None

            # Some smaller models confuse a skill with a callable tool.  Preserve
            # backward compatibility with such plans by converting the unambiguous
            # form `tools: ["url_accessor"]` into a nested-skill step.  The
            # executor can then expose invoke_skill instead of silently dropping
            # the unknown "tool".
            skill_names = set(available_skills or [])
            tool_skills = [name for name in tools if name in skill_names]
            if len(tool_skills) == 1 and len(tools) == 1 and not nested_skill:
                nested_skill = tool_skills[0]
                # `skill` selects a top-level graph; nested skills are invoked
                # through SkillTool and must not replace that graph.
                skill = None
                tools = []
                step_type = "skill"
                logger.warning(
                    "TaskPlanner: normalized skill %r mistakenly listed as a tool "
                    "in step %s",
                    nested_skill,
                    s.get("step_id", f"step_{i + 1}"),
                )
            else:
                step_type = s.get("type", "agent")

            steps.append(
                TaskStep(
                    step_id=s.get("step_id", f"step_{i+1}"),
                    title=s.get("title", f"步骤 {i+1}"),
                    description=s.get("description", ""),
                    type=step_type,
                    skill=skill,
                    tools=tools,
                    nested_skill=nested_skill,
                    input_spec=s.get("input_spec") or {},
                    output_spec=s.get("output_spec") or {},
                    reason=s.get("reason", ""),
                )
            )

        return TaskPlan(
            user_request=user_request,
            overall_strategy=data.get("overall_strategy", ""),
            available_tools=data.get("available_tools") or [],
            input_spec=data.get("input_spec") or {},
            output_spec=data.get("output_spec") or {},
            steps=steps,
        )
