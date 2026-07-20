"""
local_agent.skills.parsed_config
==================================
定义 Skill 解析后的结构化数据模型。

SkillParser 将 SKILL.md 转换为 ParsedSkillConfig，其中包含：
  - skill 整体描述、工具列表、输入输出规格
  - skill 公共 prompt（每步均注入）
  - 每个步骤的详细规格（类型、工具、输入输出映射、失败处理、专属 step_prompt）
  - 每步的期望输入输出校验规格及不满足时的处理策略

这些结构化信息由 SkillExecutor 用于步骤隔离执行：
  - 每步只从上下文提取 input_mapping 声明的变量
  - 每步只传入上一步的输出结果，不传入上一步的输入细节
  - 每步 LLM 调用消息全新构建，不累积历史
  - 每步注入 skill 公共 prompt + 步骤专属 prompt
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """步骤执行类型。"""
    TOOL  = "tool"   # 直接工具调用，完全不经过 LLM。支持 tools 列表中多个工具按序调用，
                     # 参数来自 tool_params 模板（支持 {变量名} 占位符替换）；
                     # tool_params 为空时直接使用 step_input 作为每个工具的参数。
    LLM   = "llm"   # 纯 LLM 推理，不调用工具
    AGENT = "agent" # 工具 + LLM ReAct 循环
    SKILL = "skill" # 嵌套调用另一个 skill


class StepSpec(BaseModel):
    """单个步骤的完整规格。"""

    id: str = Field(description="步骤唯一标识，如 'step_1'")
    name: str = Field(description="步骤人读名称")
    type: StepType = Field(description="步骤执行类型")
    description: str = Field(default="", description="该步骤的详细说明（做什么）")

    # 工具与嵌套 skill
    tools: List[str] = Field(
        default_factory=list,
        description="本步骤可调用的工具名列表（TOOL/AGENT 类型有效）",
    )
    tool_params: List[Dict[str, object]] = Field(
        default_factory=list,
        description=(
            "TOOL 类型专用：按顺序调用 tools 列表中每个工具的参数模板。"
            "每个元素是一个字典，支持 '{变量名}' 占位符（变量从 step_input 中取值）。"
            "例如: [{\"path\": \"{module_dir}\"}, {\"path\": \"{design_path}\", \"content\": \"# placeholder\"}]。"
            "为空时直接用 step_input 作为每个工具的参数。"
        ),
    )
    nested_skill: Optional[str] = Field(
        default=None,
        description="嵌套调用的 skill 名称（仅 SKILL 类型有效）",
    )

    # 输入输出映射
    input_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "本步骤从执行上下文中取值的映射: "
            "{'本步参数名': '上下文变量名'}。"
            "只有这里声明的变量会传入本步骤，隔离前序步骤的输入细节。"
        ),
    )
    output_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "本步骤执行结果写回上下文的映射: "
            "{'上下文变量名': '本步返回结果中的键名'}。"
        ),
    )

    # 期望输入输出规格（说明性，用于 LLM 解析和运行时校验）
    input_spec: Dict[str, str] = Field(
        default_factory=dict,
        description="期望输入说明: {'参数名': '类型/描述'}",
    )
    output_spec: Dict[str, str] = Field(
        default_factory=dict,
        description="期望输出说明: {'结果键': '类型/描述'}",
    )

    # 输入输出校验失败策略
    input_validation_policy: str = Field(
        default="warn",
        description=(
            "输入不满足 input_spec 时的处理策略: "
            "'raise' - 抛出异常; "
            "'warn' - 打印警告继续; "
            "'ignore' - 静默继续"
        ),
    )
    output_validation_policy: str = Field(
        default="warn",
        description=(
            "输出不满足 output_spec 时的处理策略: "
            "'raise' - 抛出异常; "
            "'warn' - 打印警告继续; "
            "'ignore' - 静默继续"
        ),
    )

    # 错误处理
    on_failure: str = Field(
        default="raise",
        description=(
            "失败处理策略: "
            "'raise' - 抛出异常停止执行; "
            "'skip' - 跳过本步骤继续; "
            "'retry:N' - 最多重试 N 次; "
            "'fallback:step_id' - 跳转到指定步骤"
        ),
    )

    # 本步骤专属 system_prompt（向后兼容旧字段名）
    system_prompt: Optional[str] = Field(
        default=None,
        description="[已弃用，请用 step_prompt] 本步骤的专属系统指令",
    )

    # 新增：本步骤专属 prompt（优先于 system_prompt）
    step_prompt: Optional[str] = Field(
        default=None,
        description=(
            "本步骤专属 prompt：仅包含与本步直接相关的执行指令和约束，"
            "不含全局规则或其他步骤内容。LLM/AGENT 类型步骤有效。"
        ),
    )

    def get_effective_step_prompt(self) -> Optional[str]:
        """返回有效的步骤 prompt（优先用 step_prompt，其次 system_prompt）。"""
        return self.step_prompt or self.system_prompt


class ParsedSkillConfig(BaseModel):
    """
    SKILL.md 经 SkillParser 解析后的完整结构化配置。

    由 SkillExecutor 驱动执行，保证每步 LLM 调用的消息输入精简且隔离。
    
    新增字段：
      - skill_prompt: Skill 公共 prompt，每步执行时均注入（定位、工作方式等）
    """

    skill_name: str
    version: str = "1.0.0"
    description: str = Field(default="", description="一句话描述")
    overview: str = Field(
        default="",
        description="Skill 整体目的，1-2 句话。每次步骤调用时作为 Skill 背景注入，应精简。",
    )

    # 新增：Skill 公共 prompt（每步均注入）
    skill_prompt: str = Field(
        default="",
        description=(
            "Skill 公共 prompt：每步执行时均注入到 SystemMessage 中。"
            "包含 skill 的核心行为规范、全局约束和通用工作方式。"
            "不含步骤特有内容（那些放在 step_prompt 中）。"
        ),
    )

    # 子 skill 标记：True 表示该 skill 仅供主 skill 内部调用
    is_sub_skill: bool = Field(
        default=False,
        description=(
            "子 skill 标记：True 表示该 skill 仅供主 skill 内部的 SKILL 步骤或 invoke_skill 工具调用，"
            "不参与 SkillSelector 的自动选择（用户无法直接触发）。"
            "主 skill（如 greenfield_developer）应设为 False（默认值）。"
        ),
    )

    # 仅对 is_sub_skill=True 有效：声明哪些父 skill 允许调用本 sub_skill
    allowed_parent_skills: Optional[List[str]] = Field(
        default=None,
        description=(
            "仅对 is_sub_skill=True 的 skill 有效。"
            "设置后，只有列表中的 skill（按 current_skill_name 匹配）才能通过 invoke_skill 工具调用本 sub_skill。"
            "None 表示不限制（所有允许调用 sub_skill 的父 skill 均可调用）。"
        ),
    )

    # 全量工具列表（来自 SKILL.md required_tools）
    available_tools: List[str] = Field(default_factory=list)

    # 整体输入输出规格
    input_spec: Dict[str, str] = Field(
        default_factory=dict,
        description="整体输入期望: {'参数名': '描述'}",
    )
    output_spec: Dict[str, str] = Field(
        default_factory=dict,
        description="整体输出期望: {'结果键': '描述'}",
    )
    on_input_mismatch: str = Field(
        default="raise",
        description="整体输入不符时的策略: 'raise' | 'warn' | 'ignore'",
    )
    on_output_mismatch: str = Field(
        default="raise",
        description="整体输出不符时的策略: 'raise' | 'warn' | 'ignore'",
    )

    # 步骤列表（有序）
    steps: List[StepSpec] = Field(default_factory=list)

    # 缓存元数据
    parsed_at: str = Field(default="", description="解析时间 ISO 字符串")
    source_md5: str = Field(default="", description="SKILL.md 内容的 MD5，用于缓存失效检测")

    def get_step(self, step_id: str) -> Optional[StepSpec]:
        """按 id 查找步骤。"""
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def to_prompt_overview(self) -> str:
        """
        生成注入每步 LLM 调用的 Skill 背景字符串（精简版）。
        格式：'<skill_name>: <overview>'
        """
        return f"{self.skill_name}: {self.overview}".strip()
