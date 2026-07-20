"""
Agent Step - Run a ReAct agent loop within a SubAgent step

Unlike LLMStep (pure LLM call), AgentStep runs a full ReAct cycle and can call
any registered tools. This enables each phase of a multi-phase workflow to
autonomously use filesystem, code-execution, or any other tools.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from local_agent.core.messages import AIMessage, HumanMessage, SystemMessage
from local_agent.core.subagent.context import ExecutionContext
from local_agent.core.subagent.steps.base import BaseStep

logger = logging.getLogger(__name__)


class AgentStep(BaseStep):
    """
    Agent 步骤 - 在 SubAgent 工作流中运行带工具调用能力的 ReAct Agent。

    与 LLMStep 的区别
    ------------------
    - LLMStep  : 单轮纯 LLM 对话，不调用任何工具
    - AgentStep: 完整 ReAct（Reason + Act）循环，可按需调用指定工具

    主要参数
    --------
    prompt : str
        任务描述提示词模板，支持 ``{variable}`` 格式的上下文变量替换。
    tools : list[str]
        要开放给 Agent 的工具名称列表（从全局 ToolRegistry 中解析）。
        如果某工具未注册，会打印 Warning 并跳过，不会中断执行。
    system_prompt : str, optional
        系统提示词。
    model : str, optional
        使用的 LLM 模型名称；为 None 时使用全局默认模型。
    provider : str, optional
        LLM 提供商（'ollama' / 'openai' / 'wanqing' / 'claude_code'）；
        为 None 时自动检测。
    max_iterations : int
        ReAct 最大迭代次数，防止无限循环，默认 20。
    tool_call_retry : bool
        是否在模型未调用工具时引导重试，默认 True。
    max_tool_retry : int
        引导重试的最大次数，默认 2。

    示例
    ----
    .. code-block:: python

        from local_agent.core.subagent import SubAgent, AgentStep

        agent = SubAgent("code_generator")
        agent.add_step(AgentStep(
            name="write_module",
            prompt="根据以下架构文档，实现 {file_path} 文件:\\n\\n{architecture}",
            tools=["fs_write_file", "fs_read_file", "fs_create_dir", "code_lint_multi"],
            system_prompt="你是资深软件工程师，请严格按照架构文档实现代码。",
            model="qwen2.5:14b",
            max_iterations=15,
            output_key="result",
        ))

    YAML 配置格式（供 SubAgent config.py 使用）::

        type: agent
        name: implement_file
        prompt: "实现文件 {file_path}: {architecture_md}"
        tools:
          - fs_write_file
          - fs_read_file
          - fs_create_dir
          - code_lint_multi
        system_prompt: "你是资深软件工程师"
        model: "qwen2.5:14b"
        max_iterations: 15
        output_key: result
    """

    def __init__(
        self,
        prompt: str,
        tools: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_iterations: int = 20,
        tool_call_retry: bool = True,
        max_tool_retry: int = 2,
        **kwargs: Any,
    ) -> None:
        """
        初始化 AgentStep。

        Args:
            prompt: 任务描述提示词（支持 {variable} 变量替换）
            tools: 工具名称列表（从 ToolRegistry 获取）
            system_prompt: 系统提示词
            model: 模型名称
            provider: LLM 提供商
            max_iterations: 最大 ReAct 迭代次数
            tool_call_retry: 是否启用工具调用引导重试
            max_tool_retry: 引导重试最大次数
            **kwargs: 传递给 BaseStep 的参数（name, output_key, retry_count 等）
        """
        super().__init__(**kwargs)

        self.prompt_template = prompt
        self.tool_names: List[str] = tools or []
        self.system_prompt = system_prompt
        self.model = model
        self.provider = provider
        self.max_iterations = max_iterations
        self.tool_call_retry = tool_call_retry
        self.max_tool_retry = max_tool_retry

    # ------------------------------------------------------------------
    # 私有辅助方法
    # ------------------------------------------------------------------

    def _resolve_tools(self) -> list:
        """
        从全局 ToolRegistry 解析工具名称列表，返回 BaseTool 实例列表。

        找不到的工具会记录 Warning，不会抛出异常。
        """
        from local_agent.tools.registry import ToolRegistry
        registry = ToolRegistry()
        resolved = []

        for tool_name in self.tool_names:
            tool = registry.get(tool_name)
            if tool is not None:
                resolved.append(tool)
                logger.debug("AgentStep '%s': resolved tool '%s'", self.name, tool_name)
            else:
                logger.warning(
                    "AgentStep '%s': tool '%s' not found in ToolRegistry, skipping",
                    self.name,
                    tool_name,
                )

        return resolved

    def _get_llm(self):
        """获取 LLM Provider（懒加载）。"""
        from local_agent.llm.factory import get_llm_provider

        return get_llm_provider(model=self.model, provider=self.provider)

    # ------------------------------------------------------------------
    # BaseStep.execute 实现
    # ------------------------------------------------------------------

    def execute(self, context: ExecutionContext) -> Any:
        """
        执行 AgentStep：运行一个完整的 ReAct 循环。

        Args:
            context: 执行上下文（用于变量解析与结果保存）

        Returns:
            最后一条 AIMessage 的内容字符串；如果 messages 为空则返回空字符串。
        """
        from local_agent.core.react import ReActEngine

        # 1. 解析 prompt 模板
        prompt = context.resolve(self.prompt_template)
        logger.debug("AgentStep '%s' resolved prompt (first 200 chars): %s", self.name, str(prompt)[:200])

        # 2. 解析工具
        tools = self._resolve_tools()
        logger.info(
            "AgentStep '%s' starting with %d tools: %s",
            self.name,
            len(tools),
            [t.name for t in tools],
        )

        # 3. 构建 LLM 并创建 ReActEngine
        llm = self._get_llm()
        engine = ReActEngine(
            llm=llm,
            tools=tools,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations,
            tool_call_retry=self.tool_call_retry,
            max_tool_retry=self.max_tool_retry,
        )

        # 4. 执行 ReAct 循环
        initial_state = {"messages": [HumanMessage(content=prompt)]}
        result_state = engine.invoke(initial_state)

        # 5. 提取最后一条 AIMessage 的内容
        messages = result_state.get("messages", [])
        final_content = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_content = msg.content
                break

        logger.info(
            "AgentStep '%s' completed. Result (first 200 chars): %s",
            self.name,
            final_content[:200],
        )
        return final_content

    def __repr__(self) -> str:
        return (
            f"AgentStep(name='{self.name}', "
            f"tools={self.tool_names}, "
            f"model='{self.model}')"
        )
