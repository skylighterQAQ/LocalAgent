"""
LocalAgent Core – Agent 调度层与运行时引擎

此包整合了原 core/ 和 engine/ 两个模块：
  - agent    : LocalAgent 主类（编排 LLM、工具、技能、记忆）
  - config   : 配置管理（Settings / get_settings）
  - graph    : ReAct 图（create_agent_graph）
  - state    : AgentState 状态定义
  - debug    : 调试输出工具
  - prompt_registry : 基于调用 ID 的 Prompt 上下文注册表
  - engine/  : 底层运行时（消息类型、工具基类、LLM 客户端、ReAct 引擎）
"""
from local_agent.core.agent import LocalAgent
from local_agent.core.config import Settings, get_settings, reset_settings

# 从 engine 子包再导出，方便外部直接从 local_agent.core 引用
from local_agent.core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage,
    ToolCall,
)
from local_agent.core.tools import BaseTool, tool, LocalAgentTool, tool_metadata
from local_agent.core.prompt_registry import (
    PromptContextEntry,
    PromptContextRegistry,
    get_registry,
    generate_invocation_id,
    save_prompt_context,
    retrieve_prompt_context,
)

__all__ = [
    # Agent 层
    "LocalAgent",
    "Settings",
    "get_settings",
    "reset_settings",
    # 消息类型
    "BaseMessage",
    "SystemMessage",
    "HumanMessage",
    "AIMessage",
    "AIMessageChunk",
    "ToolMessage",
    "ToolCall",
    # 工具基类
    "BaseTool",
    "tool",
    "LocalAgentTool",
    "tool_metadata",
    # Prompt 上下文注册表
    "PromptContextEntry",
    "PromptContextRegistry",
    "get_registry",
    "generate_invocation_id",
    "save_prompt_context",
    "retrieve_prompt_context",
]
