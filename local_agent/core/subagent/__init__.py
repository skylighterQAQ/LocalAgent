"""
LocalAgent Core SubAgent System

提供灵活的编排能力，支持通过 Python 代码或配置文件定义线性工作流：
- 大模型调用（可选择不同模型）
- 工具调用（本地工具、MCP 工具）
- SubAgent 嵌套调用

示例::

    from local_agent.core.subagent import SubAgent, LLMStep, ToolStep
    
    agent = SubAgent("my_workflow")
    agent.add_step(LLMStep(model="qwen2.5:7b", prompt="分析文本"))
    agent.add_step(ToolStep(tool="file_writer", params={"path": "output.txt"}))
    
    result = agent.run(input_data={"text": "Hello World"})
"""

from local_agent.core.subagent.subagent import SubAgent
from local_agent.core.subagent.context import ExecutionContext
from local_agent.core.subagent.steps import BaseStep, LLMStep, ToolStep, SubAgentStep, AgentStep
from local_agent.core.subagent.config import (
    load_subagent_from_file,
    create_subagent_from_config,
    save_subagent_to_file,
)

__all__ = [
    "SubAgent",
    "ExecutionContext",
    "BaseStep",
    "LLMStep",
    "ToolStep",
    "SubAgentStep",
    "AgentStep",
    "load_subagent_from_file",
    "create_subagent_from_config",
    "save_subagent_to_file",
]
