"""
local_agent.llm.providers
==========================
LLM Provider 封装层，提供对底层 LLM 客户端的高层封装。

每个 Provider 管理连接配置、模型选择，并提供 get_llm() / get_llm_with_tools() 接口。
"""
from local_agent.llm.providers.ollama import OllamaProvider
from local_agent.llm.providers.openai import OpenAIProvider
from local_agent.llm.providers.wanqing import WanqingProvider
from local_agent.llm.providers.claude_code import ClaudeCodeProvider

__all__ = [
    "OllamaProvider",
    "OpenAIProvider",
    "WanqingProvider",
    "ClaudeCodeProvider",
]
