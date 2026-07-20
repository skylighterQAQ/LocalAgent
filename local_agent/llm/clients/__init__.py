"""
local_agent.llm.clients
========================
底层 LLM HTTP 客户端包。

提供：
  - BaseLLM        : 统一接口抽象类（local_agent.llm.base）
  - OllamaLLM      : Ollama /api/chat HTTP 客户端
  - OpenAILLM      : OpenAI-compatible /chat/completions HTTP 客户端
  - AnthropicLLM   : Anthropic /v1/messages HTTP 客户端
"""
from local_agent.llm.base import BaseLLM, BoundLLM
from local_agent.llm.clients.ollama import OllamaLLM
from local_agent.llm.clients.openai_compat import OpenAILLM
from local_agent.llm.clients.anthropic import AnthropicLLM

__all__ = [
    "BaseLLM",
    "BoundLLM",
    "OllamaLLM",
    "OpenAILLM",
    "AnthropicLLM",
]
