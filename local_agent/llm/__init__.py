"""
LLM Integration Layer
======================
统一 LLM 层，整合底层 HTTP 客户端、Provider 封装和工厂函数。

结构：
  llm/
  ├── base.py          - BaseLLM, BoundLLM 抽象接口
  ├── clients/         - 底层 HTTP 客户端
  │   ├── ollama.py    - OllamaLLM
  │   ├── openai_compat.py - OpenAILLM
  │   └── anthropic.py - AnthropicLLM
  ├── providers/       - Provider 封装（配置管理）
  │   ├── ollama.py    - OllamaProvider
  │   ├── openai.py    - OpenAIProvider
  │   ├── wanqing.py   - WanqingProvider
  │   └── claude_code.py - ClaudeCodeProvider
  └── factory.py       - get_llm_provider() 单例工厂
"""
from local_agent.llm.base import BaseLLM, BoundLLM
from local_agent.llm.clients.ollama import OllamaLLM
from local_agent.llm.clients.openai_compat import OpenAILLM
from local_agent.llm.clients.anthropic import AnthropicLLM
from local_agent.llm.providers.ollama import OllamaProvider
from local_agent.llm.providers.openai import OpenAIProvider
from local_agent.llm.providers.wanqing import WanqingProvider
from local_agent.llm.providers.claude_code import ClaudeCodeProvider
from local_agent.llm.factory import get_llm_provider, reset_llm_provider

__all__ = [
    # 抽象基类
    "BaseLLM",
    "BoundLLM",
    # 底层客户端
    "OllamaLLM",
    "OpenAILLM",
    "AnthropicLLM",
    # Provider 封装
    "OllamaProvider",
    "OpenAIProvider",
    "WanqingProvider",
    "ClaudeCodeProvider",
    # 工厂函数
    "get_llm_provider",
    "reset_llm_provider",
]
