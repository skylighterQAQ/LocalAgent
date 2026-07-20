"""
Claude Code (Anthropic) Provider
==================================
支持 Claude 模型，直接调用 Anthropic HTTP API，不再依赖 langchain-anthropic。

配置方式（环境变量或 config.yaml）：
  - CLAUDE_CODE_API_KEY        : Anthropic API 密钥
  - CLAUDE_CODE_BASE_URL       : 自定义端点（默认 https://api.anthropic.com）
  - CLAUDE_CODE_DEFAULT_MODEL  : 默认模型（默认 claude-opus-4-5）
"""
from typing import Optional, List

from local_agent.llm.clients.anthropic import AnthropicLLM
from local_agent.llm.base import BaseLLM
from local_agent.core.config import get_settings


class ClaudeCodeProvider:
    """管理 Claude Code (Anthropic) LLM 连接与模型选择。"""

    KNOWN_MODELS = [
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.claude_code_api_key
        self.base_url = base_url or settings.claude_code_base_url
        self.model = model or settings.claude_code_default_model
        self.timeout = timeout or settings.claude_code_timeout
        self.extra_kwargs = kwargs

    def get_llm(self, model: Optional[str] = None, **kwargs) -> AnthropicLLM:
        """返回 AnthropicLLM 实例。"""
        model_name = model or self.model
        temperature = kwargs.get("temperature", 0.7)
        return AnthropicLLM(
            model=model_name,
            api_key=self.api_key or "",
            base_url=self.base_url or "https://api.anthropic.com",
            temperature=temperature,
            timeout=float(self.timeout or 120),
        )

    def get_llm_with_tools(
        self,
        tools: list,
        model: Optional[str] = None,
        **kwargs,
    ) -> BaseLLM:
        """返回已绑定工具的 AnthropicLLM 实例。"""
        llm = self.get_llm(model=model, **kwargs)
        return llm.bind_tools(tools)

    def list_models(self) -> List[str]:
        """返回已知 Claude 模型列表。"""
        return list(self.KNOWN_MODELS)

    def check_connection(self) -> bool:
        """如果 API Key 已配置则返回 True（不发起实际网络请求）。"""
        return bool(self.api_key and self.api_key.strip())
