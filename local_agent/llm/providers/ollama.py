"""
Ollama LLM Provider
====================
直接调用 Ollama HTTP API，不再依赖 langchain-ollama。
"""
from typing import Optional, List

from local_agent.llm.clients.ollama import OllamaLLM
from local_agent.llm.base import BaseLLM
from local_agent.core.config import get_settings


class OllamaProvider:
    """管理 Ollama LLM 连接与模型选择。"""

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_default_model

    def get_llm(self, model: Optional[str] = None, **kwargs) -> OllamaLLM:
        """返回 OllamaLLM 实例。
        
        Args:
            model: 模型名称，默认使用 self.model
            **kwargs: 额外参数，支持：
                - temperature: 采样温度（默认 0.1）
                - timeout: HTTP 请求超时（秒），默认读取 settings.ollama_timeout
                - disable_thinking: 是否禁用 thinking 模式（None 表示自动检测）
        """
        settings = get_settings()
        model_name = model or self.model
        temperature = kwargs.get("temperature", 0.1)
        timeout = kwargs.get("timeout", settings.ollama_timeout)
        disable_thinking = kwargs.get("disable_thinking", None)
        return OllamaLLM(
            model=model_name,
            base_url=self.base_url,
            temperature=temperature,
            timeout=timeout,
            disable_thinking=disable_thinking,
        )

    def get_llm_with_tools(self, tools: list, model: Optional[str] = None) -> BaseLLM:
        """返回已绑定工具的 OllamaLLM 实例。"""
        llm = self.get_llm(model=model)
        return llm.bind_tools(tools)

    def list_models(self) -> List[str]:
        """列出 Ollama 可用模型。"""
        llm = OllamaLLM(model=self.model, base_url=self.base_url)
        return llm.list_models()

    def check_connection(self) -> bool:
        """检查 Ollama 是否可达。"""
        llm = OllamaLLM(model=self.model, base_url=self.base_url)
        return llm.check_connection()
