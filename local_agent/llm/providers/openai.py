"""
OpenAI-compatible API Provider
================================
直接调用 OpenAI-compatible HTTP API，不再依赖 langchain-openai。
支持 OpenAI 官方 API 及所有兼容服务（Azure OpenAI、本地代理等）。
"""
from typing import Optional, List, Dict, Any

from local_agent.llm.clients.openai_compat import OpenAILLM
from local_agent.llm.base import BaseLLM
from local_agent.core.config import get_settings


class OpenAIProvider:
    """管理 OpenAI-compatible LLM 连接与模型选择。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url
        self.model = model or settings.openai_default_model
        self.extra_kwargs = kwargs

    def get_llm(self, model: Optional[str] = None, **kwargs) -> OpenAILLM:
        """返回 OpenAILLM 实例。"""
        model_name = model or self.model
        temperature = kwargs.get("temperature", 0.7)
        return OpenAILLM(
            model=model_name,
            api_key=self.api_key or "",
            base_url=self.base_url or "https://api.openai.com/v1",
            temperature=temperature,
        )

    def get_llm_with_tools(
        self,
        tools: list,
        model: Optional[str] = None,
        **kwargs,
    ) -> BaseLLM:
        """返回已绑定工具的 OpenAILLM 实例。"""
        llm = self.get_llm(model=model, **kwargs)
        return llm.bind_tools(tools)

    def list_models(self) -> List[str]:
        """返回常见 OpenAI 模型列表（API 不提供枚举接口）。"""
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
        ]

    def check_connection(self) -> bool:
        """检查 API Key 是否已配置（不发起实际请求）。"""
        return bool(self.api_key and self.api_key.strip())
