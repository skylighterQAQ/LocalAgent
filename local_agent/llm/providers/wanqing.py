"""
Wanqing API Provider
====================
快手内部 Wanqing 模型服务，使用 OpenAI-compatible API 格式。
直接调用 HTTP API，不再依赖 langchain-openai。
"""
from typing import Optional, List

from local_agent.llm.clients.openai_compat import OpenAILLM
from local_agent.llm.base import BaseLLM
from local_agent.core.config import get_settings


class WanqingProvider:
    """管理 Wanqing API 连接与模型选择。"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ):
        settings = get_settings()
        self.api_key = api_key or settings.wanqing_api_key
        self.base_url = base_url or settings.wanqing_base_url
        self.model = model or settings.wanqing_default_model
        self.timeout = timeout or settings.wanqing_timeout
        self.extra_kwargs = kwargs

    def get_llm(self, model: Optional[str] = None, **kwargs) -> OpenAILLM:
        """返回配置好的 OpenAILLM 实例（指向 Wanqing 端点）。"""
        model_name = model or self.model
        return OpenAILLM(
            model=model_name,
            api_key=self.api_key or "",
            base_url=self.base_url or "",
            timeout=float(self.timeout or 120),
        )

    def get_llm_with_tools(
        self,
        tools: list,
        model: Optional[str] = None,
        **kwargs,
    ) -> BaseLLM:
        """返回已绑定工具的 LLM 实例。"""
        llm = self.get_llm(model=model, **kwargs)
        return llm.bind_tools(tools)

    def list_models(self) -> List[str]:
        """返回当前配置的模型名称。"""
        return [self.model]

    def check_connection(self) -> bool:
        """检查 API Key 和 base_url 是否已配置。"""
        if not self.api_key or not self.base_url:
            return False
        return bool(self.api_key.strip() and self.base_url.strip())
