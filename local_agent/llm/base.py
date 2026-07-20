"""
local_agent.engine.llm.base
============================
LLM 客户端抽象基类与 BoundLLM 包装器。

所有 LLM 客户端均继承 BaseLLM，实现以下接口：
  invoke()        → AIMessage（完整响应）
  stream()        → Iterator[AIMessageChunk]（逐 token 流式）
  bind_tools()    → BoundLLM（注入工具定义后的新客户端）
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Dict, Iterator, List, Optional

from local_agent.core.messages import AIMessage, AIMessageChunk, BaseMessage
from local_agent.core.tools import BaseTool


logger = logging.getLogger(__name__)


class BaseLLM:
    """LLM 客户端统一抽象接口。"""

    # ------------------------------------------------------------------
    # 必须由子类实现
    # ------------------------------------------------------------------

    @abstractmethod
    def invoke(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """发送消息列表，返回完整 AIMessage。"""
        ...

    @abstractmethod
    def stream(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        """发送消息列表，逐 token yield AIMessageChunk。"""
        ...

    # ------------------------------------------------------------------
    # bind_tools – 返回已注入工具定义的 BoundLLM
    # ------------------------------------------------------------------

    def bind_tools(self, tools: List[BaseTool]) -> "BoundLLM":
        """
        返回注入了工具定义的包装器。调用 invoke/stream 时会自动传入 tools。

        用法::

            llm = OllamaLLM(model="qwen2.5:7b")
            llm_with_tools = llm.bind_tools(my_tools)
            response = llm_with_tools.invoke(messages)
        """
        return BoundLLM(llm=self, tools=tools)


class BoundLLM(BaseLLM):
    """
    bind_tools() 返回的包装器，持有工具列表并透传给底层 LLM。
    """

    def __init__(self, llm: BaseLLM, tools: List[BaseTool]):
        self._llm = llm
        self._tools: List[BaseTool] = tools

    def invoke(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        effective_tools = tools if tools is not None else self._tools
        return self._llm.invoke(messages, tools=effective_tools, **kwargs)

    def stream(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        effective_tools = tools if tools is not None else self._tools
        return self._llm.stream(messages, tools=effective_tools, **kwargs)

    def bind_tools(self, tools: List[BaseTool]) -> "BoundLLM":
        """重新绑定工具（替换，不叠加）。"""
        return BoundLLM(llm=self._llm, tools=tools)
