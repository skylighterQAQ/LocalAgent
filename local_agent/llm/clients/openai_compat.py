"""
local_agent.engine.llm.openai_compat
======================================
OpenAI-compatible HTTP LLM 客户端。

支持所有兼容 OpenAI Chat Completions API 的服务：
  - OpenAI (api.openai.com)
  - Wanqing (快手内部，OpenAI-compatible)
  - 其他兼容接口

端点：POST /chat/completions
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Iterator, List, Optional

import httpx

from local_agent.llm.base import BaseLLM
from local_agent.core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolCall,
    messages_to_openai,
)
from local_agent.core.tools import BaseTool

logger = logging.getLogger(__name__)


def _parse_tool_calls(raw_calls: List[Dict[str, Any]]) -> List[ToolCall]:
    """解析 OpenAI 格式的 tool_calls。"""
    result: List[ToolCall] = []
    for tc in raw_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"raw": args_raw}
        else:
            args = args_raw or {}
        call_id = tc.get("id") or str(uuid.uuid4())
        result.append(ToolCall(name=name, args=args, id=call_id))
    return result


def _build_tools_param(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    return [t.get_schema() for t in tools]


class OpenAILLM(BaseLLM):
    """
    OpenAI-compatible Chat Completions HTTP 客户端。

    Args:
        model    : 模型名称，如 "gpt-4o"
        api_key  : API 密钥
        base_url : API 基础 URL（不含 /chat/completions），
                   默认 https://api.openai.com/v1
        temperature : 采样温度
        timeout  : HTTP 超时（秒）
        extra_headers : 额外 HTTP 请求头
    """

    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.7,
        timeout: float = 120.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self.extra_headers: Dict[str, str] = extra_headers or {}

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def invoke(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        payload = self._build_payload(messages, tools, stream=False, **kwargs)
        resp = self._post(payload)
        return self._parse_response(resp)

    def stream(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        payload = self._build_payload(messages, tools, stream=True, **kwargs)
        headers = self._build_headers()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    # SSE 格式：data: {...}\n\n
                    for line in resp.iter_lines():
                        line = line.strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            line = line[len("data: "):]
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk = self._parse_stream_chunk(data)
                        if chunk is not None:
                            yield chunk
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenAI streaming error: {exc}") from exc

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _build_payload(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages_to_openai(messages),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": stream,
        }
        if tools:
            payload["tools"] = _build_tools_param(tools)
        return payload

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = self._build_headers()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc

    def _parse_response(self, data: Dict[str, Any]) -> AIMessage:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        content: str = msg.get("content") or ""
        raw_tool_calls: List[Dict[str, Any]] = msg.get("tool_calls") or []
        tool_calls = _parse_tool_calls(raw_tool_calls)
        return AIMessage(content=content, tool_calls=tool_calls)

    def _parse_stream_chunk(self, data: Dict[str, Any]) -> Optional[AIMessageChunk]:
        choice = (data.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        content: str = delta.get("content") or ""
        raw_tool_calls: List[Dict[str, Any]] = delta.get("tool_calls") or []
        tool_calls = _parse_tool_calls(raw_tool_calls)
        if not content and not tool_calls:
            return None
        return AIMessageChunk(content=content, tool_calls=tool_calls)
