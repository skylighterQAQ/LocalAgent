"""
local_agent.engine.llm.anthropic
==================================
Anthropic Messages HTTP LLM 客户端。

API 文档：https://docs.anthropic.com/en/api/messages

端点：POST /v1/messages

注意事项：
  - Anthropic 工具格式与 OpenAI 不同，使用 tool_use 内容块
  - system prompt 作为顶层字段，不作为消息列表中的一条
  - 流式响应使用 Server-Sent Events (SSE)
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
    SystemMessage,
    ToolCall,
)
from local_agent.core.tools import BaseTool

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"


def _build_tools_param(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """将工具列表转为 Anthropic tools 格式。"""
    result = []
    for t in tools:
        schema = t.get_schema()
        fn = schema.get("function", schema)
        result.append(
            {
                "name": fn.get("name", t.name),
                "description": fn.get("description", t.description),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return result


def _messages_to_anthropic(
    messages: List[BaseMessage],
) -> tuple[Optional[str], List[Dict[str, Any]]]:
    """
    将消息列表转为 Anthropic 格式。

    Returns:
        (system_prompt_or_None, anthropic_messages_list)
    """
    system_prompt: Optional[str] = None
    anthropic_msgs: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.role
        if role == "system":
            # Anthropic 要求 system 作为顶层字段，不在 messages 中
            system_prompt = msg.content
            continue
        elif role == "user":
            anthropic_msgs.append({"role": "user", "content": msg.content})
        elif role == "assistant":
            from local_agent.core.messages import AIMessage as _AIMsg
            if isinstance(msg, _AIMsg) and msg.tool_calls:
                content_blocks: List[Dict[str, Any]] = []
                if msg.content:
                    content_blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.args,
                        }
                    )
                anthropic_msgs.append({"role": "assistant", "content": content_blocks})
            else:
                anthropic_msgs.append({"role": "assistant", "content": msg.content})
        elif role == "tool":
            from local_agent.core.messages import ToolMessage as _ToolMsg
            # Anthropic 要求工具结果在 user 消息中，以 tool_result 块的形式
            if isinstance(msg, _ToolMsg):
                anthropic_msgs.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content,
                            }
                        ],
                    }
                )

    return system_prompt, anthropic_msgs


class AnthropicLLM(BaseLLM):
    """
    Anthropic Messages API HTTP 客户端。

    Args:
        model    : 模型名称，如 "claude-3-5-sonnet-20241022"
        api_key  : Anthropic API 密钥
        base_url : API 基础 URL，默认 https://api.anthropic.com
        max_tokens  : 最大生成 token 数
        temperature : 采样温度
        timeout  : HTTP 超时（秒）
    """

    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = _DEFAULT_API_URL,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 120.0,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

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
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                ) as resp:
                    resp.raise_for_status()
                    current_tool_calls: Dict[int, Dict[str, Any]] = {}
                    for line in resp.iter_lines():
                        line = line.strip()
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[len("data: "):]
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        chunk = self._parse_stream_event(data, current_tool_calls)
                        if chunk is not None:
                            yield chunk
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Anthropic streaming error: {exc}") from exc

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _build_payload(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        system_prompt, anthropic_msgs = _messages_to_anthropic(messages)
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "messages": anthropic_msgs,
            "stream": stream,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if tools:
            payload["tools"] = _build_tools_param(tools)
        return payload

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = self._build_headers()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Anthropic API error: {exc}") from exc

    def _parse_response(self, data: Dict[str, Any]) -> AIMessage:
        content_blocks: List[Dict[str, Any]] = data.get("content", [])
        text = ""
        tool_calls: List[ToolCall] = []
        for block in content_blocks:
            btype = block.get("type", "")
            if btype == "text":
                text += block.get("text", "")
            elif btype == "tool_use":
                tool_calls.append(
                    ToolCall(
                        name=block.get("name", ""),
                        args=block.get("input", {}),
                        id=block.get("id") or str(uuid.uuid4()),
                    )
                )
        return AIMessage(content=text, tool_calls=tool_calls)

    def _parse_stream_event(
        self,
        event: Dict[str, Any],
        current_tool_calls: Dict[int, Dict[str, Any]],
    ) -> Optional[AIMessageChunk]:
        """
        解析 Anthropic SSE 事件。

        Anthropic 流式事件类型（部分）：
          content_block_start   → 新内容块开始（type=text / tool_use）
          content_block_delta   → 内容块增量（text_delta / input_json_delta）
          content_block_stop    → 内容块结束
          message_stop          → 消息结束
        """
        etype = event.get("type", "")

        if etype == "content_block_start":
            idx = event.get("index", 0)
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                current_tool_calls[idx] = {
                    "id": block.get("id") or str(uuid.uuid4()),
                    "name": block.get("name", ""),
                    "args_buf": "",
                }
            return None

        elif etype == "content_block_delta":
            idx = event.get("index", 0)
            delta = event.get("delta", {})
            dtype = delta.get("type", "")
            if dtype == "text_delta":
                text = delta.get("text", "")
                return AIMessageChunk(content=text) if text else None
            elif dtype == "input_json_delta":
                if idx in current_tool_calls:
                    current_tool_calls[idx]["args_buf"] += delta.get("partial_json", "")
            return None

        elif etype == "content_block_stop":
            idx = event.get("index", 0)
            if idx in current_tool_calls:
                tc_data = current_tool_calls.pop(idx)
                args_str = tc_data.get("args_buf", "{}")
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    args = {"raw": args_str}
                tc = ToolCall(name=tc_data["name"], args=args, id=tc_data["id"])
                return AIMessageChunk(content="", tool_calls=[tc])
            return None

        return None
