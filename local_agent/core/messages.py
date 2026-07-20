"""
local_agent.engine.messages
============================
自主实现的消息类型，完全替换 langchain_core.messages。

消息角色映射（遵循 OpenAI Chat Completions 规范）：
  - SystemMessage   → role="system"
  - HumanMessage    → role="user"
  - AIMessage       → role="assistant"
  - ToolMessage     → role="tool"

ToolCall 数据类描述一次工具调用请求（由 AIMessage 携带）。
AIMessageChunk 用于流式输出，可通过 + 运算符累积成完整 AIMessage。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# ToolCall – 单次工具调用请求
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """描述 LLM 请求调用某个工具。"""
    name: str
    args: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def get(self, key: str, default=None):
        """dict-like get，兼容旧代码 tool_call.get('name') 等写法。"""
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __contains__(self, key: str):
        return hasattr(self, key)


# ---------------------------------------------------------------------------
# 消息基类
# ---------------------------------------------------------------------------

class BaseMessage:
    """所有消息类型的基类。"""

    role: str = "base"

    def __init__(self, content: str, **kwargs):
        self.content: str = content
        # 允许子类扩展额外属性（如 tool_calls）
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(content={self.content!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseMessage):
            return NotImplemented
        return self.__class__ is other.__class__ and self.content == other.content

    # ------------------------------------------------------------------
    # 序列化 – 转为 OpenAI-compatible dict
    # ------------------------------------------------------------------

    def to_openai_dict(self) -> Dict[str, Any]:
        """将消息转换为 OpenAI Chat Completions 格式的字典。"""
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# 具体消息类型
# ---------------------------------------------------------------------------

class SystemMessage(BaseMessage):
    """系统提示消息（role="system"）。"""
    role: str = "system"


class HumanMessage(BaseMessage):
    """用户输入消息（role="user"）。"""
    role: str = "user"


class AIMessage(BaseMessage):
    """
    LLM 回复消息（role="assistant"）。

    tool_calls: 该消息包含的工具调用请求列表（可为空列表）。
    """
    role: str = "assistant"

    def __init__(
        self,
        content: str = "",
        tool_calls: Optional[List[ToolCall]] = None,
        **kwargs,
    ):
        super().__init__(content, **kwargs)
        self.tool_calls: List[ToolCall] = tool_calls or []

    def to_openai_dict(self) -> Dict[str, Any]:
        # 当有 tool_calls 且 content 为空字符串时，将 content 设为 None（null）。
        # Ollama（及遵循 OpenAI 规范的后端）要求：带 tool_calls 的 assistant 消息
        # 其 content 字段必须为 null 而非空字符串，否则会触发 400 Bad Request。
        content: Any = self.content if self.content else None
        d: Dict[str, Any] = {"role": self.role, "content": content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": _to_dict(tc.args),
                    },
                }
                for tc in self.tool_calls
            ]
        return d


class AIMessageChunk(AIMessage):
    """
    流式 LLM 输出的片段（chunk）。

    可通过 + 运算符将多个 chunk 累积为一个完整消息。
    """
    role: str = "assistant"

    def __add__(self, other: "AIMessageChunk") -> "AIMessageChunk":
        if not isinstance(other, AIMessageChunk):
            raise TypeError(f"Cannot add AIMessageChunk and {type(other)}")
        merged_content = self.content + other.content
        merged_tool_calls = list(self.tool_calls)
        # 合并工具调用：按 id 去重，同 id 的 args 字符串拼接
        existing_ids = {tc.id: tc for tc in merged_tool_calls}
        for tc in other.tool_calls:
            if tc.id in existing_ids:
                # args 在流式场景下可能是分片 JSON 字符串，直接追加内容
                existing = existing_ids[tc.id]
                if isinstance(existing.args, str) and isinstance(tc.args, str):
                    existing.args = existing.args + tc.args  # type: ignore[assignment]
                else:
                    existing.args.update(tc.args)
            else:
                merged_tool_calls.append(tc)
                existing_ids[tc.id] = tc
        return AIMessageChunk(content=merged_content, tool_calls=merged_tool_calls)


class ToolMessage(BaseMessage):
    """
    工具执行结果消息（role="tool"）。

    tool_call_id 对应触发该工具调用的 ToolCall.id。
    """
    role: str = "tool"

    def __init__(self, content: str, tool_call_id: str = "", **kwargs):
        super().__init__(content, **kwargs)
        self.tool_call_id: str = tool_call_id

    def to_openai_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _dumps(obj: Any) -> str:
    """将 Python 对象序列化为 JSON 字符串（用于工具参数）。"""
    import json
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False)


def _to_dict(obj: Any) -> Any:
    """确保工具参数是 dict 对象。

    Ollama 的 /api/chat 接口要求历史消息中 tool_calls 的 arguments 字段
    必须是 dict 对象（而非 JSON 字符串），否则会返回 400 Bad Request。
    """
    import json
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            return {"raw": obj}
    return obj


def messages_to_openai(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """将消息列表转为 OpenAI Chat Completions 请求格式。"""
    return [m.to_openai_dict() for m in messages]
