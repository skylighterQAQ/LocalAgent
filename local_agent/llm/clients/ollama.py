"""
local_agent.engine.llm.ollama
==============================
Ollama LLM HTTP 客户端，直接调用 Ollama REST API。

API 文档：https://github.com/ollama/ollama/blob/main/docs/api.md

端点：
  POST /api/chat       → 非流式 / 流式对话（支持 tools）
  GET  /api/tags       → 列出本地模型
"""
from __future__ import annotations

import json
import logging
import time
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


def _build_tools_param(tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """将工具列表转换为 Ollama 支持的 OpenAI function-calling 格式。"""
    return [t.get_schema() for t in tools]


def _parse_tool_calls(raw_calls: List[Dict[str, Any]]) -> List[ToolCall]:
    """解析 Ollama 响应中的 tool_calls 数组。"""
    result: List[ToolCall] = []
    for tc in raw_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        args_raw = fn.get("arguments", {})
        # arguments 可能是 dict 或 JSON string
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


# 这些模型会默认开启 thinking/reasoning 模式，在工具调用场景下会导致 content 为空
# 需要在 payload 中设置 think=False 来禁用 thinking 模式
_THINKING_MODELS = (
    "qwen3",
    "qwq",
    "deepseek-r1",
    "deepseek-r2",
)


def _is_thinking_model(model_name: str) -> bool:
    """判断是否为带 thinking 模式的模型。"""
    name_lower = model_name.lower()
    return any(prefix in name_lower for prefix in _THINKING_MODELS)


class OllamaLLM(BaseLLM):
    """
    直接通过 HTTP 访问 Ollama /api/chat 的 LLM 客户端。

    Args:
        model    : 模型名称，如 "qwen2.5:7b"
        base_url : Ollama 服务地址，默认 http://localhost:11434
        temperature : 采样温度
        timeout  : HTTP 请求超时（秒）
        disable_thinking : 对 qwen3/qwq/deepseek-r1 等 thinking 模型，
                           在工具调用时禁用 thinking 模式（默认自动检测）。
                           设为 True 强制禁用，False 保持模型默认行为。
                           None（默认）= 自动检测：若为 thinking 模型则禁用。
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        timeout: float = 120.0,
        disable_thinking: Optional[bool] = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        # 自动检测：若为 thinking 模型则禁用 thinking（避免工具调用时 content 为空）
        if disable_thinking is None:
            self.disable_thinking = _is_thinking_model(model)
        else:
            self.disable_thinking = disable_thinking
        if self.disable_thinking:
            logger.info(
                "Thinking mode disabled for model '%s' (tool-calling compatibility)",
                model,
            )

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def invoke(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """非流式调用，返回完整 AIMessage。

        使用分离的超时设置：
          - connect: 10 秒（连接建立超时）
          - read: self.timeout（等待模型生成完整响应的超时）
          - write: 30 秒（发送请求超时）
          - pool: 10 秒（连接池等待超时）

        这样可以避免 httpx 默认的总超时（同时限制 connect + read）
        在模型生成长文本时误触发 ReadTimeout。
        """
        payload = self._build_payload(messages, tools, stream=False, **kwargs)
        # 使用分离超时：connect/write 用较短值，read 用完整 timeout
        timeout = httpx.Timeout(
            connect=10.0,
            read=self.timeout,
            write=30.0,
            pool=10.0,
        )
        t_start = time.time()
        logger.info(
            "[LLM invoke] model=%s messages=%d timeout=%.0fs",
            self.model, len(messages), self.timeout,
        )
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
        except httpx.ReadTimeout as exc:
            elapsed = time.time() - t_start
            logger.error(
                "[LLM invoke] ReadTimeout after %.1fs (timeout=%.0fs) model=%s",
                elapsed, self.timeout, self.model,
            )
            raise RuntimeError(
                f"Ollama API error: timed out waiting for model response "
                f"(read_timeout={self.timeout}s). "
                f"Consider increasing 'ollama.timeout' in config.yaml, "
                f"or switching to a faster/smaller model."
            ) from exc
        except httpx.HTTPError as exc:
            elapsed = time.time() - t_start
            logger.error(
                "[LLM invoke] HTTPError after %.1fs model=%s: %s",
                elapsed, self.model, exc,
            )
            raise RuntimeError(f"Ollama API error: {exc}") from exc

        elapsed = time.time() - t_start
        result = self._parse_response(resp.json(), messages)
        logger.info(
            "[LLM invoke] done model=%s elapsed=%.1fs tool_calls=%d content_len=%d",
            self.model, elapsed,
            len(result.tool_calls) if result.tool_calls else 0,
            len(result.content) if result.content else 0,
        )
        return result

    def stream(
        self,
        messages: List[BaseMessage],
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        """流式调用，逐 token yield AIMessageChunk。

        流式场景使用分离超时，read 超时适当放宽（流式响应是持续写入的，
        每两个 token 之间的间隔不会触发整体超时）。
        
        注意：httpx streaming 的 read timeout 表示读取单个 chunk 的最长等待时间，
        不是整体响应的总超时，因此只要模型持续输出 tokens，就不会触发超时。
        真正触发超时的情况是两个 token 之间的间隔超过 read timeout。
        """
        payload = self._build_payload(messages, tools, stream=True, **kwargs)
        timeout = httpx.Timeout(
            connect=10.0,
            read=self.timeout,
            write=30.0,
            pool=10.0,
        )
        t_start = time.time()
        chunk_count = 0
        logger.info(
            "[LLM stream] model=%s messages=%d timeout=%.0fs (per-chunk)",
            self.model, len(messages), self.timeout,
        )
        try:
            with httpx.Client(timeout=timeout) as client:
                with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        chunk = self._parse_stream_chunk(data)
                        if chunk is not None:
                            chunk_count += 1
                            yield chunk
                        if data.get("done"):
                            break
        except httpx.ReadTimeout as exc:
            elapsed = time.time() - t_start
            logger.error(
                "[LLM stream] ReadTimeout after %.1fs (timeout=%.0fs) chunks=%d model=%s",
                elapsed, self.timeout, chunk_count, self.model,
            )
            raise RuntimeError(
                f"Ollama streaming error: timed out waiting for model response "
                f"(read_timeout={self.timeout}s). "
                f"Consider increasing 'ollama.timeout' in config.yaml, "
                f"or switching to a faster/smaller model."
            ) from exc
        except httpx.HTTPError as exc:
            elapsed = time.time() - t_start
            logger.error(
                "[LLM stream] HTTPError after %.1fs chunks=%d model=%s: %s",
                elapsed, chunk_count, self.model, exc,
            )
            raise RuntimeError(f"Ollama streaming error: {exc}") from exc

        elapsed = time.time() - t_start
        logger.info(
            "[LLM stream] done model=%s elapsed=%.1fs chunks=%d",
            self.model, elapsed, chunk_count,
        )

    # ------------------------------------------------------------------
    # 辅助工具
    # ------------------------------------------------------------------

    def check_connection(self) -> bool:
        """检查 Ollama 是否可达。"""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """列出 Ollama 本地可用模型。"""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=10)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            pass
        return []

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

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
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
            },
        }
        # 禁用 thinking 模式（qwen3/qwq/deepseek-r1 等模型在工具调用时
        # thinking 内容会使 content 字段为空，导致 Agent 误判为无输出）
        if self.disable_thinking:
            payload["think"] = False
        if tools:
            payload["tools"] = _build_tools_param(tools)
            logger.debug(
                "Sending %d tool(s) to Ollama: %s",
                len(tools),
                [t.name for t in tools],
            )
        else:
            logger.debug("No tools bound – sending plain chat request to Ollama")
        return payload

    def _parse_response(
        self, 
        data: Dict[str, Any], 
        messages: Optional[List[Any]] = None
    ) -> AIMessage:
        """解析非流式响应体。

        对于 thinking 模型（qwen3 等），若 content 为空但 thinking 字段有内容，
        则将 thinking 作为 content 返回（确保 Agent 不会误判为空响应）。
        
        如果 content 和 thinking 都为空且没有工具调用，尝试从消息历史中提取
        工具结果并生成有意义的输出，而不是返回通用的保底消息。
        
        Args:
            data: Ollama API 返回的响应数据
            messages: 消息历史（可选），用于在空响应时提取工具结果
        """
        msg = data.get("message", {})
        content: str = msg.get("content", "")
        # Fallback: 若 content 为空，使用 thinking 字段内容
        if not content:
            content = msg.get("thinking", "") or ""
        raw_tool_calls: List[Dict[str, Any]] = msg.get("tool_calls") or []
        tool_calls = _parse_tool_calls(raw_tool_calls)
        
        # 保底：如果 content 和 tool_calls 都为空，尝试从工具结果中生成输出
        if not content and not tool_calls:
            logger.warning(
                "Model returned empty response (no content, no thinking, no tool_calls). "
                "Attempting to synthesize response from tool results."
            )
            
            # 尝试从消息历史中提取工具结果
            synthesized_content = self._synthesize_from_tool_results(messages)
            if synthesized_content:
                content = synthesized_content
                logger.info("Successfully synthesized response from %d tool results", 
                           synthesized_content.count("【工具结果"))
            else:
                # 如果没有工具结果可用，使用通用保底消息
                content = "根据以上工具调用的结果，这是我找到的信息。"
                logger.warning("No tool results found in history, using generic fallback message")
        
        return AIMessage(content=content, tool_calls=tool_calls)
    
    def _synthesize_from_tool_results(self, messages: Optional[List[Any]]) -> str:
        """从消息历史中的工具结果合成有意义的输出。
        
        Args:
            messages: 消息历史列表
            
        Returns:
            合成的输出文本，如果没有工具结果则返回空字符串
        """
        if not messages:
            return ""
        
        # 导入 ToolMessage 类型（避免循环导入）
        from local_agent.core.messages import ToolMessage
        
        # 提取所有 ToolMessage
        tool_results = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_results.append(msg.content)
        
        if not tool_results:
            return ""
        
        # 构建结构化输出
        output_lines = ["根据工具调用的结果，我为您整理了以下信息：\n"]
        
        for idx, result in enumerate(tool_results, 1):
            # 截断过长的结果（保留前1000字符）
            truncated_result = result[:1000] + "..." if len(result) > 1000 else result
            output_lines.append(f"【工具结果 {idx}】")
            output_lines.append(truncated_result)
            output_lines.append("")  # 空行分隔
        
        return "\n".join(output_lines)

    def _parse_stream_chunk(self, data: Dict[str, Any]) -> Optional[AIMessageChunk]:
        """解析流式响应的单行 JSON。

        对于 thinking 模型（qwen3 等），streaming 时分为两个阶段：
          阶段 1：若干 thinking chunks：{"message": {"content": "", "thinking": "..."}}
          阶段 2：若干 content chunks：{"message": {"content": "...", "thinking": ""}}

        旧逻辑在阶段 1 直接返回 None，若模型只输出 thinking 阶段（如工具调用场景），
        accumulated_chunks 为空 → AIMessage.content="" → 系统链路误判为空响应。

        新逻辑：当 content 为空但 thinking 有内容时，将 thinking 内容作为 content
        累积进 AIMessageChunk，确保最终合并结果不为空。
        """
        msg = data.get("message", {})
        if msg is None:
            return None
        content: str = msg.get("content", "")
        thinking: str = msg.get("thinking", "") or ""
        raw_tool_calls: List[Dict[str, Any]] = msg.get("tool_calls") or []
        tool_calls = _parse_tool_calls(raw_tool_calls)
        # Fallback：content 为空但 thinking 有内容时，用 thinking 作为内容
        if not content and thinking:
            content = thinking
        if not content and not tool_calls:
            return None
        return AIMessageChunk(content=content, tool_calls=tool_calls)
