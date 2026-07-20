"""
local_agent.core.prompt_registry
=================================
基于调用 ID 的 Prompt 上下文注册表。

设计目标：
  当 ReActEngine 在 invoke_skill 工具调用前后需要记录父 prompt 上下文时，
  通过唯一的 invocation_id（UUID）将快照存入注册表。子 skill 执行完毕后，
  用相同的 invocation_id 从注册表取回并清除。

为什么不用堆栈（LIFO）？
  - 堆栈依赖调用顺序，多线程/异步场景下不同请求的 push/pop 可能交叉干扰
  - 调用 ID 是唯一标识，并发执行的多个 skill 调用各自持有独立 UUID，
    注册表以 ID 为 key 存取，绝对不会因并发而关联出错
  - 调用完成后 ID 对应的数据立即清除，无内存泄漏

使用示例::

    from local_agent.core.prompt_registry import (
        generate_invocation_id,
        save_prompt_context,
        retrieve_prompt_context,
    )

    # 调用子 skill 前
    inv_id = generate_invocation_id()
    save_prompt_context(inv_id, current_messages, skill_name="url_accessor", task="...")

    # 子 skill 执行（期间 current_messages 不被修改）
    result = skill_tool._run(...)

    # 子 skill 执行完成后，用 ID 取回父上下文并清除
    entry = retrieve_prompt_context(inv_id)
    if entry:
        # entry.saved_messages 是调用前的消息快照
        pass
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PromptContextEntry:
    """
    一次子 skill 调用的上下文快照条目。

    Attributes:
        invocation_id:  唯一调用 ID（UUID4 字符串）
        skill_name:     被调用的子 skill 名称
        task:           子 skill 的任务描述
        saved_messages: 调用前父 prompt 的消息列表快照（深拷贝）
        timestamp:      条目创建的时间戳（Unix 时间，单位秒）
    """
    invocation_id: str
    skill_name: str
    task: str
    saved_messages: List[Any]  # List[BaseMessage] 的类型擦除版本，避免循环导入
    timestamp: float = field(default_factory=time.time)


class PromptContextRegistry:
    """
    基于调用 ID 的 prompt 上下文注册表。

    线程安全：内部使用 threading.Lock 保护字典读写，可安全用于多线程环境。

    典型生命周期：
      1. save(inv_id, messages, ...)   — 子 skill 调用前存入
      2. retrieve(inv_id)              — 子 skill 返回后取出并删除
      3. （可选）peek(inv_id)          — 仅查看不删除（用于 debug）
    """

    def __init__(self) -> None:
        self._store: Dict[str, PromptContextEntry] = {}
        self._lock = threading.Lock()

    def save(
        self,
        invocation_id: str,
        messages: List[Any],
        skill_name: str,
        task: str,
    ) -> None:
        """
        保存父 prompt 上下文，与调用 ID 绑定。

        Args:
            invocation_id: 唯一调用 ID（由 generate_invocation_id() 生成）
            messages:      当前消息列表（会被深拷贝，不受后续修改影响）
            skill_name:    将要调用的子 skill 名称
            task:          子 skill 的任务描述
        """
        entry = PromptContextEntry(
            invocation_id=invocation_id,
            skill_name=skill_name,
            task=task,
            saved_messages=list(messages),  # 快照：浅拷贝列表，消息对象本身不可变
        )
        with self._lock:
            self._store[invocation_id] = entry

    def retrieve(self, invocation_id: str) -> Optional[PromptContextEntry]:
        """
        用 ID 取出条目并从注册表删除（调用完成后清理）。

        Args:
            invocation_id: 之前传给 save() 的调用 ID

        Returns:
            对应的 PromptContextEntry，若 ID 不存在则返回 None
        """
        with self._lock:
            return self._store.pop(invocation_id, None)

    def peek(self, invocation_id: str) -> Optional[PromptContextEntry]:
        """
        用 ID 查看条目但不删除（用于 debug 或断言检查）。

        Args:
            invocation_id: 调用 ID

        Returns:
            对应的 PromptContextEntry，若不存在则返回 None
        """
        with self._lock:
            return self._store.get(invocation_id)

    def list_all(self) -> List[PromptContextEntry]:
        """返回注册表中所有条目的列表副本（用于 debug 打印）。"""
        with self._lock:
            return list(self._store.values())

    def size(self) -> int:
        """返回注册表当前条目数量。"""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """清空所有条目（用于测试或异常恢复）。"""
        with self._lock:
            self._store.clear()


# ── 进程级全局单例（线程安全）──────────────────────────────────────────────────
_registry = PromptContextRegistry()


# ── 模块级便捷函数 ──────────────────────────────────────────────────────────────

def get_registry() -> PromptContextRegistry:
    """获取全局 PromptContextRegistry 单例。"""
    return _registry


def generate_invocation_id() -> str:
    """
    生成一个唯一的调用 ID（UUID4 字符串）。

    每次调用子 skill 前调用此函数，获得唯一 ID，后续用此 ID 存取注册表。

    Returns:
        UUID4 格式的字符串，如 '550e8400-e29b-41d4-a716-446655440000'
    """
    return str(uuid.uuid4())


def save_prompt_context(
    invocation_id: str,
    messages: List[Any],
    skill_name: str,
    task: str,
) -> None:
    """
    保存父 prompt 上下文到全局注册表。

    Args:
        invocation_id: 唯一调用 ID（由 generate_invocation_id() 生成）
        messages:      当前消息列表快照
        skill_name:    将要调用的子 skill 名称
        task:          子 skill 的任务描述
    """
    _registry.save(invocation_id, messages, skill_name, task)


def retrieve_prompt_context(invocation_id: str) -> Optional[PromptContextEntry]:
    """
    从全局注册表取出并删除对应条目。

    Args:
        invocation_id: 之前传给 save_prompt_context() 的调用 ID

    Returns:
        对应的 PromptContextEntry，若 ID 不存在则返回 None
    """
    return _registry.retrieve(invocation_id)
