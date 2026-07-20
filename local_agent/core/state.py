"""
Agent State Definition
======================
定义贯穿 ReAct 循环的状态字典，替换原 LangGraph TypedDict。

改动说明：
  - 移除对 langchain_core.messages.BaseMessage 的依赖
  - 使用本地 local_agent.engine.messages.BaseMessage
  - 保持字段名称、类型完全不变（上层代码零改动）
"""
import operator
from typing import TypedDict, Annotated, List, Optional, Dict, Any

from local_agent.core.messages import BaseMessage


class AgentState(TypedDict):
    """主 Agent 状态字典，在 ReActEngine 调用前后传递。"""
    messages: Annotated[List[BaseMessage], operator.add]
    active_skill: Optional[str]
    tool_results: List[Dict[str, Any]]
    memory_context: str
    user_id: str
    session_id: str
    iteration_count: int
    error: Optional[str]
