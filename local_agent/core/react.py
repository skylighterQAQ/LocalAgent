"""
local_agent.engine.react
=========================
自主实现的 ReAct (Reason + Act) 循环引擎，完整替换 LangGraph。

ReAct 循环流程：
  START
    ↓
  llm.invoke(messages)
    ├─ 有 tool_calls  → 执行每个工具 → 追加 ToolMessage → 回 llm.invoke
    └─ 无 tool_calls  → （可选）追加引导消息重试 → 若还是无 → 返回最终 AIMessage
  END

与原 LangGraph 接口的兼容性：
  - invoke()  返回 {"messages": [...]} 与原 graph.invoke() 格式一致
  - stream()  产生 ("messages", (chunk, metadata)) 和 ("values", {...}) 事件元组，
              与原 graph.stream(stream_mode=["messages","values"]) 格式一致

工具调用引导重试（tool_call_retry）：
  当本地弱模型没有调用工具时，自动追加一条 HumanMessage，提示模型
  "请调用工具完成任务"，然后再次请求 LLM。最多重试 max_tool_retry 次。

使用示例::

    engine = ReActEngine(
        llm=OllamaLLM(model="qwen2.5:7b"),
        tools=my_tools,
        system_prompt="你是一个有用的助手。",
        max_iterations=20,
        tool_call_retry=True,
        max_tool_retry=2,
    )
    result = engine.invoke(messages)            # 同步
    for event in engine.stream(messages):       # 流式
        mode, data = event
        ...
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from local_agent.llm.base import BaseLLM
from local_agent.core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from local_agent.core.tools import BaseTool

logger = logging.getLogger(__name__)


class ToolExecutionError(RuntimeError):
    """Raised when a tool fails after exhausting all retries."""

# 触发引导重试的关键词：模型在回复中出现这些词时，很可能本应调用工具
_SHOULD_HAVE_USED_TOOL_HINTS = [
    "i cannot", "i can't", "i don't have", "i'm unable",
    "无法", "不能", "我没有", "我无法", "我不能",
    "unable to", "cannot access", "don't have access",
    "you would need to", "you should", "please run",
    "请运行", "请执行", "需要运行",
]

# 第一次引导消息（提示模型调用工具）
_TOOL_CALL_GUIDANCE_MSG = (
    "You need to use the available tools to complete this task. "
    "Please call the appropriate tool now — do not just describe what you would do. "
    "Look at the tool list and call the right tool directly."
)

# 第二次及以后的引导消息（更强硬的提示）
_TOOL_CALL_RETRY_MSG = (
    "IMPORTANT: You must call a tool. "
    "You have not called any tool yet, but this task requires tool usage. "
    "Pick the most relevant tool from the list and call it immediately."
)

# 已有工具结果时的上下文感知引导消息（搜索后需继续读取URL等）
# 注意：不包含具体 URL 占位符示例，避免弱模型按示例编造 URL
_TOOL_CALL_WITH_RESULTS_MSG = (
    "You have received tool results above. "
    "Based on these results, you MUST now call the next tool to continue the task. "
    "Do NOT output a text summary yet — call a tool immediately. "
    "IMPORTANT: Only use URLs that actually appear in the tool results shown above. "
    "Do NOT invent, guess, or use any URL that is not explicitly listed in the results."
)

# 编码任务中工具调用后、任务未完成时的引导消息
_TOOL_CALL_CODING_CONTINUE_MSG = (
    "You have completed some steps. "
    "Please continue the task by calling the next appropriate tool. "
    "Review what has been done so far and determine what files, "
    "directories, or actions still need to be completed to finish the task. "
    "Call the next tool now — do NOT stop until the task is fully complete."
)


import re as _re



def _is_relevant_url(url: str) -> bool:
    """
    判断 URL 是否与用户研究任务相关，过滤掉明显无关的域名。

    排除：
      - 通用视频/社交媒体平台（youtube.com, bilibili.com, twitter.com 等）
      - 地图平台（map.baidu.com, maps.google.com 等）
      - 搜索引擎首页（google.com, bing.com, baidu.com 等）
      - 广告/营销域名
    """
    _IRRELEVANT_PATTERNS = [
        # 视频/社交媒体
        r'youtube\.com/?$',
        r'youtu\.be/?$',
        r'bilibili\.com/?$',
        r'twitter\.com/?$',
        r'facebook\.com/?$',
        r'instagram\.com/?$',
        r'tiktok\.com/?$',
        r'douyin\.com/?$',
        # 地图
        r'map\.baidu\.com',
        r'maps\.google\.',
        r'maps\.apple\.com',
        r'ditu\.baidu\.com',
        # 搜索引擎首页
        r'^https?://(?:www\.)?google\.com/?$',
        r'^https?://(?:www\.)?bing\.com/?$',
        r'^https?://(?:www\.)?baidu\.com/?$',
        # 社交/论坛首页（无具体内容路径）
        r'^https?://(?:www\.)?reddit\.com/?$',
        r'^https?://(?:www\.)?weibo\.com/?$',
    ]
    for pattern in _IRRELEVANT_PATTERNS:
        if _re.search(pattern, url, _re.IGNORECASE):
            return False
    return True


def _extract_urls_from_messages(messages: "List[BaseMessage]") -> "List[str]":
    """
    从历史 ToolMessage 中提取搜索结果的 URL，用于填入引导消息。
    只提取来自 search_web 工具的 ToolMessage 中的 http/https 绝对 URL，最多返回 5 个。

    只处理 search_web 工具的返回内容，跳过 invoke_skill / url_accessor / fetch_url 等
    工具的返回内容，避免将已访问 URL 重复加入"待访问列表"，从而导致
    _has_unvisited_urls 永远返回 True 的死循环问题。
    """
    # 1. 建立 tool_call_id -> tool_name 映射，用于判断每条 ToolMessage 来自哪个工具
    call_id_to_tool: "dict[str, str]" = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.id:
                    call_id_to_tool[tc.id] = tc.name

    url_pattern = _re.compile(r'https?://[^\s,"\'>\)\]\|]+')
    seen: "set[str]" = set()
    urls: "List[str]" = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # 只处理 search_web 工具的结果，跳过 invoke_skill / fetch_url 等
            tool_name = call_id_to_tool.get(msg.tool_call_id or "", "")
            if tool_name != "search_web":
                continue
            for url in url_pattern.findall(msg.content or ""):
                # 去掉末尾的标点符号
                url = url.rstrip('.,;)')
                # 过滤无关 URL（如 youtube.com、map.baidu.com 等）
                if not _is_relevant_url(url):
                    continue
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
                    if len(urls) >= 5:
                        return urls
    return urls


def _has_tool_results_with_real_content(messages: "List[BaseMessage]") -> bool:
    """
    检测历史 ToolMessage 中是否有来自 url_accessor 的实质性内容。

    url_accessor 失败时结果格式为:
      "[Result from url_accessor skill]\n{原始URL}"
    即只回显了输入 URL，没有实际页面内容。

    如果所有 url_accessor 结果都是这种"只回显 URL"的形式，
    说明 URL 无法访问，应让 LLM 直接总结而不是继续 retry。
    """
    url_accessor_results = [
        m.content
        for m in messages
        if isinstance(m, ToolMessage)
        and isinstance(m.content, str)
        and "[Result from url_accessor skill]" in m.content
    ]
    if not url_accessor_results:
        return True  # 没有 url_accessor 结果，不需要特殊处理

    # 如果有至少一个结果包含实质内容（超过纯 URL 回显），认为有内容
    for content in url_accessor_results:
        lines = [l.strip() for l in content.splitlines() if l.strip()]
        # 去掉 "[Result from url_accessor skill]" 那一行
        non_header_lines = [l for l in lines if l != "[Result from url_accessor skill]"]
        # 如果剩余内容超过 3 行，认为有实质内容
        if len(non_header_lines) > 3:
            return True
    return False  # 所有 url_accessor 结果都是空/只回显 URL


def _build_context_aware_guidance(messages: "List[BaseMessage]", retry_count: int) -> str:
    """
    根据对话历史构建上下文感知的引导消息。

    - 若历史中 url_accessor 结果都是"无内容/只回显 URL"，
      说明 URL 无法访问，应让 LLM 直接总结，而不是继续强推 retry。
    - 若搜索结果中有尚未访问的 URL，明确列出并要求 LLM 逐个访问（不提前总结）。
    - 若有 ToolMessage 但无法提取 URL，使用通用提示。
    - 否则返回通用引导或强硬引导。
    """
    has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
    if has_tool_results:
        # 检查 url_accessor 是否已全部失败（只回显 URL，无内容）
        if not _has_tool_results_with_real_content(messages):
            return (
                "The URLs you tried to access did not return useful content "
                "(they appear to be redirect wrappers that cannot be opened). "
                "You already have the search result titles from the earlier search. "
                "Please summarize the search results you have and provide a helpful answer "
                "to the user's original question. Do NOT attempt to access more URLs."
            )

        all_urls = _extract_urls_from_messages(messages)
        if all_urls:
            # 计算尚未访问的 URL
            visited = _get_visited_urls(messages)
            unvisited = [u for u in all_urls if u not in visited]

            if unvisited:
                url_list = "\n".join(f"  - {u}" for u in unvisited)
                return (
                    "You have processed some URLs but there are still unvisited URLs "
                    "from the search results. "
                    "You MUST visit ALL of the following URLs before writing a final summary:\n"
                    f"{url_list}\n"
                    "Call invoke_skill with skill_name='url_accessor' for the NEXT unvisited URL. "
                    "Do NOT write a final summary yet — visit all URLs first."
                )
            else:
                # 所有 URL 已访问，可以总结了
                return (
                    "You have visited all available URLs from the search results. "
                    "Now please summarize all the information you have collected and "
                    "provide a complete, helpful answer to the user's original question."
                )

        # 没有搜索 URL → 编码/文件操作任务，引导模型继续完成任务
        return _TOOL_CALL_CODING_CONTINUE_MSG
    return _TOOL_CALL_RETRY_MSG if retry_count > 1 else _TOOL_CALL_GUIDANCE_MSG


def _get_visited_urls(messages: "List[BaseMessage]") -> "set[str]":
    """
    从历史消息中获取已经被 invoke_skill(url_accessor) 或 fetch_url 访问过的 URL。

    遍历所有 AIMessage 的 tool_calls，提取：
    - invoke_skill 调用中的 url 参数
    - invoke_skill 调用中 task 字符串里嵌入的 URL
    - fetch_url 调用中的 url 参数

    Returns:
        已访问 URL 的集合（均为绝对 http/https URL）
    """
    visited: "set[str]" = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.name == "invoke_skill":
                    # 从 url 参数提取
                    url_arg = tc.args.get("url", "")
                    if url_arg and url_arg.startswith("http"):
                        visited.add(url_arg.rstrip(".,;)"))
                    # 从 task 字符串中提取嵌入的 URL
                    task_arg = tc.args.get("task", "")
                    if task_arg:
                        match = _re.search(r'https?://[^\s,\'"，。）\]）]+', task_arg)
                        if match:
                            visited.add(match.group(0).rstrip(".,;)"))
                elif tc.name == "fetch_url":
                    url_arg = tc.args.get("url", "")
                    if url_arg and url_arg.startswith("http"):
                        visited.add(url_arg.rstrip(".,;)"))
    return visited


def _has_unvisited_urls(messages: "List[BaseMessage]") -> bool:
    """
    检测消息历史中是否存在从搜索结果获取的、尚未被 invoke_skill(url_accessor) 访问过的 URL。

    用于判断 LLM 在访问了部分 URL 后是否还有剩余 URL 需要继续访问，
    防止模型在访问第一个 URL 获得内容后就立即输出最终答案而跳过其余 URL。

    Returns:
        True  → 还有未访问的 URL，需要继续引导模型调用工具
        False → 所有已知 URL 都已访问，或根本没有 URL
    """
    all_urls = _extract_urls_from_messages(messages)
    if not all_urls:
        return False
    visited = _get_visited_urls(messages)
    unvisited = [u for u in all_urls if u not in visited]
    return len(unvisited) > 0


def _extract_user_query(messages: "List[BaseMessage]") -> str:
    """
    从消息历史中提取第一条用户query（第一条 HumanMessage 的内容，跳过引导重试消息）。

    引导重试消息的特征：
      - 以 "You have processed" 开头
      - 以 "You need to use" 开头
      - 以 "IMPORTANT: You must" 开头
      - 以 "The URLs you tried" 开头
      - 以 "You have visited" 开头

    Returns:
        用户原始 query 字符串，找不到时返回空字符串
    """
    _GUIDANCE_PREFIXES = (
        "You have processed",
        "You need to use",
        "IMPORTANT: You must",
        "The URLs you tried",
        "You have visited",
        "You have completed some steps",
    )
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content or ""
            if not content.startswith(_GUIDANCE_PREFIXES):
                return content
    return ""


def _collect_tool_results(messages: "List[BaseMessage]") -> str:
    """
    收集所有 ToolMessage 的内容，组合成一段文字，供总结阶段传入 LLM。

    只收集 success 状态的工具结果（失败结果也收集，标注状态），
    去掉无实质内容的重复消息。

    Returns:
        组合后的工具结果字符串
    """
    parts: "List[str]" = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            if content.strip():
                parts.append(content.strip())
    if not parts:
        return ""
    return "\n\n---\n\n".join(parts)


def _get_last_retry_guidance(messages: "List[BaseMessage]") -> str:
    """
    获取最后一条引导重试 HumanMessage 的内容（如果有的话）。

    引导重试消息的特征与 _extract_user_query 中定义的相同。
    """
    _GUIDANCE_PREFIXES = (
        "You have processed",
        "You need to use",
        "IMPORTANT: You must",
        "The URLs you tried",
        "You have visited",
        "You have completed some steps",
    )
    last_guidance = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = msg.content or ""
            if content.startswith(_GUIDANCE_PREFIXES):
                last_guidance = content
    return last_guidance


def _build_minimal_messages_for_llm(
    messages: "List[BaseMessage]",
    is_summary_phase: bool = False,
) -> "List[BaseMessage]":
    """
    为 LLM 调用构建最精简的消息列表，减少冗余上下文。

    两种模式：
    1. 工具调用阶段（is_summary_phase=False）：
       - [SystemMessage] + [HumanMessage: 用户query] + [最近一条ToolMessage: 最新工具结果] + [可选: HumanMessage: 最新retry引导]
       - 目的：让 LLM 知道上一步工具的执行结果，从而决定下一步行动

    2. 总结分析阶段（is_summary_phase=True）：
       - [SystemMessage] + [HumanMessage: 用户query + 所有工具结果]
       - 目的：给 LLM 所有已收集的信息用于生成最终答案

    Args:
        messages:         当前完整消息列表（含 SystemMessage）
        is_summary_phase: True 表示所有工具调用已完成，进入总结阶段

    Returns:
        精简后的消息列表
    """
    if not messages:
        return messages

    # 提取 SystemMessage（第一条）
    system_msg: Optional[BaseMessage] = None
    if messages and isinstance(messages[0], SystemMessage):
        system_msg = messages[0]

    # 提取用户原始 query
    user_query = _extract_user_query(messages)

    if is_summary_phase:
        # ── 总结阶段：system + (query + 所有工具结果) ────────────────────────
        tool_results = _collect_tool_results(messages)
        if tool_results and user_query:
            combined = f"{user_query}\n\n[已收集的工具调用结果]\n{tool_results}"
        elif tool_results:
            combined = f"[已收集的工具调用结果]\n{tool_results}"
        else:
            combined = user_query or "请根据以上信息回答。"

        result: List[BaseMessage] = []
        if system_msg:
            result.append(system_msg)
        result.append(HumanMessage(content=combined))
        logger.debug(
            "_build_minimal_messages_for_llm: summary_phase, %d chars combined",
            len(combined),
        )
        return result
    else:
        # ── 工具调用阶段：system + query + (可选) 最近工具结果 + (可选) 最新retry引导 ──
        # URL 访问阶段（已有 invoke_skill ToolMessage 且还有未访问 URL）：
        #   不传入上一次的完整 ToolMessage，避免"第 N 次调用包含前 N-1 次结果"的污染。
        #   retry guidance 消息已包含剩余 URL 列表，LLM 知道下一步要访问哪个 URL，
        #   无需再看到上一次的完整页面内容。
        # 非 URL 访问阶段（普通工具调用循环）：传入最近一条 ToolMessage 让模型了解上一步结果。
        last_guidance = _get_last_retry_guidance(messages)

        # 检测是否处于 URL 访问阶段
        # 条件：已经发起过至少一次 invoke_skill 调用（有已访问 URL），且还有未访问 URL
        # 注意：不能只靠 _has_unvisited_urls 来判断——搜索刚完成时 invoke_skill 尚未调用，
        #       但搜索结果中有 URL，此时若屏蔽 ToolMessage 会导致 LLM 看不到搜索结果，
        #       反复触发重复搜索的死循环。
        visited_urls = _get_visited_urls(messages)
        has_invoke_skill_result = len(visited_urls) > 0
        is_url_visiting_phase = has_invoke_skill_result and _has_unvisited_urls(messages)

        # 提取最近一条 ToolMessage（仅在非 URL 访问阶段使用）
        last_tool_msg: Optional[BaseMessage] = None
        if not is_url_visiting_phase:
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    last_tool_msg = msg
                    break

        result2: List[BaseMessage] = []
        if system_msg:
            result2.append(system_msg)
        if last_tool_msg is not None:
            # ── 已有工具结果：传 user_query + 最新工具结果 ──────────────────────
            # 对于编码/文件写入任务，模型需要同时看到"任务目标"（user_query）和
            # "最新执行结果"（last_tool_msg），才能判断任务是否完成以及下一步应该做什么。
            # 如果只传 ToolMessage 而不传 user_query，弱模型可能只看到"文件写入成功"
            # 就不知道任务是什么，导致继续执行无关操作或重复写文件。
            # 注意：SystemMessage 虽然包含步骤指令，但某些弱模型在多轮工具调用后
            # 倾向于只关注最近几条消息，所以同时传入 user_query 可以强化任务意识。
            if user_query:
                result2.append(HumanMessage(content=user_query))
            result2.append(last_tool_msg)
        else:
            # ── 无工具结果：这是首轮调用，需要传 user_query 让模型知道任务 ────
            if user_query:
                result2.append(HumanMessage(content=user_query))
        if last_guidance:
            result2.append(HumanMessage(content=last_guidance))

        logger.debug(
            "_build_minimal_messages_for_llm: tool_phase, result has %d messages "
            "(last_tool_msg=%s, is_url_visiting_phase=%s)",
            len(result2),
            last_tool_msg is not None,
            is_url_visiting_phase,
        )
        return result2


def _compress_invoke_skill_history(messages: "List[BaseMessage]") -> "List[BaseMessage]":
    """
    在每次 LLM 调用前，将历史消息中"非最新"的 invoke_skill ToolMessage 压缩为摘要。

    背景：
      当父 Agent（AGENT 步骤的子 ReActEngine）通过 invoke_skill 多次访问不同 URL 时，
      历史消息会不断积累每次 url_accessor 的完整输出（可能数千字），导致：
        1. 每次 LLM 调用输入越来越长
        2. 模型输入中包含大量与当前 URL 无关的历史内容

    策略：
      - 找出所有 invoke_skill 相关的 ToolMessage（通过 tool_call_id 匹配 AIMessage.tool_calls）
      - 只保留最后一条 invoke_skill ToolMessage 的完整内容
      - 之前的 invoke_skill ToolMessage 压缩为摘要（保留 URL + 状态行，截断正文）
      - 非 invoke_skill 的 ToolMessage 保持不变
      - SystemMessage、HumanMessage、AIMessage 保持不变

    压缩后的 ToolMessage 格式：
      [已访问 invoke_skill]
      URL: https://example.com
      状态: 成功
      [完整内容已折叠，本次不重复输入]

    Args:
        messages: 当前完整消息列表

    Returns:
        压缩后的消息列表（原列表不被修改，返回新列表）
    """
    if not messages:
        return messages

    # 1. 找出所有 invoke_skill 的 tool_call_id
    invoke_skill_call_ids: "set[str]" = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.name == "invoke_skill":
                    if tc.id:
                        invoke_skill_call_ids.add(tc.id)

    # 若没有任何 invoke_skill 调用，直接返回
    if not invoke_skill_call_ids:
        return messages

    # 2. 找出对应的 ToolMessage 列表，按消息顺序排列
    invoke_skill_tool_msgs: "List[ToolMessage]" = []
    for msg in messages:
        if (
            isinstance(msg, ToolMessage)
            and msg.tool_call_id in invoke_skill_call_ids
        ):
            invoke_skill_tool_msgs.append(msg)

    # 若只有 0 或 1 条，不需要压缩
    if len(invoke_skill_tool_msgs) <= 1:
        return messages

    # 3. 确定需要压缩的 ToolMessage（除最后一条外的所有）
    to_compress_ids: "set[str]" = {
        m.tool_call_id
        for m in invoke_skill_tool_msgs[:-1]
        if m.tool_call_id
    }

    # 4. 对需要压缩的 ToolMessage 生成摘要内容
    def _make_summary(original_content: str) -> str:
        """从 invoke_skill ToolMessage 内容中提取 URL 和状态，生成折叠摘要。"""
        import re
        url = ""
        status = ""
        # 提取 [URL: ...] 标记（新格式，由 SkillTool 写入）
        url_match = re.search(r'\[URL:\s*(https?://[^\]]+)\]', original_content)
        if url_match:
            url = url_match.group(1).strip()
        else:
            # fallback：从内容中提取第一个 URL
            url_match2 = re.search(r'https?://[^\s\'"，。\]）]+', original_content)
            if url_match2:
                url = url_match2.group(0).rstrip('.,;)')
        # 提取 [Status: ...] 或 [状态: ...] 标记
        status_match = re.search(r'\[(?:Status|状态):\s*([^\]]+)\]', original_content)
        if status_match:
            status = status_match.group(1).strip()
        else:
            status = "已完成"

        parts = ["[已访问 invoke_skill]"]
        if url:
            parts.append(f"URL: {url}")
        parts.append(f"状态: {status}")
        parts.append("[完整内容已折叠，本次不重复输入]")
        return "\n".join(parts)

    # 5. 重建消息列表，替换需要压缩的 ToolMessage
    result: "List[BaseMessage]" = []
    compressed_count = 0
    for msg in messages:
        if (
            isinstance(msg, ToolMessage)
            and msg.tool_call_id in to_compress_ids
        ):
            # 压缩：替换为摘要
            summary = _make_summary(msg.content or "")
            result.append(ToolMessage(content=summary, tool_call_id=msg.tool_call_id))
            compressed_count += 1
        else:
            result.append(msg)

    if compressed_count > 0:
        logger.debug(
            "_compress_invoke_skill_history: compressed %d invoke_skill ToolMessages "
            "(kept last 1 full, %d summarized)",
            compressed_count, compressed_count,
        )

    return result


def _get_tool_calls_signature(tool_calls: list) -> "Optional[frozenset]":
    """获取一组工具调用的"签名"（用于检测重复调用）。

    注意：对于 fs_write_file 工具，只用工具名+路径作为签名，不含 content，
    因为弱模型每次写入相同文件时内容可能略有不同（换行、格式差异），
    但实质上是在反复重写同一个文件，属于需要被检测的循环行为。
    """
    if not tool_calls:
        return None
    sig_parts = []
    for tc in tool_calls:
        if tc.name == "fs_write_file":
            # 只用路径，忽略内容差异，精准检测"重复写同一文件"
            path_key = tc.args.get("path", "")
            sig_parts.append((tc.name, f"path={path_key}"))
        else:
            sig_parts.append((tc.name, str(tuple(sorted(tc.args.items())))))
    return frozenset(sig_parts)


def _get_last_ai_tool_calls_signature(messages: "List[BaseMessage]") -> "Optional[frozenset]":
    """从消息历史中获取最近一次 AIMessage 的工具调用签名。"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            return _get_tool_calls_signature(msg.tool_calls)
    return None


def _has_write_tool_called(messages: "List[BaseMessage]", write_tool_names: "tuple[str, ...]") -> bool:
    """
    检查消息历史中是否已经调用过写文件类工具（如 fs_write_file）。

    通过遍历 AIMessage 的 tool_calls 来判断，因为写文件工具的调用记录在
    AIMessage（含 tool_calls）中，而不是在 ToolMessage 中。

    Args:
        messages: 当前消息历史
        write_tool_names: 需要检查的写文件工具名称元组

    Returns:
        True 表示至少已调用过一次写文件工具，False 表示尚未调用
    """
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.name in write_tool_names:
                    return True
    # 也检查 ToolMessage 的 tool_call_id 方式（通过 ToolMessage 内容关键字）
    # 因为 sub-engine 中 AIMessage with tool_calls 不一定被保留在历史中，
    # 但 ToolMessage 的内容会反映工具执行结果
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            # fs_write_file 成功时通常返回 "Successfully wrote file: ..." 或 "Successfully appended to file: ..."
            if "successfully wrote" in content.lower() or "successfully appended to" in content.lower():
                return True
    return False


# 文件写入类工具名称（需要确认已写文件才算完成任务）
_FILE_WRITE_TOOL_NAMES = ("fs_write_file",)


def _is_substantive_final_response(content: str) -> bool:
    """Return True for an answer that is complete enough to end a turn.

    The old first-turn retry rule treated every text-only response as a failed
    tool call.  That made a completed answer re-enter the tool loop merely
    because tools happened to be bound (including file-writing tools).
    """
    text = (content or "").strip()
    if len(text) < 80:
        return False
    # A multi-line answer or a reasonably detailed paragraph is a terminal
    # response unless another explicit retry condition below applies.
    return "\n" in text or len(text) >= 180


def _should_retry_tool_call(
    response: AIMessage,
    iteration_count: int,
    messages: "Optional[List[BaseMessage]]" = None,
    available_tool_names: "Optional[set[str]]" = None,
) -> bool:
    """
    判断是否应该引导模型重试调用工具。

    触发条件（满足任一）：
      1. 模型回复为空（无 content、无 tool_calls）—— 通常是 qwen3 等 thinking 模型
         在上下文混乱或子引擎超时后的异常状态
      2. 模型回复包含"我无法/unable to"等词汇 + 没有工具调用
      3. 首轮回复很短、且没有工具调用（通常是模型尚未启动任务）
      4. 消息历史中还有来自搜索结果的 URL 尚未被访问
         （防止模型访问第一个 URL 后满足于内容直接输出最终答案，跳过其余 URL）
      5. 文件写入任务未完成 — 可用工具中有 fs_write_file，但历史中尚未出现
         fs_write_file 的成功调用（模型只创建了目录就停下来了）

    Note: 仅在 tool_call_retry=True 时生效（由调用方控制）。
    """
    if response.tool_calls:
        return False  # 已经调用了工具，不需要重试

    content_lower = (response.content or "").lower()

    # 条件 0：空响应兜底（qwen3 在上下文混乱或子引擎超时后可能返回空内容）
    if not content_lower.strip():
        return True

    # 条件 1：包含"无法/cannot"类词汇
    if any(hint in content_lower for hint in _SHOULD_HAVE_USED_TOOL_HINTS):
        return True

    # 条件 2：前两轮的简短回复没有调用工具（初始阶段未启动工具调用链）
    # 注意：仅当消息历史中没有任何 ToolMessage 时才适用。
    # 如果模型已经调用过工具并产生了结果（如写了文件、执行了搜索），说明工具链已正常启动。
    # 此时不应因为 iteration_count <= 2 而强制注入重试引导，否则会造成不必要的额外 LLM 调用，
    # 多次叠加后容易触发 Ollama 超时（timed out）错误。
    if iteration_count <= 2:
        has_any_tool_results = messages is not None and any(isinstance(m, ToolMessage) for m in messages)
        if not has_any_tool_results and not _is_substantive_final_response(response.content or ""):
            return True

    # 条件 3：消息历史中还有未访问的 URL（需继续遍历所有搜索结果）
    if messages is not None and _has_unvisited_urls(messages):
        return True

    # 条件 4：编码/文件操作任务 — 模型在「工具调用后」返回空内容，可能是任务还未完成
    # 只在有 ToolMessage 且没有搜索 URL 的场景下适用（纯编码任务）
    # 注意：若模型已经输出了非空文字（如 module_list= ...），则已满足任务要求，不应再 push
    if not content_lower.strip() and messages is not None:
        has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
        has_search_urls = len(_extract_urls_from_messages(messages)) > 0
        if has_tool_results and not has_search_urls:
            return True

    # 条件 5：文件写入任务未完成
    # 当可用工具包含 fs_write_file，但历史中 fs_write_file 尚未被成功调用时，
    # 模型可能只做了一半（如只创建了目录），就用文字总结代替了实际写文件。
    # 注意：如果模型已经调用过 fs_write_file，则任务已启动，
    # 只在回复为空（卡住）时才 retry，避免对已写文件的任务反复推动造成死循环。
    if messages is not None and available_tool_names is not None:
        has_write_tool_available = bool(available_tool_names & set(_FILE_WRITE_TOOL_NAMES))
        if has_write_tool_available:
            has_search_urls = len(_extract_urls_from_messages(messages)) > 0
            if not has_search_urls:  # 非搜索任务（纯文件操作任务）
                has_any_tool_result = any(isinstance(m, ToolMessage) for m in messages)
                if has_any_tool_result:
                    write_already_called = _has_write_tool_called(messages, _FILE_WRITE_TOOL_NAMES)
                    if not write_already_called:
                        # 还没写文件，强制推动
                        return True
                    elif not content_lower.strip():
                        # 写文件已被调用，但本轮回复为空（任务卡住），再 retry 一次
                        return True

    return False


class ReActEngine:
    """
    ReAct 循环引擎。

    Args:
        llm             : BaseLLM 实例（建议使用 bind_tools 前的原始实例，
                          ReActEngine 内部会在每次 invoke 前自动 bind_tools）
        tools           : 可用工具列表
        system_prompt   : 系统提示（可选）
        max_iterations  : 最大迭代次数（防止无限循环）
        debug_hooks     : 可选调试钩子字典（见 _run_hooks）
        tool_call_retry : 是否启用工具调用引导重试（针对弱模型）
        max_tool_retry  : 最多引导重试次数
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: List[BaseTool],
        system_prompt: Optional[str] = None,
        max_iterations: int = 20,
        debug_hooks: Optional[Dict[str, Any]] = None,
        tool_call_retry: bool = True,
        max_tool_retry: int = 5,
        max_tool_exec_retries: int = 3,
        # Context management parameters
        max_tool_result_length: int = 4000,
        enable_tool_result_summarization: bool = True,
        message_sliding_window: int = 10,
        reuse_system_prompt: bool = True,
    ):
        self._llm = llm.bind_tools(tools) if tools else llm
        self._tools: Dict[str, BaseTool] = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self._hooks = debug_hooks or {}
        self.tool_call_retry = tool_call_retry
        self.max_tool_retry = max_tool_retry
        self.max_tool_exec_retries = max_tool_exec_retries
        
        # Context management settings
        self.max_tool_result_length = max_tool_result_length
        self.enable_tool_result_summarization = enable_tool_result_summarization
        self.message_sliding_window = message_sliding_window
        self.reuse_system_prompt = reuse_system_prompt
        self._system_prompt_injected = False  # Track if system prompt is already in messages
        self._current_messages: List[BaseMessage] = []  # 实时引用，供 _execute_tool 访问（用于注册表快照）

    # ------------------------------------------------------------------
    # 公开接口（与 LangGraph compiled graph 兼容）
    # ------------------------------------------------------------------

    def invoke(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        同步完整运行。

        Args:
            state : AgentState 兼容字典，必须包含 "messages" 键
            config: 兼容参数（保留，暂不使用）

        Returns:
            {"messages": [所有消息列表]} 格式的字典
        """
        messages = list(state.get("messages", []))
        memory_context = state.get("memory_context", "")
        iteration_count = state.get("iteration_count", 0)

        messages = self._prepend_system(messages, memory_context)

        # 工具调用引导重试计数器
        tool_retry_count = 0
        # 重复工具调用检测：记录上一轮工具调用签名和连续重复次数
        last_tool_signature: Optional[frozenset] = None
        consecutive_same_tool_count = 0
        _MAX_CONSECUTIVE_SAME_TOOL = 2  # 允许相同工具调用的最大连续次数（fs_write_file 路径级检测，2次即触发）

        for _ in range(self.max_iterations):
            if iteration_count >= self.max_iterations:
                logger.warning("Max iterations (%d) reached", self.max_iterations)
                messages.append(
                    AIMessage(
                        content=(
                            f"I've reached the maximum number of iterations "
                            f"({self.max_iterations}). "
                            "Here is what I've accomplished so far."
                        )
                    )
                )
                break

            # ── 更新当前消息引用（供 _execute_tool 在存取注册表时使用快照）──
            self._current_messages = messages

            # ── 判断当前所处阶段，决定 LLM 输入构建方式 ────────────────────
            # 工具阶段：还有未访问的 URL，或尚无 ToolMessage（初始阶段），或是编码任务
            # 总结阶段：这是一个网络搜索任务（有搜索 URL）且所有 URL 已访问完
            has_tool_msgs_now = any(isinstance(m, ToolMessage) for m in messages)
            # 只有存在搜索 URL 时才可能进入总结阶段（编码/文件任务不走总结阶段）
            has_search_urls_now = len(_extract_urls_from_messages(messages)) > 0
            is_summary_phase = (
                has_tool_msgs_now
                and has_search_urls_now
                and not _has_unvisited_urls(messages)
            )

            # 构建精简的 LLM 输入消息（工具阶段只传query+引导，总结阶段传query+所有工具结果）
            messages_for_llm = _build_minimal_messages_for_llm(messages, is_summary_phase=is_summary_phase)

            # before_llm hook：传完整 messages（状态展示）以及 messages_for_llm（实际输入展示）
            self._call_hook("before_llm", messages=messages, iteration=iteration_count,
                            messages_for_llm=messages_for_llm)
            try:
                response: AIMessage = self._llm.invoke(messages_for_llm)
            except Exception as exc:
                import sys
                msg = f"LLM call failed: {exc}"
                logger.error(msg)
                print(
                    f"\n\033[1;31m[LLM Fatal Error]\033[0m {msg}\n\033[33mExiting agent loop.\033[0m\n",
                    file=sys.stderr,
                )
                raise

            iteration_count += 1
            self._call_hook("after_llm", message=response, iteration=iteration_count)

            if not response.tool_calls:
                # 无工具调用：追加最终回复到历史
                messages.append(response)

                # 检查是否需要引导重试工具调用
                _available_tool_names = set(self._tools.keys()) if self._tools else set()
                if (
                    self.tool_call_retry
                    and tool_retry_count < self.max_tool_retry
                    and self._tools  # 有可用工具才重试
                    and _should_retry_tool_call(response, iteration_count, messages, _available_tool_names)
                ):
                    tool_retry_count += 1
                    guidance = _build_context_aware_guidance(messages, tool_retry_count)
                    logger.info(
                        "Tool call guidance retry #%d (iteration=%d)",
                        tool_retry_count, iteration_count,
                    )
                    
                    # ── 打印引导重试信息 ──────────────────────────────────────
                    from local_agent.core.debug import print_retry_guidance
                    retry_reason = "Model did not call any tool"
                    if any(hint in (response.content or "").lower() for hint in _SHOULD_HAVE_USED_TOOL_HINTS):
                        retry_reason = "Model response contains 'unable to/cannot' keywords"
                    elif iteration_count <= 2:
                        retry_reason = "First iteration without tool call"
                    elif _has_unvisited_urls(messages):
                        retry_reason = "There are still unvisited URLs from search results"
                    elif not (response.content or "").strip() and not _extract_urls_from_messages(messages):
                        retry_reason = "Coding task: model returned empty response mid-task"
                    elif _available_tool_names & set(_FILE_WRITE_TOOL_NAMES) and not _has_write_tool_called(messages, _FILE_WRITE_TOOL_NAMES):
                        retry_reason = "File-write task: model stopped without calling fs_write_file"
                    print_retry_guidance(tool_retry_count, retry_reason, guidance)
                    
                    # 追加引导消息，继续循环
                    messages.append(HumanMessage(content=guidance))
                    continue

                # 无工具调用 → ReAct 循环结束
                break

            # ── 有工具调用：不将含 tool_calls 的 AIMessage 放入历史，仅追加工具结果 ──

            # ── 重复工具调用检测 ─────────────────────────────────────────────
            current_sig = _get_tool_calls_signature(response.tool_calls)
            if current_sig is not None and current_sig == last_tool_signature:
                consecutive_same_tool_count += 1
                logger.warning(
                    "Detected repeated tool call (same tools+args) for %d consecutive time(s). "
                    "Signature: %s",
                    consecutive_same_tool_count, current_sig,
                )
                if consecutive_same_tool_count >= _MAX_CONSECUTIVE_SAME_TOOL:
                    logger.error(
                        "Breaking out of agent loop: identical tool calls repeated %d times. "
                        "This indicates a stuck loop. Returning current results.",
                        consecutive_same_tool_count,
                    )
                    import sys
                    print(
                        f"\n\033[1;33m[Loop Detection]\033[0m Identical tool call repeated "
                        f"{consecutive_same_tool_count} times — breaking loop to avoid infinite recursion.\n",
                        file=sys.stderr,
                    )
                    # 追加一条空 AI 消息（不带内容），让调用方的 final_text 提取跳过这条消息，
                    # 从而回退到兜底逻辑，而不是把这条错误提示当作有效的步骤输出。
                    # （调用方 _run_agent_step 中的 "兜底 1" 已过滤 startswith("I detected") 消息）
                    messages.append(AIMessage(
                        content=(
                            "I detected that I'm stuck in a loop calling the same tool repeatedly "
                            "without making progress (possibly because the URLs returned no content). "
                            "Here is a summary based on the search results I obtained earlier."
                        )
                    ))
                    break
            else:
                consecutive_same_tool_count = 0
            last_tool_signature = current_sig

            # 重置重试计数（成功调用工具后重置）
            tool_retry_count = 0

            # 执行工具调用，只将工具结果追加到历史（不追加含 tool_calls 的 AIMessage）
            for tc in response.tool_calls:
                tool_msg = self._execute_tool(tc)
                messages.append(tool_msg)
                self._call_hook("after_tool", tool_call=tc, tool_message=tool_msg)
            
            # 应用滑动窗口策略（在工具调用后）
            messages = self._apply_sliding_window(messages)

        return {"messages": messages}

    def stream(
        self,
        state: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        stream_mode: Any = None,
    ) -> Iterator[Tuple[str, Any]]:
        """
        流式运行，产生 (mode, data) 元组。

        事件格式（与 LangGraph stream_mode=["messages","values"] 兼容）：
          ("messages", (AIMessageChunk, {}))  – LLM 输出 chunk
          ("messages", (ToolMessage, {}))      – 工具执行完成
          ("values",   {"messages": [...]})    – 每次迭代完成后的全量快照

        Args:
            state      : AgentState 兼容字典
            config     : 兼容参数（保留）
            stream_mode: 兼容参数（保留，内部统一产生两种事件）
        """
        messages = list(state.get("messages", []))
        memory_context = state.get("memory_context", "")
        iteration_count = state.get("iteration_count", 0)

        messages = self._prepend_system(messages, memory_context)

        # 工具调用引导重试计数器
        tool_retry_count = 0
        # 重复工具调用检测：记录上一轮工具调用签名和连续重复次数
        last_tool_signature: Optional[frozenset] = None
        consecutive_same_tool_count = 0
        _MAX_CONSECUTIVE_SAME_TOOL = 2  # 允许相同工具调用的最大连续次数（fs_write_file 路径级检测，2次即触发）

        for _ in range(self.max_iterations):
            if iteration_count >= self.max_iterations:
                final_msg = AIMessage(
                    content=(
                        f"I've reached the maximum number of iterations "
                        f"({self.max_iterations}). "
                        "Here is what I've accomplished so far."
                    )
                )
                messages.append(final_msg)
                yield ("messages", (AIMessageChunk(content=final_msg.content), {}))
                yield ("values", {"messages": list(messages)})
                break

            # ── 更新当前消息引用（供 _execute_tool 在存取注册表时使用快照）──
            self._current_messages = messages

            # 收集流式 chunks，同时产生 ("messages", ...) 事件
            accumulated_chunks: List[AIMessageChunk] = []
            announced_tool_ids: set[str] = set()

            # ── 判断当前所处阶段，决定 LLM 输入构建方式 ────────────────────
            # 工具阶段：还有未访问的 URL，或尚无 ToolMessage（初始阶段），或是编码任务
            # 总结阶段：这是一个网络搜索任务（有搜索 URL）且所有 URL 已访问完
            has_tool_msgs_now = any(isinstance(m, ToolMessage) for m in messages)
            # 只有存在搜索 URL 时才可能进入总结阶段（编码/文件任务不走总结阶段）
            has_search_urls_now = len(_extract_urls_from_messages(messages)) > 0
            is_summary_phase = (
                has_tool_msgs_now
                and has_search_urls_now
                and not _has_unvisited_urls(messages)
            )

            # 构建精简的 LLM 输入消息（工具阶段只传query+引导，总结阶段传query+所有工具结果）
            messages_for_llm = _build_minimal_messages_for_llm(messages, is_summary_phase=is_summary_phase)

            # before_llm hook：传完整 messages（状态展示）以及 messages_for_llm（实际输入展示）
            self._call_hook("before_llm", messages=messages, iteration=iteration_count,
                            messages_for_llm=messages_for_llm)
            try:
                for chunk in self._llm.stream(messages_for_llm):
                    # 通知新工具调用
                    for tc in chunk.tool_calls:
                        if tc.id not in announced_tool_ids:
                            announced_tool_ids.add(tc.id)
                    yield ("messages", (chunk, {}))
                    accumulated_chunks.append(chunk)
            except Exception as exc:
                import sys
                msg = f"LLM streaming failed: {exc}"
                logger.error(msg)
                print(
                    f"\n\033[1;31m[LLM Fatal Error]\033[0m {msg}\n\033[33mExiting agent loop.\033[0m\n",
                    file=sys.stderr,
                )
                raise

            # 合并 chunks 为完整 AIMessage
            if accumulated_chunks:
                merged = accumulated_chunks[0]
                for c in accumulated_chunks[1:]:
                    merged = merged + c
                response = AIMessage(
                    content=merged.content,
                    tool_calls=merged.tool_calls,
                )
            else:
                response = AIMessage(content="")

            iteration_count += 1
            self._call_hook("after_llm", message=response, iteration=iteration_count)

            if not response.tool_calls:
                # 无工具调用：追加最终回复到历史
                messages.append(response)

                # 产生全量快照
                yield ("values", {"messages": list(messages)})

                # 检查是否需要引导重试工具调用
                _available_tool_names_stream = set(self._tools.keys()) if self._tools else set()
                if (
                    self.tool_call_retry
                    and tool_retry_count < self.max_tool_retry
                    and self._tools  # 有可用工具才重试
                    and _should_retry_tool_call(response, iteration_count, messages, _available_tool_names_stream)
                ):
                    tool_retry_count += 1
                    guidance = _build_context_aware_guidance(messages, tool_retry_count)
                    logger.info(
                        "Tool call guidance retry #%d (iteration=%d) [stream]",
                        tool_retry_count, iteration_count,
                    )
                    
                    # ── 打印引导重试信息 ──────────────────────────────────────
                    from local_agent.core.debug import print_retry_guidance
                    retry_reason = "Model did not call any tool"
                    if any(hint in (response.content or "").lower() for hint in _SHOULD_HAVE_USED_TOOL_HINTS):
                        retry_reason = "Model response contains 'unable to/cannot' keywords"
                    elif iteration_count <= 2:
                        retry_reason = "First iteration without tool call"
                    elif _has_unvisited_urls(messages):
                        retry_reason = "There are still unvisited URLs from search results"
                    elif not (response.content or "").strip() and not _extract_urls_from_messages(messages):
                        retry_reason = "Coding task: model returned empty response mid-task"
                    elif _available_tool_names_stream & set(_FILE_WRITE_TOOL_NAMES) and not _has_write_tool_called(messages, _FILE_WRITE_TOOL_NAMES):
                        retry_reason = "File-write task: model stopped without calling fs_write_file"
                    print_retry_guidance(tool_retry_count, retry_reason, guidance)
                    
                    # 追加引导消息，继续循环
                    messages.append(HumanMessage(content=guidance))
                    yield ("values", {"messages": list(messages)})
                    continue

                # 无工具调用 → 结束
                break

            # ── 有工具调用：不将含 tool_calls 的 AIMessage 放入历史，仅追加工具结果 ──

            # ── 重复工具调用检测 ─────────────────────────────────────────────
            current_sig = _get_tool_calls_signature(response.tool_calls)
            if current_sig is not None and current_sig == last_tool_signature:
                consecutive_same_tool_count += 1
                logger.warning(
                    "Detected repeated tool call (same tools+args) for %d consecutive time(s). "
                    "Signature: %s",
                    consecutive_same_tool_count, current_sig,
                )
                if consecutive_same_tool_count >= _MAX_CONSECUTIVE_SAME_TOOL:
                    logger.error(
                        "Breaking out of agent loop [stream]: identical tool calls repeated %d times.",
                        consecutive_same_tool_count,
                    )
                    import sys
                    print(
                        f"\n\033[1;33m[Loop Detection]\033[0m Identical tool call repeated "
                        f"{consecutive_same_tool_count} times — breaking loop to avoid infinite recursion.\n",
                        file=sys.stderr,
                    )
                    # 追加空内容消息（loop detection marker），调用方 _run_agent_step
                    # 中的 "兜底 1" 已过滤 startswith("I detected") 消息，不会把
                    # 这个错误消息当作步骤有效输出。
                    stuck_msg = AIMessage(
                        content=(
                            "I detected that I'm stuck in a loop calling the same tool repeatedly "
                            "without making progress (possibly because the URLs returned no content). "
                            "Here is a summary based on the search results I obtained earlier."
                        )
                    )
                    messages.append(stuck_msg)
                    yield ("messages", (AIMessageChunk(content=stuck_msg.content), {}))
                    yield ("values", {"messages": list(messages)})
                    break
            else:
                consecutive_same_tool_count = 0
            last_tool_signature = current_sig

            # 重置重试计数（成功调用工具后重置）
            tool_retry_count = 0

            # 执行工具调用，只将工具结果追加到历史（不追加含 tool_calls 的 AIMessage）
            for tc in response.tool_calls:
                tool_msg = self._execute_tool(tc)
                messages.append(tool_msg)
                yield ("messages", (tool_msg, {}))
                self._call_hook("after_tool", tool_call=tc, tool_message=tool_msg)

            # 应用滑动窗口策略（在工具调用后）
            messages = self._apply_sliding_window(messages)

            # 工具执行后再次产生快照
            yield ("values", {"messages": list(messages)})

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _prepend_system(
        self, messages: List[BaseMessage], memory_context: str
    ) -> List[BaseMessage]:
        """确保消息列表以 SystemMessage 开头，含记忆上下文。
        
        如果启用了 reuse_system_prompt，只在第一次调用时注入 SystemMessage，
        后续迭代会检测到已存在的 SystemMessage 并跳过重复注入。
        """
        if not self.system_prompt:
            return messages

        full_prompt = self.system_prompt
        if memory_context:
            full_prompt = f"{full_prompt}\n\n## Relevant Context from Memory:\n{memory_context}"

        # 如果启用了复用模式且已经注入过，直接返回
        if self.reuse_system_prompt and self._system_prompt_injected:
            # 检查第一条消息是否是 SystemMessage
            if messages and isinstance(messages[0], SystemMessage):
                return messages
            # 如果不是（被外部修改了），重新注入
            self._system_prompt_injected = False

        if messages and isinstance(messages[0], SystemMessage):
            # 替换已有 SystemMessage
            result = [SystemMessage(content=full_prompt)] + messages[1:]
        else:
            result = [SystemMessage(content=full_prompt)] + messages
        
        # 标记为已注入
        if self.reuse_system_prompt:
            self._system_prompt_injected = True
        
        return result

    def _execute_tool(self, tool_call: ToolCall) -> ToolMessage:
        """查找并执行工具，返回 ToolMessage。

        工具执行失败（抛出异常或返回 [Tool Error] 前缀）时，
        最多自动重试 max_tool_exec_retries 次。
        工具执行成功后：
          1. 应用上下文管理策略（截断/总结）
          2. 调用 ToolResultParser 解析结果
          3. 将结构化摘要注入 ToolMessage content，供下一步 LLM 使用
        
        注意：debug 打印由上层的 after_tool hook（graph.py）统一处理，
        此处不直接调用 print_tool_call，以避免重复输出。
        
        Context Management:
        - 截断过长的工具结果（超过 max_tool_result_length）
        - 如果结果超过 8000 字符且启用了 summarization，调用 LLM 进行总结

        Prompt Registry:
        - 若工具为 invoke_skill，在执行前生成唯一 invocation_id，
          将当前 messages 快照存入 PromptContextRegistry
        - 执行完成后，用 invocation_id 取回条目（并从注册表清除）以打印调试信息
        """
        from local_agent.core.prompt_registry import (
            generate_invocation_id,
            save_prompt_context,
            retrieve_prompt_context,
        )
        from local_agent.core.debug import (
            print_prompt_context_save,
            print_prompt_context_restore,
        )

        is_invoke_skill = tool_call.name == "invoke_skill"
        invocation_id: Optional[str] = None

        if is_invoke_skill:
            # ── 调用子 skill 前：生成唯一 ID，存入注册表 ──────────────────────
            invocation_id = generate_invocation_id()
            skill_name = tool_call.args.get("skill_name", "")
            task = tool_call.args.get("task", "") or tool_call.args.get("url", "")
            save_prompt_context(
                invocation_id=invocation_id,
                messages=self._current_messages,
                skill_name=skill_name,
                task=task,
            )
            print_prompt_context_save(
                invocation_id=invocation_id,
                skill_name=skill_name,
                task=task,
                messages_count=len(self._current_messages),
            )
            logger.debug(
                "_execute_tool: saved prompt context for invoke_skill '%s' "
                "(invocation_id=%s, messages=%d)",
                skill_name, invocation_id, len(self._current_messages),
            )

        tool = self._tools.get(tool_call.name)
        if tool is None:
            error_content = f"[Tool Error] Unknown tool: '{tool_call.name}'"
            logger.warning("Tool '%s' not found in registry", tool_call.name)
            if is_invoke_skill and invocation_id:
                retrieve_prompt_context(invocation_id)  # 清除注册表中的无效条目
            return ToolMessage(content=error_content, tool_call_id=tool_call.id)

        last_error: str = f"[Tool Error] Unknown failure for tool '{tool_call.name}'"
        for attempt in range(self.max_tool_exec_retries + 1):
            if attempt > 0:
                logger.info(
                    "Retrying tool '%s' (attempt %d/%d) after failure: %s",
                    tool_call.name, attempt, self.max_tool_exec_retries, last_error,
                )
            try:
                logger.debug(
                    "Executing tool '%s' with args: %s (attempt %d)",
                    tool_call.name, tool_call.args, attempt + 1,
                )
                result = tool.run(**tool_call.args)
                # 工具内部已将异常转为 [Tool Error] 字符串（BaseTool.run 的行为）
                if not result.startswith("[Tool Error]"):
                    # 执行成功，应用上下文管理策略
                    result = self._manage_tool_result(result, tool_call.name)
                    # 解析工具结果，生成结构化摘要注入 ToolMessage
                    content = self._build_tool_message_content(tool_call.name, result)
                    if is_invoke_skill and invocation_id:
                        # ── 子 skill 执行完成：用 ID 取回父上下文并清除 ──────────
                        entry = retrieve_prompt_context(invocation_id)
                        if entry:
                            print_prompt_context_restore(
                                invocation_id=invocation_id,
                                skill_name=entry.skill_name,
                                result_len=len(content),
                            )
                            logger.debug(
                                "_execute_tool: retrieved prompt context for '%s' "
                                "(invocation_id=%s, saved_messages=%d, result_len=%d)",
                                entry.skill_name, invocation_id,
                                len(entry.saved_messages), len(content),
                            )
                    return ToolMessage(content=content, tool_call_id=tool_call.id)
                # 工具返回错误字符串，记录后继续重试
                last_error = result
                logger.warning(
                    "Tool '%s' returned error on attempt %d/%d: %s",
                    tool_call.name, attempt + 1, self.max_tool_exec_retries + 1, result,
                )
            except Exception as exc:
                last_error = f"[Tool Error] {exc}"
                logger.error(
                    "Tool '%s' raised exception on attempt %d/%d: %s",
                    tool_call.name, attempt + 1, self.max_tool_exec_retries + 1,
                    exc, exc_info=True,
                )

        # 所有重试耗尽，返回错误 ToolMessage 而非抛出异常，让主循环可以继续执行
        if is_invoke_skill and invocation_id:
            # 即使失败也要清除注册表条目，避免内存泄漏
            retrieve_prompt_context(invocation_id)

        final_msg = (
            f"Tool '{tool_call.name}' failed after "
            f"{self.max_tool_exec_retries + 1} attempt(s). Last error: {last_error}"
        )
        logger.error(final_msg)

        # 终端醒目输出（但不退出，让 LLM 能收到失败通知后继续）
        import sys
        print(
            f"\n\033[1;31m[Tool Error]\033[0m {final_msg}\n"
            f"\033[33mReturning error message to LLM instead of exiting.\033[0m\n",
            file=sys.stderr,
        )
        # 返回包含简短错误说明的 ToolMessage，不传递详细的错误堆栈给 LLM
        error_summary = (
            f"[Tool Error] Tool '{tool_call.name}' failed and cannot be retried. "
            f"Please try a different approach or tool."
        )
        return ToolMessage(content=error_summary, tool_call_id=tool_call.id)

    def _build_tool_message_content(self, tool_name: str, raw_result: str) -> str:
        """
        解析工具执行结果，构建注入 ToolMessage 的结构化内容。

        结构：
          [工具结果: tool_name | 状态: success]
          <结构化摘要>

        若解析失败，回退到原始结果。

        Args:
            tool_name:  工具名称
            raw_result: 经上下文管理处理后的原始结果字符串

        Returns:
            注入 ToolMessage 的内容字符串
        """
        try:
            from local_agent.core.tool_result_parser import ToolResultParser
            parser = ToolResultParser()
            parsed = parser.parse(tool_name, raw_result)
            logger.debug(
                "ToolResultParser: tool='%s' status='%s' items=%s truncated=%s",
                tool_name, parsed.status, parsed.item_count, parsed.truncated,
            )
            return parsed.to_llm_context()
        except Exception as exc:
            logger.debug(
                "ToolResultParser: parsing failed for tool '%s': %s (falling back to raw)",
                tool_name, exc,
            )
            return raw_result

    def _manage_tool_result(self, result: str, tool_name: str) -> str:
        """
        管理工具结果的长度，防止上下文爆炸。
        
        策略：
          1. 如果结果长度 <= max_tool_result_length，直接返回
          2. 如果 max_tool_result_length < 结果长度 <= 8000，截断并添加提示
          3. 如果结果长度 > 8000 且启用了 summarization，调用 LLM 总结
          4. 否则截断到 max_tool_result_length
        
        Args:
            result: 工具返回的原始结果字符串
            tool_name: 工具名称（用于日志）
        
        Returns:
            处理后的结果字符串
        """
        original_length = len(result)
        
        # 策略 0: 不限制模式（max_tool_result_length <= 0），直接返回
        if self.max_tool_result_length <= 0:
            return result
        
        # 策略 1: 结果不长，直接返回
        if original_length <= self.max_tool_result_length:
            return result
        
        # 策略 2: 中等长度（4000-8000），直接截断
        if original_length <= 8000:
            truncated = result[:self.max_tool_result_length]
            suffix = f"\n\n[Note: Tool result truncated from {original_length} to {self.max_tool_result_length} characters]"
            logger.info(
                "Tool '%s' result truncated: %d → %d chars",
                tool_name, original_length, self.max_tool_result_length,
            )
            return truncated + suffix
        
        # 策略 3: 超长结果且启用总结，使用 LLM 总结
        if self.enable_tool_result_summarization:
            try:
                summary = self._summarize_tool_result(result, tool_name)
                logger.info(
                    "Tool '%s' result summarized: %d → %d chars",
                    tool_name, original_length, len(summary),
                )
                return summary
            except Exception as exc:
                logger.warning(
                    "Failed to summarize tool '%s' result: %s. Falling back to truncation.",
                    tool_name, exc,
                )
        
        # 策略 4: 总结失败或未启用，截断
        truncated = result[:self.max_tool_result_length]
        suffix = f"\n\n[Note: Tool result truncated from {original_length} to {self.max_tool_result_length} characters. Original content was too long.]"
        logger.info(
            "Tool '%s' result truncated (fallback): %d → %d chars",
            tool_name, original_length, self.max_tool_result_length,
        )
        return truncated + suffix

    def _summarize_tool_result(self, result: str, tool_name: str) -> str:
        """
        使用 LLM 总结超长工具结果。
        
        Args:
            result: 原始工具结果
            tool_name: 工具名称
        
        Returns:
            总结后的文本（带原始长度提示）
        """
        # 构造总结提示
        summarization_prompt = f"""The tool '{tool_name}' returned a very long result ({len(result)} characters).
Please provide a concise summary of the key information, preserving important details, URLs, and data points.
Keep the summary under 3000 characters but ensure all critical information is retained.

Original tool result:
{result[:10000]}  
{"[... content continues beyond 10000 chars ...]" if len(result) > 10000 else ""}

Provide a structured summary:"""

        try:
            # 创建一个不绑定工具的 LLM 实例用于总结（避免递归调用工具）
            summarization_llm = self._llm.__class__(model=self._llm.model) if hasattr(self._llm, 'model') else self._llm  # type: ignore[attr-defined,call-arg]
            
            summary_messages: List[BaseMessage] = [
                SystemMessage(content="You are a helpful assistant that summarizes tool output concisely while preserving key information."),
                HumanMessage(content=summarization_prompt),
            ]
            
            response = summarization_llm.invoke(summary_messages)  # type: ignore[arg-type]
            summary_content = response.content if hasattr(response, 'content') else str(response)
            
            # 添加元信息
            prefix = f"[Summarized from {len(result)} chars by LLM]\n\n"
            return prefix + summary_content
            
        except Exception as exc:
            logger.error("LLM summarization failed: %s", exc)
            raise

    def _apply_sliding_window(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        应用滑动窗口策略，只保留最近的 N 条消息（除了 SystemMessage）。
        
        策略：
          - SystemMessage 始终保留在第一位
          - 保留最近的 message_sliding_window 条非 SystemMessage
          - 如果 message_sliding_window = 0，不应用限制
        
        Args:
            messages: 完整消息列表
        
        Returns:
            应用滑动窗口后的消息列表
        """
        if self.message_sliding_window <= 0:
            return messages
        
        if not messages:
            return messages
        
        # 分离 SystemMessage 和其他消息
        system_msg = None
        other_messages = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_msg = msg
            else:
                other_messages.append(msg)
        
        # 如果消息数量在窗口范围内，直接返回
        if len(other_messages) <= self.message_sliding_window:
            return messages
        
        # 只保留最近的 N 条消息
        kept_messages = other_messages[-self.message_sliding_window:]
        dropped_count = len(other_messages) - self.message_sliding_window
        
        logger.info(
            "Sliding window applied: dropped %d old messages, kept last %d",
            dropped_count, len(kept_messages),
        )
        
        # 重新组装：SystemMessage + 最近的消息
        result_msgs: List[BaseMessage] = []
        if system_msg:
            result_msgs.append(system_msg)
        for m in kept_messages:
            result_msgs.append(m)  # type: ignore[arg-type]
        return result_msgs

    def _call_hook(self, name: str, **kwargs: Any) -> None:
        """执行调试钩子（若已注册）。"""
        hook = self._hooks.get(name)
        if callable(hook):
            try:
                hook(**kwargs)
            except Exception as exc:
                logger.debug("Debug hook '%s' raised: %s", name, exc)
