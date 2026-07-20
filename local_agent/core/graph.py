"""
LocalAgent ReAct Graph
======================
原本基于 LangGraph StateGraph 的 ReAct 实现，现已替换为自主实现的 ReActEngine。

对外接口保持不变：
  create_agent_graph(llm, tools, system_prompt) → ReActEngine 实例

ReActEngine 实现了与 LangGraph compiled graph 完全兼容的接口：
  engine.invoke(state, config) → {"messages": [...]}
  engine.stream(state, config, stream_mode) → Iterator[(mode, data)]
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from local_agent.core.config import get_settings
from local_agent.core.react import ReActEngine
from local_agent.core.tools import BaseTool

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 默认系统提示（针对本地弱模型优化，强调 ReAct 格式 + 工具调用规则）
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_SYSTEM_PROMPT = """You are LocalAgent, a powerful AI assistant that can help with a wide variety of tasks.

## CRITICAL: Tool Usage Rules
You MUST follow these rules strictly:

1. **Always use tools for actions**: When the user asks you to perform any operation (read files, run code, search the web, execute commands, etc.), you MUST call the appropriate tool. Do NOT just describe what you would do — actually do it by calling the tool.

2. **Think before acting**: Before calling a tool, briefly state your reasoning (one sentence is enough). Then call the tool immediately.

3. **Use tool results**: After a tool returns its result, incorporate the result into your response. If the result is incomplete, call another tool to get more information.

4. **Do not guess**: If you need information from a file, database, or the internet, always call the tool to get it. Never make up file contents or command outputs.

5. **Complete the task**: Keep calling tools until the task is fully completed. Only give your final answer when you have all the information needed.

## Available Tool Categories
- **File system**: read_file, write_file, list_directory, search_files, create_directory, delete_file
- **Code execution**: execute_python, execute_shell / run_shell_command
- **Web browsing**: browse_web, click_element, extract_content
- **Web search**: web_search, search_wikipedia
- **Data analysis**: load_csv, query_dataframe, visualize_data
- **Memory**: save_memory, recall_memory
- **Git**: git_status, git_diff, git_commit
- **System**: get_system_info, list_processes

## Tool Calling Examples

Example 1 – List files:
User: "What files are in the current directory?"
Thought: I need to list the directory contents. I'll call list_directory.
[CALL list_directory with path="."]

Example 2 – Read a file:
User: "Show me the contents of README.md"
Thought: I need to read the file. I'll call read_file.
[CALL read_file with path="README.md"]

Example 3 – Run a command:
User: "Check the Python version"
Thought: I'll run the shell command to check.
[CALL run_shell_command with command="python --version"]

Example 4 – Multi-step task:
User: "Search for Python async tutorials and save the top result to a file"
Thought: First I'll search the web.
[CALL web_search with query="Python async tutorials"]
Thought: Got results. Now I'll save the top result to a file.
[CALL write_file with path="async_tutorial.txt" content="..."]

## Response Format
- Always call tools when action is needed
- Keep your reasoning brief (1-2 sentences before each tool call)
- After all tool calls complete, provide a concise summary to the user
"""


# ──────────────────────────────────────────────────────────────────────────────
# 工具精简：限制传给本地弱模型的工具数量
# ──────────────────────────────────────────────────────────────────────────────

# 工具类别优先级（越靠前越重要，当需要裁剪时优先保留）
_TOOL_PRIORITY_CATEGORIES = [
    "filesystem",
    "shell",
    "code",
    "search",
    "browser",
    "memory",
    "data",
    "git",
    "system",
    "general",
]


def _prune_tools(tools: List[BaseTool], max_tools: int) -> List[BaseTool]:
    """
    按优先级精简工具列表，确保传给本地模型的工具数量不超过 max_tools。

    策略：
      1. 按工具的 metadata["category"] 排序（优先级高的类别排前面）
      2. 截取前 max_tools 个
      3. 始终保留名称包含常用关键词的工具（如 list_directory, read_file 等）

    Args:
        tools    : 完整工具列表
        max_tools: 最大工具数量，0 表示不限制

    Returns:
        精简后的工具列表
    """
    if max_tools <= 0 or len(tools) <= max_tools:
        return tools

    def _priority(tool: BaseTool) -> int:
        category = tool.metadata.get("category", "general") if tool.metadata else "general"
        try:
            return _TOOL_PRIORITY_CATEGORIES.index(category)
        except ValueError:
            return len(_TOOL_PRIORITY_CATEGORIES)

    # 核心工具：无论如何都保留（按名称精确匹配或子串匹配，涵盖 fs_* 前缀）
    _CORE_TOOL_KEYWORDS = {
        # filesystem (fs_ prefix)
        "fs_write_file", "fs_read_file", "fs_list_dir", "fs_create_dir",
        "fs_search_files", "fs_grep", "fs_delete_file",
        # shell
        "shell_run", "run_shell", "execute_shell",
        # code execution
        "code_execute_python", "code_execute_python_inline",
        "code_execute_shell", "code_execute_js", "code_execute_go",
        "code_execute_generic", "code_run_file",
        "execute_python", "execute_shell",
        # search
        "search_web", "web_search", "search",
        # project
        "project_scaffold", "project_tree", "project_run_command",
        # skill invocation（必须保留，确保 nested skill 调用不被裁剪）
        "invoke_skill",
    }
    # Also match by partial name for broader coverage (e.g. any tool ending in _write_file)
    _CORE_PARTIAL_KEYWORDS = {
        "write_file", "read_file", "list_dir", "list_directory",
        "create_dir", "run_shell", "execute_shell", "execute_python",
        "web_search",
    }

    core: List[BaseTool] = []
    rest: List[BaseTool] = []
    for t in tools:
        is_core = (
            t.name in _CORE_TOOL_KEYWORDS
            or any(partial in t.name for partial in _CORE_PARTIAL_KEYWORDS)
        )
        if is_core:
            core.append(t)
        else:
            rest.append(t)

    # 按优先级排序 rest
    rest.sort(key=_priority)

    # 合并：core 优先，然后从 rest 补充到 max_tools
    pruned = core[:max_tools]
    remaining_slots = max_tools - len(pruned)
    if remaining_slots > 0:
        pruned.extend(rest[:remaining_slots])

    logger.info(
        "Tool pruning: %d → %d tools (max_tools_for_local_model=%d)",
        len(tools), len(pruned), max_tools,
    )
    return pruned


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────

def make_debug_hooks(
    system_prompt: Optional[str] = None,
    max_iterations: int = 20,
) -> "Dict[str, Any]":  # type: ignore[type-arg]
    """
    构建并返回一套标准调试钩子字典。

    每个钩子均在运行时动态检查 debug 状态，确保会话中途通过 /debug 命令
    切换模式时即时生效。

    此函数被 create_agent_graph() 及 SkillExecutor 的子引擎共同使用，
    保证主 agent 与 skill 内部的 AGENT 步骤、LLM 步骤都能正确写入 debug 日志。

    Args:
        system_prompt:  用于展示在 "📥 Model Input" 面板的系统提示文本。
        max_iterations: 用于展示 iteration 进度信息。

    Returns:
        包含 "before_llm"、"after_llm"、"after_tool" 钩子的字典。
    """
    effective_prompt = system_prompt or ""
    _max_iterations = max_iterations

    def before_llm_hook(messages, iteration, messages_for_llm=None, step_context=None, **_):
        from local_agent.core.debug import (
            should_print_model_input,
            should_print_messages_state,
            should_print_debug,
            print_model_input,
            print_messages_state,
            print_iteration_info,
        )
        if should_print_messages_state():
            print_messages_state(messages, iteration)
        if should_print_model_input():
            display_msgs = messages_for_llm if messages_for_llm is not None else messages
            print_model_input(display_msgs, effective_prompt, step_context=step_context)
        if should_print_debug():
            print_iteration_info(iteration, _max_iterations)

    def after_llm_hook(message, step_context=None, **_):
        from local_agent.core.debug import (
            should_print_model_output,
            should_print_debug,
            print_model_output,
            print_tool_calls_detail,
        )
        if should_print_model_output():
            print_model_output(message, step_context=step_context)
        if should_print_debug() and hasattr(message, "tool_calls") and message.tool_calls:
            print_tool_calls_detail(message.tool_calls)

    def after_tool_hook(tool_call, tool_message, **_):
        from local_agent.core.debug import should_print_tool_calls, print_tool_call
        if not should_print_tool_calls():
            return
        content = tool_message.content if hasattr(tool_message, "content") else ""
        if isinstance(content, str) and (
            content.startswith("[Tool Error]") or "[状态: error]" in content
        ):
            print_tool_call(tool_call.name, tool_call.args, error=content)
        else:
            print_tool_call(tool_call.name, tool_call.args, tool_output=content)

    return {
        "before_llm": before_llm_hook,
        "after_llm": after_llm_hook,
        "after_tool": after_tool_hook,
    }


def create_agent_graph(
    llm,
    tools: List[BaseTool],
    system_prompt: Optional[str] = None,
    provider: Optional[str] = None,
) -> ReActEngine:
    """
    构建并返回一个 ReActEngine 实例。

    此函数签名与原 LangGraph 版本完全一致，调用方无需修改。

    Args:
        llm           : LLM 实例（BaseLLM 或兼容 bind_tools/invoke/stream 的对象）
        tools         : 工具列表
        system_prompt : 自定义系统提示；不传则使用默认
        provider      : LLM provider 名称（用于决定是否启用工具精简）

    Returns:
        ReActEngine 实例（支持 .invoke() / .stream()）
    """
    settings = get_settings()
    effective_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    # ── 提前初始化 file console，确保整个 session 使用同一个文件句柄 ──────
    # 必须在所有 print_* 调用之前执行，避免首次调用 _get_console() 以 "w" 模式
    # 打开文件后，后续某次重新初始化把已写入的内容清空。
    from local_agent.core.debug import init_file_console
    init_file_console()

    # ── 打印模型选择信息 ──────────────────────────────────────────────────
    from local_agent.core.debug import print_model_selection
    
    model_name = getattr(llm, "model", "unknown")
    effective_provider = provider or settings.llm_provider or "ollama"
    reason = "Default model from config"
    
    print_model_selection(model_name, effective_provider, reason)

    # 工具精简：本地模型（ollama）启用
    # 注意：精简不会移除任何 fs_* / shell_* / code_execute_* 核心工具，
    # 因为 _prune_tools 已将它们加入白名单保护。
    original_tool_count = len(tools)
    max_tools = settings.agent_max_tools_for_local_model
    if effective_provider == "ollama" and max_tools > 0:
        tools = _prune_tools(tools, max_tools)
        if len(tools) < original_tool_count:
            logger.info(
                "Tool pruning: %d → %d tools (max_tools_for_local_model=%d)",
                original_tool_count, len(tools), max_tools,
            )

    # ── 打印工具绑定信息 ──────────────────────────────────────────────────
    from local_agent.core.debug import print_tools_binding
    print_tools_binding(tools)

    # 构建调试钩子（使用公共工厂函数，供 SkillExecutor 复用）
    debug_hooks = make_debug_hooks(
        system_prompt=effective_prompt,
        max_iterations=settings.agent_max_iterations,
    )

    engine = ReActEngine(
        llm=llm,
        tools=tools,
        system_prompt=effective_prompt,
        max_iterations=settings.agent_max_iterations,
        debug_hooks=debug_hooks,
        tool_call_retry=settings.agent_tool_call_retry,
        max_tool_retry=settings.agent_max_tool_retry,
        # Context management parameters
        max_tool_result_length=settings.agent_max_tool_result_length,
        enable_tool_result_summarization=settings.agent_enable_tool_result_summarization,
        message_sliding_window=settings.agent_message_sliding_window,
        reuse_system_prompt=settings.agent_reuse_system_prompt,
    )

    logger.debug(
        "ReActEngine created (tools=%d, max_iterations=%d, tool_call_retry=%s, max_tool_retry=%d, "
        "context_mgmt: max_result=%d, summarize=%s, window=%d, reuse_sys=%s)",
        len(tools),
        settings.agent_max_iterations,
        settings.agent_tool_call_retry,
        settings.agent_max_tool_retry,
        settings.agent_max_tool_result_length,
        settings.agent_enable_tool_result_summarization,
        settings.agent_message_sliding_window,
        settings.agent_reuse_system_prompt,
    )
    return engine
