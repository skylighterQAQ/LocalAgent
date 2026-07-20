"""
Debug output utilities for LocalAgent.

Provides formatted printing functions for debugging model inputs, outputs, and tool calls.
Supports both terminal output and file output (configured via debug.log_file in config.yaml).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from local_agent.core.messages import BaseMessage, SystemMessage
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from local_agent.core.config import get_settings

# Default terminal console (always available)
console = Console()

# Lazily initialised file console; created on first use when log_file is set
_file_console: Optional[Console] = None
_file_console_path: str = ""  # Track which path the console was opened for
_file_handle = None           # Keep a reference to prevent GC from closing the handle


def _is_file_handle_open() -> bool:
    """Return True if the current file handle is still open and writable."""
    global _file_handle
    if _file_handle is None:
        return False
    try:
        return not _file_handle.closed
    except Exception:
        return False


def init_file_console() -> None:
    """
    Eagerly initialise the file console based on current settings.

    Call this **once** at agent startup (before any print_* function is called)
    so that the log file is created/truncated exactly once and every subsequent
    call to _get_console() reuses the same handle.

    If log_file is empty or debug is disabled, this is a no-op.
    """
    global _file_console, _file_console_path, _file_handle
    settings = get_settings()
    if not settings.debug_print_mode:
        return
    log_file = settings.debug_log_file.strip() if settings.debug_log_file else ""
    if not log_file:
        return
    # Already initialised for this path → skip (avoid re-opening with "w")
    if _file_console is not None and _file_console_path == log_file and _is_file_handle_open():
        return
    try:
        _file_handle = open(log_file, "w", encoding="utf-8")
        _file_console = Console(file=_file_handle, highlight=False, markup=True, width=200)
        _file_console_path = log_file
    except Exception as exc:
        console.print(f"[red]Cannot open debug log file '{log_file}': {exc}[/red]")


def _get_console() -> Console:
    """Return the active Console (file or terminal) based on current settings."""
    global _file_console, _file_console_path, _file_handle
    settings = get_settings()
    log_file = settings.debug_log_file.strip() if settings.debug_log_file else ""

    if log_file:
        # Re-use existing console if the path matches AND the file handle is still open.
        if (
            _file_console is not None
            and _file_console_path == log_file
            and _is_file_handle_open()
        ):
            return _file_console

        # Either first call, path changed, or handle was closed — (re-)create.
        # Use "w" only when the file doesn't yet exist or path changed; otherwise
        # append so we don't lose output from earlier in the same session.
        open_mode = "w" if (_file_console is None or _file_console_path != log_file) else "a"
        try:
            _file_handle = open(log_file, open_mode, encoding="utf-8")
            _file_console = Console(file=_file_handle, highlight=False, markup=True, width=200)
            _file_console_path = log_file
        except Exception as exc:
            console.print(f"[red]Cannot open debug log file '{log_file}': {exc}[/red]")
            return console
        return _file_console

    return console


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------

def should_print_debug() -> bool:
    """Check if debug print mode is enabled."""
    settings = get_settings()
    return settings.debug_print_mode


def should_print_model_input() -> bool:
    """Check if model input printing is enabled (part of print_llm group)."""
    settings = get_settings()
    return settings.debug_print_mode and settings.debug_print_llm


def should_print_model_output() -> bool:
    """Check if model output printing is enabled (part of print_llm group)."""
    settings = get_settings()
    return settings.debug_print_mode and settings.debug_print_llm


def should_print_tool_calls() -> bool:
    """Check if tool call printing is enabled (part of print_tools group)."""
    settings = get_settings()
    return settings.debug_print_mode and settings.debug_print_tools


def should_print_tools_binding() -> bool:
    """Check if tools binding printing is enabled (part of print_tools group)."""
    settings = get_settings()
    return settings.debug_print_mode and settings.debug_print_tools


def should_print_model_selection() -> bool:
    """Check if model selection printing is enabled (part of print_llm group)."""
    settings = get_settings()
    return settings.debug_print_mode and settings.debug_print_llm


def should_print_messages_state() -> bool:
    """Check if messages state printing is enabled (part of print_agent group)."""
    settings = get_settings()
    return settings.debug_print_mode and settings.debug_print_agent


def should_print_retry_guidance() -> bool:
    """Check if retry guidance printing is enabled (part of print_tools group)."""
    settings = get_settings()
    return settings.debug_print_mode and settings.debug_print_tools


def should_print_skill_activation() -> bool:
    """Check if skill activation printing is enabled (part of print_agent group)."""
    settings = get_settings()
    return settings.debug_print_mode and settings.debug_print_agent


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def format_message_for_debug(message: BaseMessage) -> Dict[str, Any]:
    """Format a message for debug output."""
    result: Dict[str, Any] = {
        "type": message.__class__.__name__,
        "content": message.content,
    }

    # Add tool calls if present — serialize ToolCall dataclass objects to plain dicts
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        serialized_tool_calls = []
        for tc in tool_calls:
            try:
                serialized_tool_calls.append({
                    "id": getattr(tc, "id", None),
                    "name": getattr(tc, "name", None),
                    "args": getattr(tc, "args", {}),
                })
            except Exception:
                serialized_tool_calls.append(str(tc))
        result["tool_calls"] = serialized_tool_calls

    # Add additional invocation metadata if present
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if additional_kwargs:
        result["additional_kwargs"] = additional_kwargs

    return result


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------


def print_model_input(messages: List[BaseMessage], system_prompt: Optional[str] = None, step_context: Optional[str] = None) -> None:
    """
    Print model input in a formatted panel.

    Args:
        messages: List of messages being sent to the model
        system_prompt: Optional system prompt being used
    """
    if not should_print_model_input():
        return

    try:
        import time
        from rich.console import Group

        # If a system_prompt is provided separately and the first message is a
        # SystemMessage with the same content, skip it to avoid printing twice.
        display_messages = list(messages)
        if system_prompt and display_messages and isinstance(display_messages[0], SystemMessage):
            if (display_messages[0].content or "").strip() == system_prompt.strip():
                display_messages = display_messages[1:]

        # Format messages for display
        formatted_messages = [format_message_for_debug(msg) for msg in display_messages]

        c = _get_console()
        renderables = []

        # ── 时间戳 ──────────────────────────────────────────────────────────
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        renderables.append(Text(f"⏱ LLM call at {ts}", style="bold yellow"))
        renderables.append(Rule(style="yellow dim"))

        if system_prompt:
            sp_label = Text("system_prompt: ", style="bold cyan")
            sp_body = Text(system_prompt, style="white", overflow="fold")
            renderables.append(sp_label)
            renderables.append(sp_body)
            renderables.append(Rule(style="cyan dim"))

        renderables.append(Text("messages:", style="bold cyan"))

        for msg in formatted_messages:
            # Render each message: metadata as JSON, content as wrapped text
            content_str: str = msg.get("content") or ""
            meta: Dict[str, Any] = {k: v for k, v in msg.items() if k != "content"}

            if meta:
                meta_json_str = json.dumps(meta, indent=2, ensure_ascii=False)
                renderables.append(Syntax(meta_json_str, "json", theme="monokai", line_numbers=False))

            if content_str:
                content_label = Text("  content: ", style="bold cyan")
                content_body = Text(content_str, style="white", overflow="fold")
                renderables.append(content_label)
                renderables.append(content_body)

            renderables.append(Rule(style="cyan dim"))

        title_suffix = f" — {step_context}" if step_context else ""
        c.print(Panel(
            Group(*renderables) if renderables else Text(""),
            title=f"[cyan]📥 Model Input{title_suffix}[/cyan]",
            border_style="cyan",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing model input: {e}[/red]")


def print_model_output(response: BaseMessage, step_context: Optional[str] = None) -> None:
    """
    Print model output in a formatted panel.

    Args:
        response: The AIMessage response from the model
        step_context: Optional step identifier for skill steps (e.g. "[step_1] 分析查询意图")
    """
    if not should_print_model_output():
        return

    try:
        import time
        output_data = format_message_for_debug(response)
        c = _get_console()

        # Separate content (may be long) from metadata (tool_calls, etc.)
        content_str: str = output_data.get("content") or ""
        meta: Dict[str, Any] = {k: v for k, v in output_data.items() if k != "content"}

        from rich.console import Group

        renderables = []

        # ── 时间戳 ──────────────────────────────────────────────────────────
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        renderables.append(Text(f"⏱ LLM response at {ts}", style="bold yellow"))
        renderables.append(Rule(style="yellow dim"))

        # Render metadata (type, tool_calls, …) as compact JSON
        if meta:
            meta_json_str = json.dumps(meta, indent=2, ensure_ascii=False)
            renderables.append(Syntax(meta_json_str, "json", theme="monokai", line_numbers=False))

        # Render content as plain wrapped text (preserves full content without truncation)
        if content_str:
            if meta:
                renderables.append(Rule(style="green dim"))
            content_label = Text("content: ", style="bold green")
            content_body = Text(content_str, style="white", overflow="fold")
            renderables.append(content_label)
            renderables.append(content_body)

        title_suffix = f" — {step_context}" if step_context else ""
        c.print(Panel(
            Group(*renderables) if renderables else Text(""),
            title=f"[green]📤 Model Output{title_suffix}[/green]",
            border_style="green",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing model output: {e}[/red]")


def print_tool_call(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_output: Optional[Any] = None,
    error: Optional[str] = None,
) -> None:
    """
    Print tool call details in a formatted panel.

    Args:
        tool_name: Name of the tool being called
        tool_input: Input arguments to the tool
        tool_output: Output/result from the tool (if available)
        error: Error message if tool execution failed
    """
    if not should_print_tool_calls():
        return

    try:
        call_data: Dict[str, Any] = {
            "tool": tool_name,
            "input": tool_input,
        }

        c = _get_console()

        if error:
            call_data["error"] = error
            c.print(Panel(
                JSON.from_data(call_data),
                title=f"[yellow]🔧 Tool Call: {tool_name}[/yellow]",
                border_style="yellow",
                expand=True,
            ))
        elif tool_output is not None:
            # Show input as JSON, output as plain text (may be long)
            from rich.console import Group
            input_json = json.dumps({"tool": tool_name, "input": tool_input}, indent=2, ensure_ascii=False)
            output_str = str(tool_output)
            renderables = [
                Syntax(input_json, "json", theme="monokai", line_numbers=False),
                Rule(style="yellow dim"),
                Text("output: ", style="bold yellow"),
                Text(output_str, style="white", overflow="fold"),
            ]
            c.print(Panel(
                Group(*renderables),
                title=f"[yellow]🔧 Tool Call: {tool_name}[/yellow]",
                border_style="yellow",
                expand=True,
            ))
        else:
            c.print(Panel(
                JSON.from_data(call_data),
                title=f"[yellow]🔧 Tool Call: {tool_name}[/yellow]",
                border_style="yellow",
                expand=True,
            ))
    except Exception as e:
        console.print(f"[red]Error printing tool call: {e}[/red]")


def print_debug_separator() -> None:
    """Print a separator line for debug output."""
    if should_print_debug():
        _get_console().print("[dim]" + "─" * 80 + "[/dim]")


def print_iteration_info(iteration: int, max_iterations: int) -> None:
    """
    Print iteration information.

    Args:
        iteration: Current iteration number
        max_iterations: Maximum allowed iterations
    """
    if should_print_debug():
        _get_console().print(f"[dim]Iteration: {iteration}/{max_iterations}[/dim]")


def print_tools_binding(tools: List[Any]) -> None:
    """
    Print the list of tools bound to the model with their schemas.

    Args:
        tools: List of BaseTool instances
    """
    if not should_print_tools_binding():
        return

    try:
        from rich.console import Group
        from rich.table import Table

        c = _get_console()
        
        # Create a table for tools
        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Tool Name", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Parameters", style="yellow")

        for idx, tool in enumerate(tools, 1):
            tool_name = getattr(tool, "name", str(tool))
            tool_desc = getattr(tool, "description", "N/A")
            
            # Extract parameters info
            params_info = []
            if hasattr(tool, "args_schema") and tool.args_schema:
                schema = tool.args_schema
                if hasattr(schema, "schema"):
                    schema_dict = schema.schema()
                    properties = schema_dict.get("properties", {})
                    required = schema_dict.get("required", [])
                    for param_name, param_info in properties.items():
                        param_type = param_info.get("type", "any")
                        is_required = param_name in required
                        req_marker = "*" if is_required else ""
                        params_info.append(f"{param_name}{req_marker}: {param_type}")
            
            params_str = "\n".join(params_info) if params_info else "N/A"
            
            table.add_row(
                str(idx),
                tool_name,
                tool_desc[:60] + "..." if len(tool_desc) > 60 else tool_desc,
                params_str
            )

        c.print(Panel(
            table,
            title=f"[cyan]🛠️  Tools Binding ({len(tools)} tools)[/cyan]",
            border_style="cyan",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing tools binding: {e}[/red]")


def print_model_selection(model_name: str, provider: str, reason: str = "") -> None:
    """
    Print model selection information.

    Args:
        model_name: Name of the selected model
        provider: Provider name (ollama, openai, etc.)
        reason: Reason for this selection
    """
    if not should_print_model_selection():
        return

    try:
        from rich.console import Group

        c = _get_console()
        
        renderables = [
            Text("Provider: ", style="bold cyan") + Text(provider, style="white"),
            Text("Model: ", style="bold cyan") + Text(model_name, style="white"),
        ]
        
        if reason:
            renderables.append(Text("Reason: ", style="bold cyan") + Text(reason, style="white"))

        c.print(Panel(
            Group(*renderables),
            title="[cyan]🔧 Model Selection[/cyan]",
            border_style="cyan",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing model selection: {e}[/red]")


def print_tool_calls_detail(tool_calls: List[Any]) -> None:
    """
    Print detailed structure of tool calls from AIMessage.

    Args:
        tool_calls: List of ToolCall objects
    """
    if not should_print_debug():
        return

    try:
        from rich.console import Group

        c = _get_console()
        renderables = []

        for idx, tc in enumerate(tool_calls):
            tc_id = getattr(tc, "id", "N/A")
            tc_name = getattr(tc, "name", "N/A")
            tc_args = getattr(tc, "args", {})

            renderables.append(Text(f"[{idx}] Tool Call ID: ", style="bold yellow") + Text(tc_id, style="white"))
            renderables.append(Text("    Tool Name: ", style="bold yellow") + Text(tc_name, style="cyan"))
            renderables.append(Text("    Arguments:", style="bold yellow"))
            
            args_json = json.dumps(tc_args, indent=6, ensure_ascii=False)
            renderables.append(Syntax(args_json, "json", theme="monokai", line_numbers=False))
            
            if idx < len(tool_calls) - 1:
                renderables.append(Text(""))  # Empty line between calls

        c.print(Panel(
            Group(*renderables),
            title=f"[yellow]🎯 Tool Calls Detail ({len(tool_calls)} call{'s' if len(tool_calls) > 1 else ''})[/yellow]",
            border_style="yellow",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing tool calls detail: {e}[/red]")


def print_messages_state(messages: List[BaseMessage], iteration: int) -> None:
    """
    Print the complete state of the messages list.

    Args:
        messages: List of messages
        iteration: Current iteration number
    """
    if not should_print_messages_state():
        return

    try:
        from rich.console import Group
        from rich.table import Table

        c = _get_console()
        
        # Create a table for messages
        table = Table(show_header=True, header_style="bold magenta", expand=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Type", style="magenta", width=20)
        table.add_column("Content Preview", style="white")
        table.add_column("Extra", style="yellow", width=15)

        for idx, msg in enumerate(messages):
            msg_type = msg.__class__.__name__
            content = getattr(msg, "content", "")
            content_preview = str(content)
            
            # Check for extra info
            extra_info: list[str] = []
            if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls", None):
                extra_info.append(f"{len(getattr(msg, 'tool_calls'))} tool call(s)")
            if hasattr(msg, "tool_call_id"):
                extra_info.append("tool result")
            
            extra_str = "\n".join(extra_info) if extra_info else ""
            
            table.add_row(
                str(idx),
                msg_type,
                content_preview,
                extra_str
            )

        summary = Text(f"Total messages: {len(messages)}", style="bold magenta")

        c.print(Panel(
            Group(summary, Rule(style="magenta dim"), table),
            title=f"[magenta]📊 Messages State (Iteration {iteration})[/magenta]",
            border_style="magenta",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing messages state: {e}[/red]")


def print_skill_activation(skill_name: Optional[str], reason: str = "") -> None:
    """
    Print activated skill information.

    Args:
        skill_name: Name of the activated skill, or None if no skill selected
        reason: Optional reason / description for the activation
    """
    if not should_print_skill_activation():
        return

    try:
        from rich.console import Group

        c = _get_console()

        if skill_name:
            renderables = [
                Text("Skill: ", style="bold magenta") + Text(skill_name, style="white"),
            ]
            if reason:
                renderables.append(Text("Reason: ", style="bold magenta") + Text(reason, style="white"))
            title = "[magenta]🎯 Skill Activated[/magenta]"
        else:
            renderables = [
                Text("No skill activated — using default agent behaviour.", style="dim"),
            ]
            if reason:
                renderables.append(Text("Reason: ", style="bold magenta") + Text(reason, style="white"))
            title = "[magenta]🎯 Skill[/magenta]"

        c.print(Panel(
            Group(*renderables),
            title=title,
            border_style="magenta",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing skill activation: {e}[/red]")


def print_retry_guidance(retry_count: int, reason: str, guidance_msg: str) -> None:
    """
    Print tool call retry guidance information.

    Args:
        retry_count: Current retry attempt number
        reason: Reason for retry
        guidance_msg: The guidance message being sent
    """
    if not should_print_retry_guidance():
        return

    try:
        from rich.console import Group

        c = _get_console()
        
        renderables = [
            Text("Retry Attempt: ", style="bold yellow") + Text(f"#{retry_count}", style="white"),
            Text("Reason: ", style="bold yellow") + Text(reason, style="white"),
            Rule(style="yellow dim"),
            Text("Guidance Message:", style="bold yellow"),
            Text(guidance_msg, style="white", overflow="fold"),
        ]

        c.print(Panel(
            Group(*renderables),
            title="[yellow]🔄 Tool Call Retry Guidance[/yellow]",
            border_style="yellow",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing retry guidance: {e}[/red]")


def print_task_plan(plan: Any) -> None:
    """
    Print task plan summary in print mode.

    Args:
        plan: TaskPlan object with overall_strategy and steps attributes
    """
    if not should_print_debug():
        return

    try:
        from rich.console import Group
        from rich.table import Table

        c = _get_console()

        strategy = getattr(plan, "overall_strategy", "")
        steps = getattr(plan, "steps", [])

        renderables = [
            Text("Strategy: ", style="bold cyan") + Text(strategy, style="white"),
        ]

        if steps:
            table = Table(show_header=True, header_style="bold cyan", expand=True)
            table.add_column("#", style="dim", width=4)
            table.add_column("Step", style="white")
            table.add_column("Skill", style="magenta", width=20)

            for step in steps:
                step_id = str(getattr(step, "step_id", ""))
                title = getattr(step, "title", "")
                skill = getattr(step, "skill", "") or "(default)"
                table.add_row(step_id, title, skill)

            renderables.append(Rule(style="cyan dim"))
            renderables.append(table)

        c.print(Panel(
            Group(*renderables) if renderables else Text(""),
            title=f"[cyan]📋 Task Plan ({len(steps)} step{'s' if len(steps) != 1 else ''})[/cyan]",
            border_style="cyan",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing task plan: {e}[/red]")


def print_task_step(step_id: str, title: str, skill: Optional[str] = None) -> None:
    """
    Print current task step information in print mode.

    Args:
        step_id: Step identifier (e.g. "1", "2")
        title: Step title / description
        skill: Optional skill name being activated for this step
    """
    if not should_print_debug():
        return

    try:
        from rich.console import Group

        c = _get_console()

        renderables = [
            Text(f"Step {step_id}: ", style="bold blue") + Text(title, style="white"),
        ]
        if skill:
            renderables.append(
                Text("Skill: ", style="bold magenta") + Text(skill, style="white")
            )

        c.print(Panel(
            Group(*renderables),
            title=f"[blue]▶ Executing Step {step_id}[/blue]",
            border_style="blue",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing task step: {e}[/red]")


def print_tool_announce(tool_name: str) -> None:
    """
    Print a tool invocation announcement in print mode (replaces yield token in stream).

    Args:
        tool_name: Name of the tool being invoked
    """
    if not should_print_debug():
        return

    try:
        c = _get_console()
        c.print(f"[bold yellow]⚙ Calling tool:[/bold yellow] [cyan]{tool_name}[/cyan]")
    except Exception as e:
        console.print(f"[red]Error printing tool announce: {e}[/red]")


def print_tool_completed() -> None:
    """Print a tool-completed notification in print mode."""
    if not should_print_debug():
        return

    try:
        c = _get_console()
        c.print("[dim yellow]✓ Tool completed[/dim yellow]")
    except Exception as e:
        console.print(f"[red]Error printing tool completed: {e}[/red]")


# ---------------------------------------------------------------------------
# Prompt Context Registry print helpers
# ---------------------------------------------------------------------------

def should_print_prompt_registry() -> bool:
    """Check if prompt context registry operations should be printed."""
    settings = get_settings()
    return settings.debug_print_mode and settings.debug_print_agent


def print_prompt_context_save(
    invocation_id: str,
    skill_name: str,
    task: str,
    messages_count: int,
) -> None:
    """
    Print a prompt context save event (before invoking a sub-skill).

    Triggered when ReActEngine saves the current messages snapshot to the
    PromptContextRegistry before calling invoke_skill.

    Args:
        invocation_id:  UUID identifying this specific sub-skill invocation
        skill_name:     Name of the sub-skill being invoked
        task:           Task description passed to the sub-skill
        messages_count: Number of messages in the saved snapshot
    """
    if not should_print_prompt_registry():
        return

    try:
        from rich.console import Group

        c = _get_console()

        task_preview = task[:120] + "..." if len(task) > 120 else task

        renderables = [
            Text("Invocation ID: ", style="bold orange1") + Text(invocation_id, style="white"),
            Text("Sub-Skill:     ", style="bold orange1") + Text(skill_name, style="cyan"),
            Text("Task:          ", style="bold orange1") + Text(task_preview, style="white", overflow="fold"),
            Text("Messages Saved:", style="bold orange1") + Text(str(messages_count), style="white"),
        ]

        c.print(Panel(
            Group(*renderables),
            title="[orange1]📦 Prompt Context Saved[/orange1]",
            border_style="orange1",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing prompt context save: {e}[/red]")


def print_prompt_context_restore(
    invocation_id: str,
    skill_name: str,
    result_len: int,
) -> None:
    """
    Print a prompt context restore event (after sub-skill returns).

    Triggered when ReActEngine retrieves the parent context from the registry
    using the invocation_id after invoke_skill completes.

    Args:
        invocation_id: UUID used to look up the saved context
        skill_name:    Name of the sub-skill that just finished
        result_len:    Length (in chars) of the sub-skill's result content
    """
    if not should_print_prompt_registry():
        return

    try:
        from rich.console import Group

        c = _get_console()

        renderables = [
            Text("Invocation ID: ", style="bold orange3") + Text(invocation_id, style="white"),
            Text("Sub-Skill:     ", style="bold orange3") + Text(skill_name, style="cyan"),
            Text("Result Length: ", style="bold orange3") + Text(f"{result_len} chars", style="white"),
        ]

        c.print(Panel(
            Group(*renderables),
            title="[orange3]📤 Prompt Context Restored[/orange3]",
            border_style="orange3",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing prompt context restore: {e}[/red]")


def print_prompt_registry_state(entries: List[Any]) -> None:
    """
    Print the current state of the PromptContextRegistry as a table.

    Shows all pending (not-yet-retrieved) entries, useful for debugging
    nested skill calls that haven't completed yet.

    Args:
        entries: List of PromptContextEntry objects from the registry
    """
    if not should_print_prompt_registry():
        return

    try:
        from rich.console import Group
        from rich.table import Table
        import time as _time

        c = _get_console()

        if not entries:
            c.print(Panel(
                Text("Registry is empty — no pending sub-skill invocations.", style="dim"),
                title="[orange1]📋 Prompt Registry State[/orange1]",
                border_style="orange1",
                expand=True,
            ))
            return

        table = Table(show_header=True, header_style="bold orange1", expand=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Invocation ID", style="white", width=38)
        table.add_column("Skill", style="cyan", width=20)
        table.add_column("Task Preview", style="white")
        table.add_column("Messages", style="yellow", width=10)
        table.add_column("Age (s)", style="dim", width=8)

        now = _time.time()
        for idx, entry in enumerate(entries, 1):
            inv_id = getattr(entry, "invocation_id", "")
            skill = getattr(entry, "skill_name", "")
            task = getattr(entry, "task", "")
            msgs = getattr(entry, "saved_messages", [])
            ts = getattr(entry, "timestamp", now)

            task_preview = task[:60] + "..." if len(task) > 60 else task
            age = f"{now - ts:.1f}"

            table.add_row(str(idx), inv_id, skill, task_preview, str(len(msgs)), age)

        summary = Text(f"Pending entries: {len(entries)}", style="bold orange1")

        c.print(Panel(
            Group(summary, Rule(style="orange1 dim"), table),
            title="[orange1]📋 Prompt Registry State[/orange1]",
            border_style="orange1",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing prompt registry state: {e}[/red]")


# ---------------------------------------------------------------------------
# Skill step / invocation trace helpers
# ---------------------------------------------------------------------------

def should_print_skill_trace() -> bool:
    """Check if skill step / invocation tracing is enabled."""
    settings = get_settings()
    return settings.debug_print_mode


def print_skill_step_start(
    skill_name: str,
    step_id: str,
    step_name: str,
    step_type: str,
    step_input: "dict[str, Any]",
) -> None:
    """
    Print the start of a skill step execution.

    Called unconditionally at the top of SkillExecutor.execute() loop,
    before _dispatch() is called – regardless of step type (LLM/AGENT/SKILL/TOOL).

    Args:
        skill_name:  Parent skill name
        step_id:     Step identifier (e.g. "step_1")
        step_name:   Human-readable step name
        step_type:   Step type string ("llm" / "agent" / "skill" / "tool")
        step_input:  Resolved input dict for this step
    """
    if not should_print_skill_trace():
        return

    try:
        from rich.console import Group

        c = _get_console()

        type_color = {
            "llm": "cyan",
            "agent": "blue",
            "skill": "magenta",
            "tool": "yellow",
        }.get(step_type.lower(), "white")

        input_preview = json.dumps(
            {k: (str(v)[:200] + "..." if isinstance(v, str) and len(str(v)) > 200 else v)
             for k, v in step_input.items()},
            ensure_ascii=False, indent=2,
        )

        renderables = [
            Text("Skill:     ", style="bold white") + Text(skill_name, style="magenta"),
            Text("Step:      ", style="bold white") + Text(f"[{step_id}] {step_name}", style="white"),
            Text("Type:      ", style="bold white") + Text(step_type.upper(), style=type_color),
            Rule(style="dim"),
            Text("Input:", style="bold white"),
            Syntax(input_preview, "json", theme="monokai", line_numbers=False),
        ]

        c.print(Panel(
            Group(*renderables),
            title=f"[{type_color}]▶ Skill Step: [{step_id}] {step_name}[/{type_color}]",
            border_style=type_color,
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing skill step start: {e}[/red]")


def print_skill_invocation_input(
    parent_skill: str,
    step_id: str,
    nested_skill: str,
    step_input: "dict[str, Any]",
) -> None:
    """
    Print the input when a SKILL step invokes a nested sub-skill.

    Called in SkillExecutor._run_skill_step() just before the nested
    skill execution begins.

    Args:
        parent_skill:  Name of the skill that owns this step
        step_id:       Step identifier
        nested_skill:  Name of the sub-skill being called
        step_input:    Input dict passed to the sub-skill
    """
    if not should_print_skill_trace():
        return

    try:
        from rich.console import Group

        c = _get_console()

        input_preview = json.dumps(
            {k: (str(v)[:300] + "..." if isinstance(v, str) and len(str(v)) > 300 else v)
             for k, v in step_input.items()},
            ensure_ascii=False, indent=2,
        )

        renderables = [
            Text("Parent Skill:  ", style="bold orange1") + Text(parent_skill, style="magenta"),
            Text("Step:          ", style="bold orange1") + Text(step_id, style="white"),
            Text("→ Sub-Skill:   ", style="bold orange1") + Text(nested_skill, style="cyan"),
            Rule(style="orange1 dim"),
            Text("Input to sub-skill:", style="bold orange1"),
            Syntax(input_preview, "json", theme="monokai", line_numbers=False),
        ]

        c.print(Panel(
            Group(*renderables),
            title="[orange1]🔀 Invoking Sub-Skill[/orange1]",
            border_style="orange1",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing skill invocation input: {e}[/red]")


def print_skill_invocation_output(
    parent_skill: str,
    step_id: str,
    nested_skill: str,
    result: "dict[str, Any]",
) -> None:
    """
    Print the output returned from a nested sub-skill invocation.

    Called in SkillExecutor._run_skill_step() right after the nested
    skill execution completes (success path).

    Args:
        parent_skill:  Name of the skill that owns this step
        step_id:       Step identifier
        nested_skill:  Name of the sub-skill that just finished
        result:        Result dict returned by the sub-skill
    """
    if not should_print_skill_trace():
        return

    try:
        from rich.console import Group

        c = _get_console()

        # Summarise result: show each key with a value preview
        result_parts: list[str] = []
        for k, v in result.items():
            v_str = str(v)
            preview = v_str[:300] + "..." if len(v_str) > 300 else v_str
            result_parts.append(f"{k}: {preview}")
        result_text = "\n".join(result_parts) if result_parts else "(empty)"

        renderables = [
            Text("Parent Skill:  ", style="bold green") + Text(parent_skill, style="magenta"),
            Text("Step:          ", style="bold green") + Text(step_id, style="white"),
            Text("← Sub-Skill:   ", style="bold green") + Text(nested_skill, style="cyan"),
            Rule(style="green dim"),
            Text("Output from sub-skill:", style="bold green"),
            Text(result_text, style="white", overflow="fold"),
        ]

        c.print(Panel(
            Group(*renderables),
            title="[green]✅ Sub-Skill Output[/green]",
            border_style="green",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing skill invocation output: {e}[/red]")


def print_tool_step_call(
    step_id: str,
    step_name: str,
    tool_name: str,
    tool_input: "dict[str, Any]",
    tool_output: Any,
) -> None:
    """
    Print a TOOL-type step's tool call and its result.

    Called from SkillExecutor._run_tool_step() after the tool completes,
    so both input and output are shown together.

    Args:
        step_id:     Step identifier
        step_name:   Human-readable step name
        tool_name:   Tool that was called
        tool_input:  Arguments passed to the tool
        tool_output: Raw output returned by the tool
    """
    if not should_print_skill_trace():
        return

    try:
        from rich.console import Group

        c = _get_console()

        input_preview = json.dumps(
            {k: (str(v)[:200] + "..." if isinstance(v, str) and len(str(v)) > 200 else v)
             for k, v in tool_input.items()},
            ensure_ascii=False, indent=2,
        )
        output_str = str(tool_output)
        output_preview = output_str[:400] + "..." if len(output_str) > 400 else output_str

        renderables = [
            Text("Step:     ", style="bold yellow") + Text(f"[{step_id}] {step_name}", style="white"),
            Text("Tool:     ", style="bold yellow") + Text(tool_name, style="cyan"),
            Rule(style="yellow dim"),
            Text("Input:", style="bold yellow"),
            Syntax(input_preview, "json", theme="monokai", line_numbers=False),
            Rule(style="yellow dim"),
            Text("Output:", style="bold yellow"),
            Text(output_preview, style="white", overflow="fold"),
        ]

        c.print(Panel(
            Group(*renderables),
            title=f"[yellow]🔧 Tool Step: {tool_name}[/yellow]",
            border_style="yellow",
            expand=True,
        ))
    except Exception as e:
        console.print(f"[red]Error printing tool step call: {e}[/red]")
