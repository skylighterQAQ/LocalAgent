#!/usr/bin/env python3
"""
LocalAgent – Python 调用入口
==============================

直接运行:
    python main.py                          # 交互式聊天（默认）
    python main.py --mode chat              # 交互式聊天
    python main.py --mode once -q "你好"    # 单次问答
    python main.py --mode stream -q "介绍一下你自己"  # 流式输出
    python main.py --mode server            # 启动 Web API 服务器
    python main.py --model llama3.1:8b     # 指定 Ollama 模型
    python main.py --skill data_analyst    # 激活指定技能
    python main.py --no-mcp                # 禁用 MCP 工具加载

作为模块导入（API 方式）:
    from main import create_agent, quick_chat, stream_chat

MCP 使用示例:
    # 1. 复制 config/mcp.json.example 为 config/mcp.json 并编辑
    # 2. 启动 Agent，MCP 工具自动加载：
    agent = create_agent(model="qwen2.5:7b", load_mcp=True)
    print(agent.get_available_tools())    # 会包含 mcp 分类的工具

示例:
    agent = create_agent(model="qwen2.5:7b", skill="code_assistant")
    print(agent.chat("写一个快速排序"))

    for token in agent.stream("分析 README.md 文件"):
        print(token, end="", flush=True)
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from typing import Iterator, Optional

# Suppress LangGraph's pending deprecation warning about `allowed_objects`
# (This is a LangGraph internal warning, not actionable from user code)
warnings.filterwarnings(
    "ignore",
    message=".*allowed_objects.*",
)

# ── 日志配置 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,          # 默认只显示警告及以上；调试时改为 DEBUG
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("local_agent")


# ─────────────────────────────────────────────────────────────────────────────
# 公开 API（供其他 Python 脚本 import 使用）
# ─────────────────────────────────────────────────────────────────────────────

def create_agent(
    model: Optional[str] = None,
    skill: Optional[str] = None,
    extra_tool_dirs: Optional[list] = None,
    extra_skill_dirs: Optional[list] = None,
    load_mcp: bool = True,
    workspace=None,
):
    """
    创建并返回一个完全初始化的 LocalAgent 实例。

    参数:
        model:            Ollama 模型名称，如 "qwen2.5:7b"（默认从 config.yaml 读取）
        skill:            激活的技能名称，如 "code_assistant"、"data_analyst"
        extra_tool_dirs:  额外的工具目录列表（自动扫描加载）
        extra_skill_dirs: 额外的技能目录列表（自动扫描加载）
        load_mcp:         是否加载 mcp_servers.json 中配置的 MCP 工具（默认 True）
        workspace:        WorkspaceManager 实例（若提供则使用工作区 skill 和 model）

    返回:
        LocalAgent 实例

    示例::

        from main import create_agent

        agent = create_agent(model="qwen2.5:7b")
        response = agent.chat("列出当前目录下的 Python 文件")
        print(response)

        # 禁用 MCP（加速启动）
        agent = create_agent(load_mcp=False)
    """
    # Resolve skill / model from workspace if not explicitly provided
    if workspace is not None:
        if skill is None and workspace.skill:
            skill = workspace.skill
        if model is None and workspace.config.model:
            model = workspace.config.model

    from local_agent.core.agent import LocalAgent
    return LocalAgent.create(
        model=model,
        skill=skill,
        extra_tool_dirs=extra_tool_dirs or [],
        extra_skill_dirs=extra_skill_dirs or [],
        load_mcp=load_mcp,
    )


def quick_chat(
    message: str,
    model: Optional[str] = None,
    skill: Optional[str] = None,
) -> str:
    """
    最简单的一次性问答接口。每次调用都会新建 agent（适合脚本场景）。

    参数:
        message: 要发送的消息
        model:   Ollama 模型名称
        skill:   激活的技能

    返回:
        助手的回答字符串

    示例::

        from main import quick_chat

        answer = quick_chat("今天是几号？")
        print(answer)
    """
    agent = create_agent(model=model, skill=skill)
    return agent.chat(message)


def stream_chat(
    message: str,
    model: Optional[str] = None,
    skill: Optional[str] = None,
) -> Iterator[str]:
    """
    流式问答接口，逐 token yield 输出。

    参数:
        message: 要发送的消息
        model:   Ollama 模型名称
        skill:   激活的技能

    Yields:
        响应 token 字符串

    示例::

        from main import stream_chat

        for token in stream_chat("用 Python 写冒泡排序"):
            print(token, end="", flush=True)
        print()
    """
    agent = create_agent(model=model, skill=skill)
    yield from agent.stream(message)


# ─────────────────────────────────────────────────────────────────────────────
# 运行模式实现
# ─────────────────────────────────────────────────────────────────────────────

# Shared mutable reference so _handle_workspace_cmd can update the active workspace
_active_workspace = None


def _handle_workspace_cmd(arg: str, workspace, agent, skill, _print, _rich) -> None:
    """Handle /workspace sub-commands in interactive mode."""
    global _active_workspace
    parts = arg.strip().split(maxsplit=1) if arg.strip() else []
    sub = parts[0].lower() if parts else ""
    sub_arg = parts[1] if len(parts) > 1 else ""

    if not sub:
        # /workspace  → show info
        if workspace is None:
            _print("[yellow]No active workspace.[/yellow]\n"
                   "Use [bold]/workspace init <dir>[/bold] to create one."
                   if _rich else
                   "No active workspace.\nUse /workspace init <dir> to create one.")
        else:
            info = workspace.get_info()
            if _rich:
                from rich.table import Table
                from rich.console import Console as _Console
                tbl = Table(title=f"Workspace: {info['name']}", header_style="bold cyan")
                tbl.add_column("Property", style="cyan")
                tbl.add_column("Value")
                for k, v in info.items():
                    tbl.add_row(k, str(v))
                _Console().print(tbl)
            else:
                _print("\nWorkspace Info:")
                for k, v in info.items():
                    _print(f"  {k}: {v}")

    elif sub == "init":
        init_dir = sub_arg or "./workspace"
        from local_agent.cli.workspace import (
            WorkspaceManager,
            set_active_workspace,
            save_last_workspace,
        )
        ws = WorkspaceManager.init(
            directory=init_dir,
            skill=agent.active_skill,
        )
        ws.save()
        ws.ensure_default_dir()
        set_active_workspace(ws)
        save_last_workspace(ws.default_dir)
        _active_workspace = ws
        workspace = ws
        _print(
            f"[green]✓ Workspace initialized:[/green] {ws.name} at {ws.default_dir}"
            if _rich else
            f"✓ Workspace initialized: {ws.name} at {ws.default_dir}"
        )

    elif sub == "cd":
        if not sub_arg:
            _print("[yellow]Usage: /workspace cd <directory>[/yellow]" if _rich else
                   "Usage: /workspace cd <directory>")
            return
        from local_agent.cli.workspace import (
            set_active_workspace,
            save_last_workspace,
        )
        if workspace is None:
            from local_agent.cli.workspace import WorkspaceManager, WorkspaceConfig
            workspace = WorkspaceManager.from_config(WorkspaceConfig(
                name="default", default_dir=sub_arg, terminal_dir=sub_arg,
            ))
        else:
            workspace.set_default_dir(sub_arg)
            workspace.set_terminal_dir(sub_arg)
        workspace.ensure_default_dir()
        set_active_workspace(workspace)
        save_last_workspace(workspace.default_dir)
        _active_workspace = workspace
        _print(
            f"[green]✓ Workspace dir → {workspace.default_dir}[/green]"
            if _rich else
            f"✓ Workspace dir → {workspace.default_dir}"
        )

    elif sub == "skill":
        if not sub_arg:
            current = workspace.skill if workspace else "(none)"
            _print(f"[dim]Workspace skill: {current}[/dim]" if _rich else f"Workspace skill: {current}")
            return
        if workspace is None:
            from local_agent.cli.workspace import WorkspaceManager, WorkspaceConfig
            workspace = WorkspaceManager.from_config(WorkspaceConfig(name="default", skill=sub_arg))
        else:
            workspace.set_skill(sub_arg)
        _active_workspace = workspace
        agent.set_skill(sub_arg)
        _print(
            f"[green]✓ Workspace skill → {sub_arg}[/green]"
            if _rich else
            f"✓ Workspace skill → {sub_arg}"
        )

    elif sub == "save":
        if workspace is None:
            _print("[yellow]No active workspace to save.[/yellow]" if _rich else
                   "No active workspace to save.")
            return
        saved = workspace.save()
        _print(
            f"[green]✓ Workspace saved to {saved}[/green]"
            if _rich else
            f"✓ Workspace saved to {saved}"
        )

    else:
        _print(
            f"[yellow]Unknown workspace sub-command: '{sub}'[/yellow]\n"
            "Available: [bold]init[/bold] [bold]cd[/bold] [bold]skill[/bold] [bold]save[/bold]"
            if _rich else
            f"Unknown workspace sub-command: '{sub}'\nAvailable: init, cd, skill, save"
        )

    # Sync back _active_workspace so the caller can read it
    _active_workspace = workspace

def _run_interactive(model: Optional[str], skill: Optional[str], load_mcp: bool = True, workspace=None) -> None:
    """交互式多轮对话（持久 session，支持内置命令）"""
    try:
        from rich.console import Console
        from rich.markdown import Markdown  # noqa: F401
        console = Console()
        _rich = True
    except ImportError:
        _rich = False

    try:
        from prompt_toolkit import prompt as pt_prompt
        from prompt_toolkit.history import InMemoryHistory
        _pt_history = InMemoryHistory()
        _use_pt = True
    except ImportError:
        _use_pt = False

    def _print(text: str, style: str = "") -> None:
        if _rich:
            console.print(text, style=style)
        else:
            print(text)

    # Workspace: auto-load if not provided and auto_load is enabled
    if workspace is None:
        from local_agent.core.config import get_settings as _gs
        _settings = _gs()
        if _settings.workspace_auto_load:
            from local_agent.cli.workspace import find_workspace, load_last_workspace, WorkspaceManager
            workspace = find_workspace()
            # If nothing found in the cwd tree, try the last-used workspace
            if workspace is None:
                last = load_last_workspace()
                if last is not None:
                    ws_file = last / "workspace.yaml"
                    if ws_file.exists():
                        workspace = WorkspaceManager.load(str(ws_file))
                    if workspace is None:
                        # No yaml at that path – treat as in-memory workspace
                        from local_agent.cli.workspace import WorkspaceConfig
                        workspace = WorkspaceManager.from_config(
                            WorkspaceConfig(
                                name=last.name or "default",
                                default_dir=str(last),
                                terminal_dir=str(last),
                            )
                        )
        elif _settings.workspace_config:
            from local_agent.cli.workspace import WorkspaceManager
            workspace = WorkspaceManager.load(_settings.workspace_config)

    # Apply workspace overrides to skill / model
    if workspace is not None:
        if skill is None and workspace.skill:
            skill = workspace.skill
        if model is None and workspace.config.model:
            model = workspace.config.model

    ws_info = ""
    if workspace is not None:
        ws_info = f"  workspace={workspace.name}"
        workspace.ensure_default_dir()
        # Persist and broadcast the active workspace
        from local_agent.cli.workspace import set_active_workspace, save_last_workspace
        set_active_workspace(workspace)
        save_last_workspace(workspace.default_dir)
        # Keep the module-level reference in sync so command handlers see it
        global _active_workspace
        _active_workspace = workspace

    _print(
        f"\n[bold cyan]🤖 LocalAgent[/bold cyan]  "
        f"[dim]model={model or 'default'}  skill={skill or 'auto'}{ws_info}[/dim]\n"
        "Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit.\n"
        if _rich else
        f"\n🤖 LocalAgent  model={model or 'default'}  skill={skill or 'auto'}{ws_info}\n"
        "Type /help for commands, /quit to exit.\n"
    )

    # 初始化
    _print("[dim]Initializing agent...[/dim]" if _rich else "Initializing agent...")
    agent = create_agent(model=model, skill=skill, load_mcp=load_mcp, workspace=workspace)
    _print("[green]✓ Ready[/green]\n" if _rich else "✓ Ready\n")

    _HELP = """
Built-in commands:
  /help              Show this help
  /skill <name>      Switch active skill  (e.g. /skill code_assistant)
  /skills            List all available skills
  /tools             List all available tools (grouped by category)
  /mcp               List all loaded MCP servers and their tools
  /models            List all available models from all providers
  /clear             Clear conversation history
  /debug             Toggle debug mode
  /model <name>      Switch Ollama model
  /workspace         Show current workspace info
  /workspace init <dir>       Initialize workspace in <dir> (default: ./workspace)
  /workspace cd <dir>         Change workspace default directory
  /workspace skill <name>     Set workspace default skill
  /workspace save             Save current workspace config to workspace.yaml
  /quit  /exit       Exit
"""

    while True:
        try:
            if _use_pt:
                # 使用 prompt_toolkit，正确支持中文删除
                user_input = pt_prompt("\nYou: ", history=_pt_history)
            elif _rich:
                console.print("\n[bold green]You[/bold green]: ", end="")
                user_input = input()
            else:
                user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            _print("\n[dim]Bye![/dim]" if _rich else "\nBye!")
            break

        if not user_input.strip():
            continue

        # ── Built-in commands ────────────────────────────────────────────
        parts = user_input.strip().split(maxsplit=1)
        cmd, arg = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

        if cmd in ("/quit", "/exit"):
            _print("[dim]Bye![/dim]" if _rich else "Bye!")
            break

        elif cmd == "/help":
            _print(_HELP)
            continue

        elif cmd == "/clear":
            agent.reset_conversation()
            _print("[green]✓ Conversation cleared[/green]" if _rich else "✓ Conversation cleared")
            continue
        
        elif cmd == "/debug":
            from local_agent.core.config import get_settings
            settings = get_settings()
            settings.debug_print_mode = not settings.debug_print_mode
            status = "ON" if settings.debug_print_mode else "OFF"
            _print(f"[green]✓ Debug mode: {status}[/green]" if _rich else f"✓ Debug mode: {status}")
            continue

        elif cmd == "/workspace":
            _handle_workspace_cmd(arg, workspace, agent, skill, _print, _rich)
            # Re-read workspace reference since it may have been updated
            workspace = _active_workspace
            # Apply any skill change from workspace
            if workspace and workspace.skill and workspace.skill != agent.active_skill:
                agent.set_skill(workspace.skill)
                skill = workspace.skill
            continue

        elif cmd == "/skills":
            from local_agent.skills.registry import SkillRegistry
            skills = SkillRegistry().get_all_info()
            if _rich:
                from rich.table import Table
                tbl = Table(title="Available Skills", header_style="bold cyan")
                tbl.add_column("Name", style="cyan")
                tbl.add_column("Description")
                for s in skills:
                    marker = " ◀ active" if s["name"] == agent.active_skill else ""
                    tbl.add_row(s["name"] + marker, s["description"])
                console.print(tbl)
            else:
                for s in skills:
                    print(f"  {s['name']}: {s['description']}")
            continue

        elif cmd == "/skill":
            if arg:
                agent.set_skill(arg)
                skill = arg
                _print(f"[green]✓ Skill → {arg}[/green]" if _rich else f"✓ Skill → {arg}")
            else:
                _print(f"[dim]Current skill: {agent.active_skill or 'none'}[/dim]" if _rich
                       else f"Current skill: {agent.active_skill or 'none'}")
            continue

        elif cmd == "/tools":
            from local_agent.tools.registry import ToolRegistry
            reg = ToolRegistry()
            for cat in sorted(reg.get_categories()):
                names = ", ".join(t.name for t in reg.get_by_category(cat))
                _print(f"  [bold]{cat}[/bold]: {names}" if _rich else f"  {cat}: {names}")
            continue

        elif cmd == "/mcp":
            from local_agent.tools.registry import ToolRegistry
            reg = ToolRegistry()
            mcp_tools = reg.get_by_category("mcp")
            if not mcp_tools:
                _print("[yellow]No MCP tools loaded.[/yellow]\n"
                       "Create mcp_servers.json from mcp_servers.json.example to add MCP servers."
                       if _rich else
                       "No MCP tools loaded.\n"
                       "Create mcp_servers.json from mcp_servers.json.example to add MCP servers.")
            else:
                # Group by server prefix (before '__')
                servers: dict = {}
                for t in mcp_tools:
                    server = t.name.split("__")[0] if "__" in t.name else "unknown"
                    servers.setdefault(server, []).append(t)

                if _rich:
                    from rich.table import Table
                    tbl = Table(
                        title=f"Loaded MCP Tools ({len(mcp_tools)} total, {len(servers)} server(s))",
                        show_lines=True,
                        header_style="bold cyan",
                    )
                    tbl.add_column("Server", style="magenta", no_wrap=True)
                    tbl.add_column("Tool", style="cyan", no_wrap=True)
                    tbl.add_column("Description", style="white")
                    for srv_name, srv_tools in sorted(servers.items()):
                        for i, t in enumerate(srv_tools):
                            tool_short = t.name.split("__", 1)[-1] if "__" in t.name else t.name
                            desc = (t.description or "")[:80]
                            tbl.add_row(
                                srv_name if i == 0 else "",
                                tool_short,
                                desc,
                            )
                    console.print(tbl)
                else:
                    print(f"\nLoaded MCP Tools ({len(mcp_tools)} total, {len(servers)} server(s)):")
                    for srv_name, srv_tools in sorted(servers.items()):
                        print(f"  [{srv_name}]")
                        for t in srv_tools:
                            tool_short = t.name.split("__", 1)[-1] if "__" in t.name else t.name
                            desc = (t.description or "")[:60]
                            print(f"    {tool_short:30s}  {desc}")
            continue

        elif cmd == "/model":
            if arg:
                previous_model = agent.model
                previous_provider = agent.provider_type
                # Support "provider:model" format (e.g., /model wanqing:ep-xxx)
                if ":" in arg:
                    provider_name, model_name = arg.split(":", 1)
                    provider_name = provider_name.lower()
                    ok, err = agent.validate_model(model_name, provider=provider_name)
                    if ok:
                        agent.provider_type = provider_name
                        agent.model = model_name
                        # Recreate the LLM provider for the new provider type
                        if provider_name == "openai":
                            from local_agent.llm.openai_provider import OpenAIProvider
                            agent.llm_provider = OpenAIProvider(model=model_name)
                        elif provider_name == "wanqing":
                            from local_agent.llm.wanqing_provider import WanqingProvider
                            agent.llm_provider = WanqingProvider(model=model_name)
                        else:
                            from local_agent.llm.ollama_provider import OllamaProvider
                            agent.llm_provider = OllamaProvider(model=model_name)
                        agent._graph = None
                        _print(f"[green]✓ Provider → {provider_name}, Model → {model_name}[/green]" if _rich
                               else f"✓ Provider → {provider_name}, Model → {model_name}")
                    else:
                        _print(
                            f"[red]✗ Cannot switch to '{arg}': {err}[/red]\n"
                            f"[dim]Keeping current: {previous_provider}:{previous_model}[/dim]"
                            if _rich else
                            f"✗ Cannot switch to '{arg}': {err}\nKeeping current: {previous_provider}:{previous_model}"
                        )
                else:
                    ok, err = agent.validate_model(arg)
                    if ok:
                        agent.model = arg
                        agent._graph = None
                        _print(f"[green]✓ Model → {arg}[/green]" if _rich else f"✓ Model → {arg}")
                    else:
                        _print(
                            f"[red]✗ Cannot switch to '{arg}': {err}[/red]\n"
                            f"[dim]Keeping current model: {previous_model}[/dim]"
                            if _rich else
                            f"✗ Cannot switch to '{arg}': {err}\nKeeping current model: {previous_model}"
                        )
            else:
                _print(f"[dim]Current provider: {agent.provider_type}, model: {agent.model}[/dim]" if _rich
                       else f"Current provider: {agent.provider_type}, model: {agent.model}")
            continue

        elif cmd == "/models":
            from local_agent.llm.ollama_provider import OllamaProvider
            from local_agent.llm.openai_provider import OpenAIProvider
            from local_agent.llm.wanqing_provider import WanqingProvider
            from local_agent.core.config import get_settings
            
            settings = get_settings()
            current_model = agent.model
            current_provider = settings.llm_provider
            
            # Get Ollama models
            ollama_provider = OllamaProvider()
            ollama_available = ollama_provider.check_connection()
            ollama_models = ollama_provider.list_models() if ollama_available else []
            
            # Get OpenAI models
            openai_provider = OpenAIProvider()
            openai_models = openai_provider.list_models()
            
            # Get Wanqing models
            wanqing_provider = WanqingProvider()
            wanqing_available = wanqing_provider.check_connection()
            wanqing_models = wanqing_provider.list_models() if wanqing_available else []
            
            if _rich:
                from rich.table import Table
                tbl = Table(
                    title=f"Available Models (Current: {current_provider} / {current_model})",
                    header_style="bold cyan",
                )
                tbl.add_column("Provider", style="magenta", no_wrap=True)
                tbl.add_column("Model", style="cyan")
                tbl.add_column("Status", style="green")
                
                # Add Ollama models
                if ollama_available and ollama_models:
                    for i, m in enumerate(ollama_models):
                        status = "◀ active" if m == current_model and current_provider == "ollama" else ""
                        tbl.add_row(
                            "Ollama" if i == 0 else "",
                            m,
                            status,
                        )
                elif ollama_available:
                    tbl.add_row("Ollama", "[dim]No models found[/dim]", "[yellow]Run: ollama pull qwen2.5:7b[/yellow]")
                else:
                    tbl.add_row("Ollama", "[dim]Not available[/dim]", "[yellow]Start with: ollama serve[/yellow]")
                
                # Add OpenAI models
                for i, m in enumerate(openai_models):
                    status = "◀ active" if m == current_model and current_provider == "openai" else ""
                    tbl.add_row(
                        "OpenAI" if i == 0 else "",
                        m,
                        status,
                    )
                
                # Add Wanqing models
                if wanqing_available and wanqing_models:
                    for i, m in enumerate(wanqing_models):
                        status = "◀ active" if m == current_model and current_provider == "wanqing" else ""
                        tbl.add_row(
                            "Wanqing" if i == 0 else "",
                            m,
                            status,
                        )
                elif wanqing_available:
                    tbl.add_row("Wanqing", "[dim]No models configured[/dim]", "[yellow]Set WANQING_API_KEY[/yellow]")
                else:
                    tbl.add_row("Wanqing", "[dim]Not available[/dim]", "[yellow]Check config[/yellow]")
                
                console.print(tbl)
            else:
                print(f"\nAvailable Models (Current: {current_provider} / {current_model}):\n")
                
                # Ollama
                print("  [Ollama]")
                if ollama_available and ollama_models:
                    for m in ollama_models:
                        marker = " ◀ active" if m == current_model and current_provider == "ollama" else ""
                        print(f"    {m}{marker}")
                elif ollama_available:
                    print("    No models found. Run: ollama pull qwen2.5:7b")
                else:
                    print("    Not available. Start with: ollama serve")
                
                # OpenAI
                print("\n  [OpenAI]")
                for m in openai_models:
                    marker = " ◀ active" if m == current_model and current_provider == "openai" else ""
                    print(f"    {m}{marker}")
                
                # Wanqing
                print("\n  [Wanqing]")
                if wanqing_available and wanqing_models:
                    for m in wanqing_models:
                        marker = " ◀ active" if m == current_model and current_provider == "wanqing" else ""
                        print(f"    {m}{marker}")
                elif wanqing_available:
                    print("    No models configured. Set WANQING_API_KEY")
                else:
                    print("    Not available. Check configuration")
            continue

        # ── Send to agent (streaming) ────────────────────────────────────
        if _rich:
            console.print("\n[bold blue]Assistant[/bold blue] ", end="")
        else:
            print("\nAssistant: ", end="", flush=True)

        for chunk in agent.stream(user_input):
            # Tool indicators - only show in terminal when debug mode is ON and no log file is set
            if chunk.startswith("\n[Tool"):
                from local_agent.core.config import get_settings as _get_settings
                _s = _get_settings()
                _log_file = (_s.debug_log_file or "").strip()
                if _s.debug_print_mode and not _log_file:
                    _print(f"[dim yellow]{chunk.strip()}[/dim yellow]" if _rich else f"\n{chunk.strip()}")
                # If debug mode is off, or debug is going to a log file, skip silently
            else:
                # Normal response content
                if _rich:
                    console.print(chunk, end="", highlight=False)
                else:
                    print(chunk, end="", flush=True)
        print()


def _run_once(
    question: str,
    model: Optional[str],
    skill: Optional[str],
    stream: bool,
    load_mcp: bool = True,
) -> None:
    """单次问答模式"""
    agent = create_agent(model=model, skill=skill, load_mcp=load_mcp)

    if stream:
        for token in agent.stream(question):
            if not token.startswith("\n[Tool:"):
                print(token, end="", flush=True)
        print()
    else:
        response = agent.chat(question)
        print(response)


def _run_server(host: str, port: int) -> None:
    """启动 FastAPI Web 服务器"""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is not installed. Run: pip install uvicorn", file=sys.stderr)
        sys.exit(1)

    print(f"🚀 Starting LocalAgent server at http://{host}:{port}")
    print(f"   API Docs: http://{host}:{port}/docs")
    uvicorn.run("local_agent.api.app:app", host=host, port=port, reload=False)


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python main.py",
        description="🤖 LocalAgent – Local AI Agent powered by Ollama & LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # interactive chat
  python main.py --mode once -q "hello"       # single question
  python main.py --mode stream -q "hello"     # streaming answer
  python main.py --mode server                # start web server
  python main.py --model qwen2.5:14b          # choose model
  python main.py --skill data_analyst         # activate skill
  python main.py --workspace ./my_project     # use workspace directory
  python main.py --workspace-init ./my_proj --skill code_executor  # init workspace
  python main.py --list-skills                # list all skills
  python main.py --list-tools                 # list all tools
        """,
    )
    p.add_argument(
        "--mode",
        choices=["chat", "once", "stream", "server"],
        default="chat",
        help="Running mode (default: chat)",
    )
    p.add_argument(
        "-q", "--question",
        metavar="TEXT",
        help="Question / message for 'once' or 'stream' mode",
    )
    p.add_argument(
        "--model",
        metavar="NAME",
        default=None,
        help="Ollama model name (e.g. qwen2.5:7b, llama3.1:8b)",
    )
    p.add_argument(
        "--skill",
        metavar="NAME",
        default=None,
        help="Activate a specific skill (e.g. code_assistant, data_analyst)",
    )
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0, only used with --mode server)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080, only used with --mode server)",
    )
    p.add_argument(
        "--list-skills",
        action="store_true",
        help="List all available skills and exit",
    )
    p.add_argument(
        "--list-tools",
        action="store_true",
        help="List all available tools and exit",
    )
    p.add_argument(
        "--no-mcp",
        action="store_true",
        help="Disable MCP tool loading (faster startup)",
    )
    p.add_argument(
        "--workspace",
        metavar="DIR",
        default=None,
        help="Workspace directory path (overrides workspace.yaml auto-discovery)",
    )
    p.add_argument(
        "--workspace-init",
        metavar="DIR",
        nargs="?",
        const="./workspace",
        default=None,
        help=(
            "Initialize a workspace in DIR (default: ./workspace) and exit. "
            "Optionally specify --skill to set the workspace default skill."
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (print model/tool I/O)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose / debug logging",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("local_agent").setLevel(logging.DEBUG)
    
    # Set debug mode if requested
    if args.debug:
        from local_agent.core.config import get_settings
        settings = get_settings()
        settings.debug_print_mode = True

    # ── Info-only flags ───────────────────────────────────────────────────
    if args.list_skills:
        from local_agent.skills.loader import SkillLoader
        from local_agent.skills.registry import SkillRegistry
        SkillLoader().load_builtin_skills()
        skills = SkillRegistry().get_all_info()
        print(f"\nAvailable Skills ({len(skills)}):\n")
        for s in skills:
            tools_str = ", ".join(s.get("required_tools") or ["all tools"])
            print(f"  {s['name']:20s}  {s['description']}")
            print(f"  {'':20s}  tools: {tools_str}\n")
        return

    if args.list_tools:
        from local_agent.tools.builtin import load_all_builtin_tools
        from local_agent.tools.registry import ToolRegistry
        from local_agent.core.config import get_settings
        load_all_builtin_tools()
        # Also load MCP tools if configured
        settings = get_settings()
        if settings.mcp_enabled:
            try:
                from local_agent.mcp import MCPManager
                reg = ToolRegistry()
                manager = MCPManager.from_config_path(settings.mcp_config_path)
                for tool in manager.load_all():
                    reg.register(tool)
            except Exception:
                pass
        reg = ToolRegistry()
        total = len(reg.get_all())
        print(f"\nAvailable Tools ({total} total):\n")
        for cat in sorted(reg.get_categories()):
            tools = reg.get_by_category(cat)
            print(f"  [{cat}]")
            for t in tools:
                print(f"    {t.name:30s}  {t.description[:60]}")
            print()
        return

    load_mcp = not args.no_mcp

    # ── Workspace init ─────────────────────────────────────────────────────
    if getattr(args, "workspace_init", None) is not None:
        from local_agent.cli.workspace import WorkspaceManager, save_last_workspace, set_active_workspace
        init_dir = args.workspace_init
        ws = WorkspaceManager.init(
            directory=init_dir,
            name=getattr(args, "skill", None) or "default",
            skill=getattr(args, "skill", None),
        )
        saved = ws.save()
        ws.ensure_default_dir()
        set_active_workspace(ws)
        save_last_workspace(ws.default_dir)
        print(f"✓ Workspace initialized: {ws.name}")
        print(f"  Directory : {ws.default_dir}")
        print(f"  Skill     : {ws.skill or '(none)'}")
        print(f"  Config    : {saved}")
        return

    # ── Resolve workspace ──────────────────────────────────────────────────
    workspace = None
    if getattr(args, "workspace", None):
        from local_agent.cli.workspace import WorkspaceManager
        ws_cfg = str(args.workspace)
        # If a directory is passed, look for workspace.yaml inside it
        from pathlib import Path as _Path
        ws_dir = _Path(ws_cfg).expanduser().resolve()
        ws_file = ws_dir / "workspace.yaml" if ws_dir.is_dir() else ws_dir
        workspace = WorkspaceManager.load(str(ws_file))
        if workspace is None:
            # Treat as new in-memory workspace pointing at that directory
            from local_agent.cli.workspace import WorkspaceConfig
            workspace = WorkspaceManager.from_config(
                WorkspaceConfig(
                    name=ws_dir.name,
                    default_dir=str(ws_dir),
                    terminal_dir=str(ws_dir),
                    skill=getattr(args, "skill", None),
                )
            )
        # Ensure directory exists and persist as the last-used workspace
        workspace.ensure_default_dir()
        from local_agent.cli.workspace import set_active_workspace, save_last_workspace
        set_active_workspace(workspace)
        save_last_workspace(workspace.default_dir)

    # ── Run modes ─────────────────────────────────────────────────────────
    if args.mode == "server":
        _run_server(args.host, args.port)

    elif args.mode in ("once", "stream"):
        if not args.question:
            parser.error("--mode once/stream requires -q / --question TEXT")
        _run_once(
            question=args.question,
            model=args.model,
            skill=args.skill,
            stream=(args.mode == "stream"),
            load_mcp=load_mcp,
        )

    else:  # chat (default)
        _run_interactive(model=args.model, skill=args.skill, load_mcp=load_mcp, workspace=workspace)


if __name__ == "__main__":
    main()
