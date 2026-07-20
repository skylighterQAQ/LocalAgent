"""
Chat CLI Command - Interactive chat with LocalAgent
"""
import typer
from typing import Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text
from rich.spinner import Spinner
from rich.columns import Columns
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import InMemoryHistory

app = typer.Typer(help="Chat with LocalAgent")
console = Console()


def interactive_chat(
    model: str = "qwen2.5:7b",
    skill: Optional[str] = None,
    debug: bool = False,
):
    """Start an interactive chat session"""
    # Set debug mode if requested
    if debug:
        from local_agent.core.config import get_settings
        settings = get_settings()
        settings.debug_print_mode = True
    
    # Banner
    debug_status = " | Debug: ON" if debug else ""
    console.print(Panel.fit(
        "[bold cyan]🤖 LocalAgent[/bold cyan]\n"
        f"[dim]Model: {model} | Skill: {skill or 'Auto'}{debug_status} | Type 'help' for commands[/dim]",
        border_style="cyan"
    ))

    # Initialize agent
    with console.status("[cyan]Initializing agent...[/cyan]", spinner="dots"):
        try:
            from local_agent.core.agent import LocalAgent
            agent = LocalAgent.create(model=model, skill=skill)
        except Exception as e:
            console.print(f"[red]Failed to initialize agent: {e}[/red]")
            raise typer.Exit(1)

    console.print("[green]✓ Agent ready![/green]\n")

    # Input history for prompt_toolkit (supports up/down arrow key navigation)
    _history = InMemoryHistory()

    # Built-in commands
    builtin_cmds = {
        "/help": "Show this help",
        "/skills": "List available skills",
        "/skill <name>": "Switch active skill",
        "/tools": "List available tools",
        "/clear": "Clear conversation history",
        "/model [provider:]model": "Switch model or provider (e.g., /model gpt-4 or /model openai:gpt-4)",
        "/debug": "Toggle debug mode",
        "/quit": "Exit",
        "/exit": "Exit",
    }

    while True:
        try:
            # Use prompt_toolkit for proper Unicode/CJK (Chinese) input support
            user_input = pt_prompt("\nYou: ", history=_history)
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input.strip():
            continue

        # Handle built-in commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "/help":
                for c, desc in builtin_cmds.items():
                    console.print(f"  [cyan]{c}[/cyan]  {desc}")
            elif cmd == "/clear":
                agent.reset_conversation()
                console.print("[green]✓ Conversation cleared[/green]")
            elif cmd == "/debug":
                from local_agent.core.config import get_settings
                settings = get_settings()
                settings.debug_print_mode = not settings.debug_print_mode
                status = "ON" if settings.debug_print_mode else "OFF"
                console.print(f"[green]✓ Debug mode: {status}[/green]")
            elif cmd == "/skills":
                from local_agent.skills.registry import SkillRegistry
                reg = SkillRegistry()
                skills = reg.get_all_info()
                if skills:
                    for s in skills:
                        marker = " [bold green]← active[/bold green]" if s["name"] == skill else ""
                        console.print(f"  [cyan]{s['name']}[/cyan]: {s['description']}{marker}")
                else:
                    console.print("[dim]No skills loaded[/dim]")
            elif cmd == "/skill":
                if arg:
                    agent.set_skill(arg)
                    skill = arg
                    console.print(f"[green]✓ Switched to skill: {arg}[/green]")
                else:
                    console.print("[yellow]Usage: /skill <name>[/yellow]")
            elif cmd == "/tools":
                tools = agent.get_available_tools()
                categories = {}
                for t in tools:
                    cat = t.get("category", "general")
                    categories.setdefault(cat, []).append(t["name"])
                for cat, names in sorted(categories.items()):
                    console.print(f"  [bold]{cat}[/bold]: {', '.join(names)}")
            elif cmd == "/model":
                if arg:
                    previous_model = model
                    # Check if switching provider (e.g., /model openai:gpt-4)
                    if ":" in arg:
                        provider_name, model_name = arg.split(":", 1)
                        ok, err = agent.validate_model(model_name, provider=provider_name)
                        if ok:
                            agent.provider_type = provider_name
                            agent.model = model_name
                            model = f"{provider_name}:{model_name}"
                            agent._graph = None
                            console.print(f"[green]✓ Switched to {provider_name} with model: {model_name}[/green]")
                        else:
                            console.print(f"[red]✗ Cannot switch to '{arg}': {err}[/red]")
                            console.print(f"[dim]Keeping current model: {previous_model}[/dim]")
                    else:
                        ok, err = agent.validate_model(arg)
                        if ok:
                            agent.model = arg
                            model = arg
                            agent._graph = None
                            console.print(f"[green]✓ Switched to model: {arg}[/green]")
                        else:
                            console.print(f"[red]✗ Cannot switch to '{arg}': {err}[/red]")
                            console.print(f"[dim]Keeping current model: {previous_model}[/dim]")
                else:
                    from local_agent.core.config import get_settings
                    from rich.table import Table
                    settings = get_settings()
                    all_models = settings.get_all_configured_models()
                    table = Table(title="Configured Models (config.yaml)", header_style="bold cyan")
                    table.add_column("Provider", style="cyan")
                    table.add_column("Model", style="white")
                    table.add_column("", style="bold green")
                    for prov, models_list in all_models.items():
                        for m in models_list:
                            active_marker = "← active" if prov == agent.provider_type and m == agent.model else ""
                            table.add_row(prov, m, active_marker)
                    console.print(table)
                    console.print(f"[dim]Usage: /model <model_name> or /model <provider>:<model_name>[/dim]")
                    console.print(f"[dim]Example: /model wanqing:ep-xxx or /model qwen3:8b[/dim]")
            else:
                console.print(f"[yellow]Unknown command: {cmd}. Type /help for commands.[/yellow]")
            continue

        # Send to agent with streaming
        console.print("\n[bold blue]Assistant[/bold blue]", end=" ")
        response_text = ""
        try:
            for chunk in agent.stream(user_input):
                # Tool markers are only emitted when debug print_mode is ON
                if chunk.startswith("\n[Tool:") or chunk.strip() == "[Tool Completed]":
                    console.print(f"[dim yellow]{chunk.strip()}[/dim yellow]")
                else:
                    console.print(chunk, end="", highlight=False)
                    response_text += chunk
            console.print()  # newline at end
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


@app.command("run")
def chat_run(
    message: str = typer.Argument(help="Message to send to the agent"),
    model: str = typer.Option("qwen2.5:7b", "--model", "-m", help="Ollama model to use"),
    skill: Optional[str] = typer.Option(None, "--skill", "-s", help="Skill to activate"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream the response"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
):
    """Send a single message to LocalAgent and get a response"""
    # Set debug mode if requested
    if debug:
        from local_agent.core.config import get_settings
        settings = get_settings()
        settings.debug_print_mode = True
    
    with console.status("[cyan]Initializing...[/cyan]", spinner="dots"):
        from local_agent.core.agent import LocalAgent
        agent = LocalAgent.create(model=model, skill=skill)

    if stream:
        console.print("[bold blue]Assistant:[/bold blue] ", end="")
        for chunk in agent.stream(message):
            # Tool markers are only emitted when debug print_mode is ON; display them styled
            if chunk.startswith("\n[Tool:") or chunk.strip() == "[Tool Completed]":
                console.print(f"[dim yellow]{chunk.strip()}[/dim yellow]")
            else:
                console.print(chunk, end="", highlight=False)
        console.print()
    else:
        with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
            response = agent.chat(message)
        console.print("[bold blue]Assistant:[/bold blue]")
        console.print(Markdown(response))


@app.command("interactive")
def chat_interactive(
    model: str = typer.Option("qwen2.5:7b", "--model", "-m", help="Ollama model to use"),
    skill: Optional[str] = typer.Option(None, "--skill", "-s", help="Skill to activate"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
):
    """Start an interactive chat session"""
    interactive_chat(model=model, skill=skill, debug=debug)
