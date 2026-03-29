"""Rich terminal UI for OpenClaw."""
import sys
import os
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from rich import box
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.layout import Layout

console = Console()

BANNER = """
██╗      ██████╗  ██████╗ █████╗ ██╗      █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██║     ██╔═══██╗██╔════╝██╔══██╗██║     ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██║     ██║   ██║██║     ███████║██║     ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   
██║     ██║   ██║██║     ██╔══██║██║     ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   
███████╗╚██████╔╝╚██████╗██║  ██║███████╗██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   
╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   
"""


def print_banner(model: str, skills: list[dict]) -> None:
    console.print(Text(BANNER, style="bold cyan"))
    console.print(
        Panel(
            f"[bold green]Model:[/bold green] [cyan]{model}[/cyan]\n"
            f"[bold green]Loaded tools:[/bold green] {', '.join(s['name'] for s in skills) or 'none'}",
            title="[bold white]OpenClaw — Local AI Agent[/bold white]",
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()


def print_tools_table(tools: list[dict]) -> None:
    table = Table(box=box.ROUNDED, border_style="dim cyan", show_header=True)
    table.add_column("Tool", style="bold cyan")
    table.add_column("Description")
    table.add_column("Version", style="dim")

    for s in tools:
        table.add_row(s["name"], s["description"], s.get("version", "?"))

    console.print(table)


def print_skills_table(skills: list[dict]) -> None:
    table = Table(box=box.ROUNDED, border_style="dim cyan", show_header=True)
    table.add_column("Skill", style="bold cyan")
    table.add_column("Description")
    table.add_column("Version", style="dim")

    for s in skills:
        table.add_row(s["name"], s["description"], s.get("version", "?"))

    console.print(table)


def show_thinking() -> Live:
    """Return a Live spinner context."""
    return Live(
        Spinner("dots", text="[cyan]Thinking...[/cyan]"),
        console=console,
        refresh_per_second=12,
    )


def print_response(text: str) -> None:
    try:
        md = Markdown(text)
        console.print(Panel(md, border_style="green", padding=(0, 2)))
    except Exception:
        console.print(Panel(text, border_style="green", padding=(0, 2)))


def print_error(text: str) -> None:
    console.print(f"[bold red]Error:[/bold red] {text}")


def print_info(text: str) -> None:
    console.print(f"[dim]{text}[/dim]")


def print_tool_call(tool_name: str, args: dict) -> None:
    console.print(f"  [yellow]⚙ Tool:[/yellow] [bold]{tool_name}[/bold]", highlight=False)


COMMANDS = {
    "/help": "Show this help",
    "/tools": "List loaded tools",
    "/clear": "Clear conversation history",
    "/model": "Show current model",
    "/exit": "Exit OpenClaw",
    "/quit": "Exit OpenClaw",
}


def handle_command(cmd: str, context: dict) -> bool:
    """Handle a slash command. Returns True if should continue, False to exit."""
    cmd = cmd.strip().lower()

    if cmd in ("/exit", "/quit"):
        console.print("[cyan]Goodbye! 👋[/cyan]")
        return False

    elif cmd == "/help":
        table = Table(box=box.SIMPLE, show_header=False)
        for k, v in COMMANDS.items():
            table.add_row(f"[bold cyan]{k}[/bold cyan]", v)
        console.print(table)

    elif cmd == "/tools":
        from core.tool_base import get_registry
        print_skills_table(get_registry().list_tools())

    elif cmd == "/clear":
        context["history"].clear()
        console.print("[green]✓ Conversation history cleared.[/green]")

    elif cmd == "/model":
        from core.config_loader import get_config
        cfg = get_config()
        console.print(f"[cyan]Model:[/cyan] {cfg.ollama.model} @ {cfg.ollama.base_url}")

    else:
        console.print(f"[red]Unknown command: {cmd}. Type /help for commands.[/red]")

    return True


def run_interactive(graph, model: str, skills: list[dict]) -> None:
    """Run the interactive REPL."""
    print_banner(model, skills)
    console.print("[dim]Type your message, or /help for commands.[/dim]\n")

    context = {"history": []}

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]Goodbye! 👋[/cyan]")
            break

        if not user_input.strip():
            continue

        if user_input.startswith("/"):
            should_continue = handle_command(user_input, context)
            if not should_continue:
                break
            continue

        # Run agent
        from core.agent import run_agent
        try:
            with show_thinking():
                response = run_agent(graph, user_input, context["history"])
        except Exception as e:
            print_error(str(e))
            continue

        # Update history
        context["history"].append(HumanMessage(content=user_input))
        context["history"].append(AIMessage(content=response))

        console.print()
        console.print("[bold green]OpenClaw[/bold green]")
        print_response(response)
        console.print()
