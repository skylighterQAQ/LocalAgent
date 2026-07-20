"""Tools management CLI commands"""
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Manage LocalAgent tools")
console = Console()


@app.command("list")
def tools_list(
    category: str = typer.Option("", "--category", "-c", help="Filter by category"),
):
    """List all registered tools"""
    from local_agent.tools.builtin import load_all_builtin_tools
    from local_agent.tools.registry import ToolRegistry

    load_all_builtin_tools()
    reg = ToolRegistry()
    tools = reg.get_tool_info()

    if category:
        tools = [t for t in tools if t.get("category") == category]

    if not tools:
        console.print("[yellow]No tools found.[/yellow]")
        return

    table = Table(title=f"Available Tools ({len(tools)})", header_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Category", style="green")
    table.add_column("Description", style="white")
    table.add_column("⚠ Confirm", style="yellow", justify="center")

    for t in sorted(tools, key=lambda x: (x.get("category", ""), x["name"])):
        table.add_row(
            t["name"],
            t.get("category", "general"),
            t["description"][:60] + ("..." if len(t["description"]) > 60 else ""),
            "Yes" if t.get("requires_confirmation") else "",
        )
    console.print(table)


@app.command("categories")
def tools_categories():
    """List all tool categories"""
    from local_agent.tools.builtin import load_all_builtin_tools
    from local_agent.tools.registry import ToolRegistry

    load_all_builtin_tools()
    reg = ToolRegistry()
    categories = reg.get_categories()

    console.print("[bold cyan]Tool Categories:[/bold cyan]")
    for cat in sorted(categories):
        tools = reg.get_by_category(cat)
        console.print(f"  [green]{cat}[/green] ({len(tools)} tools): {', '.join(t.name for t in tools)}")
