"""
CLI commands for MCP (Model Context Protocol) server management.

Usage:
    la mcp list                              – list all configured MCP servers
    la mcp add <name> --cmd "npx ..."        – add a stdio-based server
    la mcp add <name> --url http://...       – add an SSE-based server
    la mcp remove <name>                     – remove a server from config
    la mcp test <name>                       – connect and list available tools
    la mcp reload                            – reload all servers (CLI test only)
"""
from __future__ import annotations

import json
import sys
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Manage MCP (Model Context Protocol) servers")
console = Console()


def _load_config():
    from local_agent.core.config import get_settings
    from local_agent.mcp import MCPConfig
    settings = get_settings()
    return MCPConfig.load(settings.mcp_config_path), settings.mcp_config_path


def _save_config(cfg, path: str):
    from local_agent.mcp import MCPConfig
    cfg.config_path = __import__("pathlib").Path(path)
    cfg.save()


# ── list ──────────────────────────────────────────────────────────────────────

@app.command("list")
def list_servers():
    """List all configured MCP servers."""
    cfg, _ = _load_config()
    if not cfg.servers:
        console.print("[yellow]No MCP servers configured.[/yellow]")
        console.print(
            "Add one with: [cyan]la mcp add <name> --cmd 'npx -y @mcp/server-xxx'[/cyan]"
        )
        return

    table = Table(title="Configured MCP Servers", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Transport", style="magenta")
    table.add_column("Command / URL", style="white")
    table.add_column("Enabled", justify="center")
    table.add_column("Description")

    for name, srv in cfg.servers.items():
        cmd_or_url = (
            f"{srv.command} {' '.join(srv.args)}" if srv.command else (srv.url or "")
        ).strip()
        enabled_mark = "✅" if srv.enabled else "❌"
        table.add_row(name, srv.transport, cmd_or_url, enabled_mark, srv.description)

    console.print(table)


# ── add ───────────────────────────────────────────────────────────────────────

@app.command("add")
def add_server(
    name: str = typer.Argument(..., help="Unique server name"),
    cmd: Optional[str] = typer.Option(None, "--cmd", "-c", help="Command to run (stdio mode)"),
    args: Optional[List[str]] = typer.Option(None, "--arg", "-a", help="Arguments (repeatable)"),
    env_pairs: Optional[List[str]] = typer.Option(
        None, "--env", "-e", help="Environment variable KEY=VALUE (repeatable)"
    ),
    url: Optional[str] = typer.Option(None, "--url", "-u", help="SSE endpoint URL"),
    description: str = typer.Option("", "--desc", "-d", help="Human-readable description"),
    disabled: bool = typer.Option(False, "--disabled", help="Add server but keep it disabled"),
):
    """Add a new MCP server to the configuration."""
    if not cmd and not url:
        console.print("[red]Error:[/red] Provide either --cmd (stdio) or --url (sse).")
        raise typer.Exit(1)

    # Parse --env KEY=VALUE pairs
    env: dict = {}
    for pair in env_pairs or []:
        if "=" in pair:
            k, v = pair.split("=", 1)
            env[k.strip()] = v.strip()
        else:
            console.print(f"[yellow]Warning:[/yellow] Ignoring malformed env pair: {pair!r}")

    try:
        from local_agent.mcp import MCPServerConfig
        srv = MCPServerConfig(
            name=name,
            command=cmd,
            args=list(args or []),
            env=env,
            url=url,
            enabled=not disabled,
            description=description,
        )
    except Exception as exc:
        console.print(f"[red]Invalid config:[/red] {exc}")
        raise typer.Exit(1)

    cfg, path = _load_config()
    if name in cfg.servers:
        overwrite = typer.confirm(f"Server '{name}' already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit(0)

    cfg.add_server(srv)
    cfg.config_path = __import__("pathlib").Path(path)
    cfg.save()
    console.print(f"[green]✓[/green] Server [cyan]{name}[/cyan] added to [dim]{path}[/dim]")
    console.print(f"  Test it with: [cyan]la mcp test {name}[/cyan]")


# ── remove ────────────────────────────────────────────────────────────────────

@app.command("remove")
def remove_server(
    name: str = typer.Argument(..., help="Server name to remove"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove an MCP server from the configuration."""
    cfg, path = _load_config()
    if name not in cfg.servers:
        console.print(f"[red]Server '{name}' not found.[/red]")
        raise typer.Exit(1)

    if not yes:
        confirmed = typer.confirm(f"Remove server '{name}'?")
        if not confirmed:
            raise typer.Exit(0)

    cfg.remove_server(name)
    cfg.config_path = __import__("pathlib").Path(path)
    cfg.save()
    console.print(f"[green]✓[/green] Server [cyan]{name}[/cyan] removed.")


# ── test ──────────────────────────────────────────────────────────────────────

@app.command("test")
def test_server(
    name: str = typer.Argument(..., help="Server name to test"),
):
    """Connect to an MCP server and list its available tools."""
    cfg, _ = _load_config()
    if name not in cfg.servers:
        console.print(f"[red]Server '{name}' not found in config.[/red]")
        raise typer.Exit(1)

    srv = cfg.servers[name]
    console.print(f"Connecting to [cyan]{name}[/cyan] ({srv.transport})…")

    try:
        from local_agent.mcp import MCPClient
        client = MCPClient(srv)
        client.connect()
        tools = client.list_tools()
        client.close()
    except Exception as exc:
        console.print(f"[red]✗ Connection failed:[/red] {exc}")
        raise typer.Exit(1)

    if not tools:
        console.print(f"[yellow]Connected, but no tools reported by '{name}'.[/yellow]")
        return

    table = Table(title=f"Tools from '{name}' ({len(tools)} total)", show_lines=True)
    table.add_column("Tool Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    for t in tools:
        table.add_row(t["name"], (t.get("description") or "")[:120])

    console.print(table)
    console.print(f"\n[green]✓[/green] Successfully connected to [cyan]{name}[/cyan].")


# ── reload ────────────────────────────────────────────────────────────────────

@app.command("reload")
def reload_all():
    """Reload and test all enabled MCP servers (CLI diagnostic)."""
    cfg, _ = _load_config()
    enabled = cfg.get_enabled()

    if not enabled:
        console.print("[yellow]No enabled servers to reload.[/yellow]")
        return

    from local_agent.mcp import MCPManager
    from local_agent.core.config import get_settings

    settings = get_settings()
    manager = MCPManager.from_config_path(settings.mcp_config_path)
    tools = manager.load_all()

    for status in manager.status():
        icon = "✅" if status["connected"] else "❌"
        err = f" — [red]{status['error']}[/red]" if status["error"] else ""
        console.print(
            f"  {icon} [cyan]{status['name']}[/cyan] "
            f"({status['transport']}, {status['tool_count']} tools){err}"
        )

    manager.stop_all()
    console.print(f"\n[green]Total: {len(tools)} tool(s) loaded from {len(enabled)} server(s).[/green]")
