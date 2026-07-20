"""
LocalAgent CLI – main entry point.

Subcommands:
  chat      – interactive / single-shot chat
  skills    – list and inspect skills
  tools     – list and inspect tools
  models    – manage Ollama models
  server    – start the web server

Running ``la`` with no subcommand starts an interactive chat session.
"""
from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="local-agent",
    help="🤖 LocalAgent – A local AI agent framework powered by Ollama and LangGraph",
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)
console = Console()

# ── Sub-command registration ─────────────────────────────────────────────────
from local_agent.cli.commands import (  # noqa: E402  (after app creation)
    chat as chat_cmd,
    skills as skills_cmd,
    tools as tools_cmd,
    models as models_cmd,
    server as server_cmd,
    mcp as mcp_cmd,
)

app.add_typer(chat_cmd.app,   name="chat",   help="Chat with LocalAgent")
app.add_typer(skills_cmd.app, name="skills", help="Manage skills")
app.add_typer(tools_cmd.app,  name="tools",  help="Manage tools")
app.add_typer(models_cmd.app, name="models", help="Manage Ollama models")
app.add_typer(server_cmd.app, name="server", help="Start the web server")
app.add_typer(mcp_cmd.app,    name="mcp",    help="Manage MCP servers")


# ── Default callback (no subcommand → interactive chat) ─────────────────────
@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """LocalAgent – Local AI Agent Framework."""
    if ctx.invoked_subcommand is None:
        from local_agent.cli.commands.chat import interactive_chat
        interactive_chat()


if __name__ == "__main__":
    app()
