"""Models management CLI commands"""
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Manage and inspect configured LLM models")
console = Console()


@app.command("list")
def models_list():
    """List all models configured in config.yaml (grouped by provider)"""
    from local_agent.core.config import get_settings
    settings = get_settings()

    all_models = settings.get_all_configured_models()
    if not all_models:
        console.print("[yellow]No models configured. Edit config.yaml to add models.[/yellow]")
        return

    table = Table(
        title="Configured Models (config.yaml)",
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Model", style="white")
    table.add_column("Default", style="dim green")

    for provider, models in all_models.items():
        default_model = getattr(settings, f"{provider}_default_model", "")
        for m in models:
            default_marker = "✓ default" if m == default_model else ""
            table.add_row(provider, m, default_marker)

    console.print(table)
    console.print(f"\n[dim]Active provider: [bold]{settings.llm_provider or 'auto'}[/bold][/dim]")
    console.print("[dim]To add models: edit the 'models' list under each provider in config.yaml[/dim]")


@app.command("check")
def models_check():
    """Check connection status for all configured providers"""
    from local_agent.core.config import get_settings
    settings = get_settings()

    console.print("[bold]Provider Connection Status[/bold]\n")

    # Ollama
    from local_agent.llm.providers.ollama import OllamaProvider
    ollama = OllamaProvider()
    if ollama.check_connection():
        models = ollama.list_models()
        configured = settings.get_configured_models("ollama")
        console.print(f"[green]✓ ollama[/green] — {settings.ollama_base_url}")
        console.print(f"  Configured models: {', '.join(configured)}")
        # Highlight which configured models are actually available locally
        available_set = set(models)
        for m in configured:
            status = "[green]available[/green]" if m in available_set else "[yellow]not pulled[/yellow]"
            console.print(f"    {m}  [{status}]")
    else:
        console.print(f"[red]✗ ollama[/red] — Cannot connect to {settings.ollama_base_url}")
        console.print("[dim]  Start with: ollama serve[/dim]")

    # OpenAI
    if settings.openai_api_key:
        configured = settings.get_configured_models("openai")
        console.print(f"\n[green]✓ openai[/green] — API key configured")
        console.print(f"  Configured models: {', '.join(configured)}")
    else:
        console.print(f"\n[dim]- openai — no API key (set openai.api_key in config.yaml)[/dim]")

    # Wanqing
    if settings.wanqing_api_key:
        configured = settings.get_configured_models("wanqing")
        console.print(f"\n[green]✓ wanqing[/green] — API key configured")
        console.print(f"  Configured models: {', '.join(configured)}")
    else:
        console.print(f"\n[dim]- wanqing — no API key (set wanqing.api_key in config.yaml)[/dim]")

    # Claude Code
    from local_agent.llm.providers.claude_code import ClaudeCodeProvider
    claude = ClaudeCodeProvider()
    if claude.check_connection():
        configured = settings.get_configured_models("claude_code")
        console.print(f"\n[green]✓ claude_code[/green] — API key configured")
        console.print(f"  Configured models: {', '.join(configured)}")
    else:
        console.print(f"\n[dim]- claude_code — no API key (set claude_code.api_key in config.yaml)[/dim]")
