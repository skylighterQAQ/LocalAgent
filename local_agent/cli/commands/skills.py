"""Skills management CLI commands"""
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(help="Manage LocalAgent skills")
console = Console()


@app.command("list")
def skills_list():
    """List all available skills"""
    from local_agent.skills.loader import SkillLoader
    from local_agent.skills.registry import SkillRegistry

    loader = SkillLoader()
    loader.load_builtin_skills()
    reg = SkillRegistry()
    skills = reg.get_all_info()

    if not skills:
        console.print("[yellow]No skills found.[/yellow]")
        return

    table = Table(title="Available Skills", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Version", style="dim")
    table.add_column("Steps", style="yellow", no_wrap=True)

    for s in skills:
        steps_count = str(s.get("steps_count", "-"))
        table.add_row(
            s["name"],
            s["description"],
            s.get("version", "1.0.0"),
            steps_count,
        )
    console.print(table)


@app.command("info")
def skills_info(name: str = typer.Argument(help="Skill name")):
    """Show detailed information about a skill"""
    from local_agent.skills.loader import SkillLoader
    from local_agent.skills.registry import SkillRegistry

    loader = SkillLoader()
    loader.load_builtin_skills()
    reg = SkillRegistry()
    skill = reg.get(name)

    if not skill:
        console.print(f"[red]Skill not found: {name}[/red]")
        raise typer.Exit(1)

    info = skill.get_info()
    parsed_cfg = getattr(skill, "parsed_config", None)

    steps_info = ""
    if parsed_cfg and hasattr(parsed_cfg, "steps"):
        steps_lines = []
        for i, step in enumerate(parsed_cfg.steps, 1):
            steps_lines.append(
                f"  {i}. [{step.type}] {step.name}"
            )
        if steps_lines:
            steps_info = "\n\n[bold]Steps:[/bold]\n" + "\n".join(steps_lines)

    overview = getattr(parsed_cfg, "overview", "") if parsed_cfg else ""
    console.print(Panel(
        f"[bold cyan]{info['name']}[/bold cyan] v{info.get('version', '1.0.0')}\n\n"
        f"[white]{info['description']}[/white]\n"
        + (f"\n[dim]{overview}[/dim]\n" if overview else "")
        + f"\n[bold]Available Tools:[/bold] {', '.join(info.get('required_tools', []) or ['All tools'])}"
        + steps_info,
        title="Skill Info",
        border_style="cyan"
    ))
