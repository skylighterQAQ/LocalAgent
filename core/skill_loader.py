"""Dynamic skill loader for OpenClaw."""
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional
from rich.console import Console

from core.skill_base import OpenClawSkill, get_registry

console = Console()


def load_skill_from_path(skill_dir: Path) -> Optional[OpenClawSkill]:
    """Load a skill from a directory containing skill.py."""
    skill_file = skill_dir / "skill.py"
    if not skill_file.exists():
        return None

    # Dynamically import the module
    module_name = f"skills.{skill_dir.name}.skill"
    spec = importlib.util.spec_from_file_location(module_name, skill_file)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        console.print(f"[red]Error loading skill '{skill_dir.name}': {e}[/red]")
        return None

    # Find the skill class
    skill_instance = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, OpenClawSkill)
            and attr is not OpenClawSkill
        ):
            try:
                skill_instance = attr()
                break
            except Exception as e:
                console.print(f"[red]Error instantiating skill '{attr_name}': {e}[/red]")

    return skill_instance


def load_skills(skill_names: List[str], skills_base_dir: str = "skills") -> None:
    """Load and register skills by name from the skills directory."""
    registry = get_registry()
    base = Path(skills_base_dir)

    for name in skill_names:
        skill_dir = base / name
        if not skill_dir.exists():
            console.print(f"[yellow]⚠ Skill directory not found: {skill_dir}[/yellow]")
            continue

        skill = load_skill_from_path(skill_dir)
        if skill is None:
            console.print(f"[yellow]⚠ Could not load skill: {name}[/yellow]")
            continue

        if skill.name in registry:
            console.print(f"[yellow]⚠ Skill already registered: {skill.name}[/yellow]")
            continue

        registry.register(skill)
        console.print(f"[green]✓ Loaded skill: [bold]{skill.name}[/bold] v{skill.version}[/green]")


def load_all_skills(skills_base_dir: str = "skills") -> None:
    """Load all skills found in the skills directory."""
    base = Path(skills_base_dir)
    if not base.exists():
        return

    skill_names = [d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith("_")]
    load_skills(skill_names, skills_base_dir)
