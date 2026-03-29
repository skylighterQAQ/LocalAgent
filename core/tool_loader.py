"""Dynamic tool loader for LocalAgent."""
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional
from rich.console import Console

from core.tool_base import LocalAgentTool, get_registry

console = Console()


def load_tool_from_path(tool_dir: Path) -> Optional[LocalAgentTool]:
    """ 从tool.py载入工具代码
    args:
        tool_dir: 工具文件
    """
    tool_file = tool_dir / "tool.py"
    if not tool_file.exists():
        return None

    # Dynamically import the module
    module_name = f"tools.{tool_dir.name}.tool"
    spec = importlib.util.spec_from_file_location(module_name, tool_file)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        console.print(f"[red]Error loading tool '{tool_dir.name}': {e}[/red]")
        return None

    # build tool
    tool_instance = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, LocalAgentTool)
            and attr is not LocalAgentTool
        ):
            try:
                tool_instance = attr()
                break
            except Exception as e:
                console.print(f"[red]Error instantiating tool '{attr_name}': {e}[/red]")

    return tool_instance


def load_tools(tools_names: List[str], tools_base_dir: str = "tools") -> None:
    """
    载入tool
    """
    registry = get_registry()
    base = Path(tools_base_dir)

    for name in tools_names:
        tool_dir = base / name
        if not tool_dir.exists():
            console.print(f"[yellow]⚠ Tool directory not found: {tool_dir}[/yellow]")
            continue

        tool = load_tool_from_path(tool_dir)
        if tool is None:
            console.print(f"[yellow]⚠ Could not load Tool: {name}[/yellow]")
            continue

        if tool.name in registry:
            console.print(f"[yellow]⚠ Tool already registered: {tool.name}[/yellow]")
            continue

        registry.register(tool)
        console.print(f"[green]✓ Loaded Tool: [bold]{tool.name}[/bold] v{tool.version}[/green]")


def load_all_tools(tools_base_dir: str = "tools") -> None:
    """Load all tools found in the tools directory."""
    base = Path(tools_base_dir)
    if not base.exists():
        return

    tool_names = [d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith("_")]
    load_tools(tool_names, tools_base_dir)
