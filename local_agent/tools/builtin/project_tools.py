"""
Project Tools - Project scaffolding, dependency management, and build helpers.

Complements shell.py (low-level) and code_executor.py (code-level) with
higher-level project-lifecycle operations.  All subprocess calls follow the
same pattern as shell.py: capture_output=True, text=True, timeout-guarded.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
from local_agent.core.tools import tool
from local_agent.cli.workspace.context import resolve_workspace_path, get_active_workspace_dir


def _resolve_project_path(path: str) -> str:
    """Resolve a project path against the active workspace.

    - If *path* is ``"."`` or empty and a workspace is active, the workspace
      ``default_dir`` is returned so that new projects land inside the
      workspace rather than the process CWD.
    - Otherwise the path is resolved the same way as filesystem.py tools
      (absolute paths unchanged, relative paths joined to the workspace dir).

    Note: This is intended for *write* operations (scaffold, install_deps,
    run_command) where new projects should land in the workspace.
    For *read* operations (tree, read_config) use ``_resolve_read_path``
    so that ``"."`` maps to the actual process CWD instead.
    """
    if path in (".", ""):
        ws = get_active_workspace_dir()
        if ws is not None:
            return str(ws)
    return resolve_workspace_path(path)


def _resolve_read_path(path: str) -> str:
    """Resolve a path for *read-only* project tools (tree, read_config).

    Unlike ``_resolve_project_path``, this function does **not** redirect
    ``"."`` to the workspace default_dir.  This means:

    - ``"."`` → the process CWD (i.e. the actual project root)
    - absolute paths → unchanged
    - other relative paths → resolved against the workspace dir as usual

    This prevents ``project_tree(".")`` from scanning the workspace
    sub-directory instead of the real project root.
    """
    if not path or path == ".":
        return os.getcwd()
    p = Path(path).expanduser()
    if p.is_absolute():
        return str(p)
    # Relative path that is not "." → resolve against workspace dir if active
    return resolve_workspace_path(path)


def _run(cmd: list[str], cwd: Optional[str] = None, timeout: int = 120) -> str:
    """Internal helper: run *cmd* in *cwd*, return formatted output string."""
    if cwd is None:
        ws = get_active_workspace_dir()
        cwd = str(ws) if ws is not None else os.getcwd()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        parts = []
        if result.stdout.strip():
            parts.append(f"STDOUT:\n{result.stdout.rstrip()}")
        if result.stderr.strip():
            parts.append(f"STDERR:\n{result.stderr.rstrip()}")
        if not parts:
            parts.append("(no output)")
        parts.append(f"Return code: {result.returncode}")
        return "\n".join(parts)
    except FileNotFoundError as exc:
        return f"Error: command not found – {exc}"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout} seconds"
    except Exception as exc:
        return f"Error: {exc}"


# ── Scaffold templates ────────────────────────────────────────────────────────

_PYTHON_SCAFFOLD = {
    "src/__init__.py": "",
    "tests/__init__.py": "",
    "tests/test_sample.py": 'def test_placeholder():\n    assert True\n',
    "README.md": "# {name}\n\nA Python project.\n",
    "requirements.txt": "# Add your dependencies here\n",
    ".gitignore": "__pycache__/\n*.py[cod]\n.venv/\ndist/\nbuild/\n*.egg-info/\n.env\n",
    "pyproject.toml": (
        '[build-system]\nrequires = ["setuptools>=61", "wheel"]\n'
        'build-backend = "setuptools.backends.legacy:build"\n\n'
        "[project]\nname = \"{name}\"\nversion = \"0.1.0\"\n"
        'requires-python = ">=3.9"\n'
    ),
}

_NODE_SCAFFOLD = {
    "src/index.js": "// Entry point\nconsole.log('Hello, world!');\n",
    "tests/sample.test.js": "test('placeholder', () => { expect(true).toBe(true); });\n",
    "README.md": "# {name}\n\nA Node.js project.\n",
    ".gitignore": "node_modules/\ndist/\n.env\n",
    "package.json": json.dumps(
        {"name": "{name}", "version": "0.1.0", "main": "src/index.js",
         "scripts": {"start": "node src/index.js", "test": "jest"}},
        indent=2,
    ) + "\n",
}

_GO_SCAFFOLD = {
    "main.go": 'package main\n\nimport "fmt"\n\nfunc main() {\n\tfmt.Println("Hello, world!")\n}\n',
    "go.mod": "module {name}\n\ngo 1.21\n",
    "README.md": "# {name}\n\nA Go project.\n",
    ".gitignore": "# Binaries\n{name}\n*.exe\n",
}

_TEMPLATES: dict = {
    "python": _PYTHON_SCAFFOLD,
    "node": _NODE_SCAFFOLD,
    "go": _GO_SCAFFOLD,
}


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool
def project_scaffold(name: str, language: str = "python", base_path: str = ".") -> str:
    """
    Create a standard project directory structure.
    Args:
        name: Project name (used as root directory name and in template vars)
        language: Template to use – 'python', 'node', or 'go'
        base_path: Parent directory where the project folder will be created
    """
    lang = language.lower().strip()
    if lang not in _TEMPLATES:
        return (
            f"Error: unknown language '{language}'. "
            f"Supported: {', '.join(_TEMPLATES)}"
        )

    root = Path(_resolve_project_path(base_path)).expanduser().resolve() / name
    if root.exists():
        return f"Error: directory already exists: {root}"

    template = _TEMPLATES[lang]
    created: list[str] = []
    try:
        for rel_path, content in template.items():
            file_path = root / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(
                content.replace("{name}", name), encoding="utf-8"
            )
            created.append(str(file_path.relative_to(root)))
    except Exception as exc:
        return f"Error creating scaffold: {exc}"

    lines = [f"Project '{name}' ({lang}) created at {root}:", "Files:"]
    lines += [f"  {f}" for f in created]
    return "\n".join(lines)


@tool
def project_install_deps(path: str = ".", extra_args: str = "") -> str:
    """
    Install project dependencies using the appropriate package manager.
    Detects: requirements.txt → pip, package.json → npm, go.mod → go mod tidy.
    Args:
        path: Project root directory (default: current directory)
        extra_args: Additional arguments passed to the package manager command
    """
    root = Path(_resolve_project_path(path)).expanduser().resolve()
    cwd = str(root)

    # Detect package manager
    if (root / "requirements.txt").exists() or (root / "pyproject.toml").exists():
        req_file = "requirements.txt" if (root / "requirements.txt").exists() else "pyproject.toml"
        cmd = [sys.executable, "-m", "pip", "install", "-r" if req_file == "requirements.txt" else ".", req_file]
        if req_file == "pyproject.toml":
            cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
        if extra_args:
            cmd += extra_args.split()
        return _run(cmd, cwd=cwd)
    elif (root / "package.json").exists():
        npm = "npm"
        cmd = [npm, "install"]
        if extra_args:
            cmd += extra_args.split()
        return _run(cmd, cwd=cwd)
    elif (root / "go.mod").exists():
        cmd = ["go", "mod", "tidy"]
        if extra_args:
            cmd += extra_args.split()
        return _run(cmd, cwd=cwd)
    else:
        return (
            "Error: no recognised dependency manifest found in "
            f"{root}. Expected: requirements.txt, pyproject.toml, package.json, or go.mod"
        )


@tool
def project_run_command(command: str, path: str = ".", timeout: int = 120) -> str:
    """
    Run a build/test/lint command inside a project directory.
    Similar to shell_run but locks the working directory to *path*.
    Args:
        command: Shell command to run (e.g. 'npm test', 'pytest', 'go build ./...')
        path: Project root directory to run the command in
        timeout: Timeout in seconds (default: 120)
    """
    try:
        cwd = str(Path(_resolve_project_path(path)).expanduser().resolve())
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        parts = []
        if result.stdout:
            parts.append(f"STDOUT:\n{result.stdout.rstrip()}")
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr.rstrip()}")
        if not parts:
            parts.append("(no output)")
        parts.append(f"Return code: {result.returncode}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout} seconds"
    except Exception as exc:
        return f"Error running command: {exc}"


@tool
def project_tree(path: str = ".", max_depth: int = 4) -> str:
    """
    Display an annotated directory tree for a project, excluding common
    noise directories (node_modules, .git, __pycache__, dist, build, .venv).
    Args:
        path: Project root directory (default: current directory)
        max_depth: Maximum depth to recurse (default: 4)
    """
    _SKIP = {
        "node_modules", ".git", "__pycache__", "dist", "build",
        ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache",
        "*.egg-info", ".DS_Store",
    }

    root = Path(_resolve_read_path(path)).expanduser().resolve()
    if not root.exists():
        return f"Error: path not found: {path}"

    lines: list[str] = [str(root)]

    def _walk(dir_path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(
                dir_path.iterdir(),
                key=lambda x: (x.is_file(), x.name.lower()),
            )
        except PermissionError:
            return

        entries = [
            e for e in entries
            if e.name not in _SKIP and not e.name.endswith(".egg-info")
        ]

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            extension = "    " if is_last else "│   "
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                _walk(entry, prefix + extension, depth + 1)
            else:
                size = entry.stat().st_size
                size_str = f"{size:,}B" if size < 1024 else f"{size / 1024:.1f}KB"
                lines.append(f"{prefix}{connector}{entry.name} ({size_str})")

    _walk(root, "", 1)
    return "\n".join(lines)


@tool
def project_read_config(path: str = ".", config_file: str = "") -> str:
    """
    Read and display the contents of common project configuration files.
    Searches for: package.json, pyproject.toml, setup.cfg, Makefile,
    requirements.txt, go.mod, Cargo.toml, .env.example.
    Args:
        path: Project root directory (default: current directory)
        config_file: Specific config filename to read (optional; if omitted all found files are listed)
    """
    root = Path(_resolve_read_path(path)).expanduser().resolve()
    _COMMON_CONFIGS = [
        "package.json", "pyproject.toml", "setup.cfg", "setup.py",
        "Makefile", "makefile", "requirements.txt", "go.mod", "go.sum",
        "Cargo.toml", ".env.example", "docker-compose.yml", "Dockerfile",
        "tox.ini", ".flake8", ".eslintrc.json", ".eslintrc.js",
    ]

    if config_file:
        target = root / config_file
        if not target.exists():
            return f"Error: config file not found: {target}"
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
            return f"=== {config_file} ===\n{content}"
        except Exception as exc:
            return f"Error reading {config_file}: {exc}"

    # List all found config files
    found: list[tuple[str, str]] = []
    for name in _COMMON_CONFIGS:
        candidate = root / name
        if candidate.exists():
            try:
                content = candidate.read_text(encoding="utf-8", errors="replace")
                found.append((name, content[:2000] + ("..." if len(content) > 2000 else "")))
            except Exception:
                found.append((name, "(unreadable)"))

    if not found:
        return f"No common config files found in {root}"

    sections: list[str] = [f"Config files found in {root}:\n"]
    for name, content in found:
        sections.append(f"=== {name} ===\n{content}\n")
    return "\n".join(sections)


# ── Metadata & export ─────────────────────────────────────────────────────────

_PROJECT_TOOLS = [
    project_scaffold,
    project_install_deps,
    project_run_command,
    project_tree,
    project_read_config,
]

for _t in _PROJECT_TOOLS:
    _t.metadata = _t.metadata or {}
    _t.metadata["category"] = "project"

TOOLS = _PROJECT_TOOLS
