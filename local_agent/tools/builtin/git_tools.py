"""
Git Tools - Version control operations using the system git binary.

All tools run via subprocess (same pattern as shell.py / code_executor.py).
Each tool returns plain text output suitable for LLM consumption.
"""
import os
import subprocess
from pathlib import Path
from typing import Optional
from local_agent.core.tools import tool


def _run_git(args: list[str], cwd: Optional[str] = None, timeout: int = 30) -> str:
    """Internal helper: run git with the given args list in *cwd*."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or os.getcwd(),
        )
        parts = []
        if result.stdout.strip():
            parts.append(result.stdout.rstrip())
        if result.stderr.strip():
            parts.append(f"[stderr] {result.stderr.rstrip()}")
        if not parts:
            parts.append("(no output)")
        parts.append(f"Exit code: {result.returncode}")
        return "\n".join(parts)
    except FileNotFoundError:
        return "Error: git not found. Please install git and ensure it is in PATH."
    except subprocess.TimeoutExpired:
        return f"Error: git command timed out after {timeout} seconds"
    except Exception as exc:
        return f"Error running git: {exc}"


# ── Tools ─────────────────────────────────────────────────────────────────────


@tool
def git_init(path: str = ".") -> str:
    """
    Initialize a new git repository at the given path.
    Args:
        path: Directory to initialise (default: current directory)
    """
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return _run_git(["init"], cwd=str(p))


@tool
def git_status(path: str = ".") -> str:
    """
    Show the working tree status of the repository at *path*.
    Args:
        path: Repository root directory (default: current directory)
    """
    return _run_git(["status"], cwd=str(Path(path).expanduser().resolve()))


@tool
def git_add(files: str = ".", path: str = ".") -> str:
    """
    Stage files for commit.
    Args:
        files: File(s) to stage – space-separated relative paths, or '.' for all
        path: Repository root directory (default: current directory)
    """
    file_list = files.split() if files.strip() != "." else ["."]
    return _run_git(["add"] + file_list, cwd=str(Path(path).expanduser().resolve()))


@tool
def git_commit(message: str, path: str = ".") -> str:
    """
    Commit staged changes with the given message.
    Args:
        message: Commit message
        path: Repository root directory (default: current directory)
    """
    return _run_git(
        ["commit", "-m", message],
        cwd=str(Path(path).expanduser().resolve()),
    )


@tool
def git_log(path: str = ".", max_count: int = 10) -> str:
    """
    Show recent commit history.
    Args:
        path: Repository root directory (default: current directory)
        max_count: Maximum number of commits to show (default: 10)
    """
    return _run_git(
        ["log", f"--max-count={max_count}", "--oneline", "--decorate"],
        cwd=str(Path(path).expanduser().resolve()),
    )


@tool
def git_diff(path: str = ".", staged: bool = False) -> str:
    """
    Show differences in the working tree or staged area.
    Args:
        path: Repository root directory (default: current directory)
        staged: If True, show staged (cached) diff; otherwise show unstaged diff
    """
    args = ["diff"]
    if staged:
        args.append("--cached")
    return _run_git(args, cwd=str(Path(path).expanduser().resolve()))


@tool
def git_branch(path: str = ".", action: str = "list", branch_name: str = "") -> str:
    """
    Manage git branches.
    Args:
        path: Repository root directory (default: current directory)
        action: One of 'list', 'create', 'delete', 'switch'
        branch_name: Branch name (required for create/delete/switch)
    """
    cwd = str(Path(path).expanduser().resolve())
    action = action.lower().strip()
    if action == "list":
        return _run_git(["branch", "-a"], cwd=cwd)
    elif action == "create":
        if not branch_name:
            return "Error: branch_name is required for action='create'"
        return _run_git(["checkout", "-b", branch_name], cwd=cwd)
    elif action == "delete":
        if not branch_name:
            return "Error: branch_name is required for action='delete'"
        return _run_git(["branch", "-d", branch_name], cwd=cwd)
    elif action == "switch":
        if not branch_name:
            return "Error: branch_name is required for action='switch'"
        return _run_git(["checkout", branch_name], cwd=cwd)
    else:
        return f"Error: unknown action '{action}'. Use one of: list, create, delete, switch"


@tool
def git_clone(url: str, destination: str = "", depth: int = 0) -> str:
    """
    Clone a remote git repository.
    Args:
        url: Repository URL (https or ssh)
        destination: Local directory name/path (default: inferred from URL)
        depth: Shallow clone depth; 0 means full clone
    """
    args = ["clone"]
    if depth > 0:
        args += ["--depth", str(depth)]
    args.append(url)
    if destination:
        args.append(destination)
    return _run_git(args, timeout=120)


@tool
def git_push(path: str = ".", remote: str = "origin", branch: str = "") -> str:
    """
    Push commits to a remote repository.
    Args:
        path: Repository root directory (default: current directory)
        remote: Remote name (default: 'origin')
        branch: Branch to push (default: current branch)
    """
    args = ["push", remote]
    if branch:
        args.append(branch)
    return _run_git(args, cwd=str(Path(path).expanduser().resolve()), timeout=60)


@tool
def git_pull(path: str = ".", remote: str = "origin", branch: str = "") -> str:
    """
    Pull changes from a remote repository.
    Args:
        path: Repository root directory (default: current directory)
        remote: Remote name (default: 'origin')
        branch: Branch to pull (default: current branch)
    """
    args = ["pull", remote]
    if branch:
        args.append(branch)
    return _run_git(args, cwd=str(Path(path).expanduser().resolve()), timeout=60)


# ── Metadata & export ─────────────────────────────────────────────────────────

_GIT_TOOLS = [
    git_init,
    git_status,
    git_add,
    git_commit,
    git_log,
    git_diff,
    git_branch,
    git_clone,
    git_push,
    git_pull,
]

for _t in _GIT_TOOLS:
    _t.metadata = _t.metadata or {}
    _t.metadata["category"] = "git"

TOOLS = _GIT_TOOLS
