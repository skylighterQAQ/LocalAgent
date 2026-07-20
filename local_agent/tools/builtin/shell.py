"""
Shell Tools - Execute shell commands
"""
import subprocess
import os
import shlex
from typing import Optional
from local_agent.core.tools import tool
from local_agent.cli.workspace.context import get_active_workspace_dir


@tool
def shell_run(command: str, timeout: int = 30, cwd: Optional[str] = None) -> str:
    """
    Run a shell command and return the output.
    Use with caution - commands are executed directly.
    Args:
        command: Shell command to run
        timeout: Timeout in seconds (default 30)
        cwd: Working directory for the command
    """
    if cwd is None:
        ws = get_active_workspace_dir()
        if ws is not None:
            cwd = str(ws)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output_parts = []
        if result.stdout:
            output_parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")
        output_parts.append(f"Return code: {result.returncode}")
        return "\n".join(output_parts) if output_parts else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {e}"


@tool
def shell_which(program: str) -> str:
    """Find the path of a program/command."""
    import shutil
    path = shutil.which(program)
    if path:
        return f"{program} is at: {path}"
    return f"{program} not found in PATH"


@tool
def shell_env_get(variable: str = "") -> str:
    """
    Get environment variable(s).
    If variable is empty, list all environment variables.
    """
    if variable:
        value = os.environ.get(variable)
        if value is not None:
            return f"{variable}={value}"
        return f"Environment variable '{variable}' not found"
    else:
        # List all (hide sensitive ones)
        sensitive = {"PASSWORD", "SECRET", "KEY", "TOKEN", "PASS", "CREDENTIAL"}
        lines = []
        for k, v in sorted(os.environ.items()):
            if any(s in k.upper() for s in sensitive):
                lines.append(f"{k}=***")
            else:
                lines.append(f"{k}={v}")
        return "\n".join(lines)


@tool
def process_list(filter_name: str = "") -> str:
    """List running processes. Optionally filter by name."""
    try:
        import psutil
        procs = []
        for proc in psutil.process_iter(["pid", "name", "status", "cpu_percent", "memory_percent"]):
            try:
                info = proc.info
                if filter_name and filter_name.lower() not in info["name"].lower():
                    continue
                procs.append(
                    f"PID={info['pid']:6d} | {info['status']:10s} | CPU={info['cpu_percent']:5.1f}% | "
                    f"MEM={info['memory_percent']:5.1f}% | {info['name']}"
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if not procs:
            return "No processes found" + (f" matching '{filter_name}'" if filter_name else "")
        return f"Running processes ({len(procs)}):\n" + "\n".join(procs[:50])
    except ImportError:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        return result.stdout[:3000]


@tool
def shell_curl(url: str, method: str = "GET", headers: str = "", data: str = "") -> str:
    """Make an HTTP request using curl-like syntax."""
    import httpx
    try:
        h = {}
        if headers:
            for line in headers.strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    h[k.strip()] = v.strip()

        resp = httpx.request(
            method=method.upper(),
            url=url,
            headers=h,
            content=data.encode() if data else None,
            timeout=30,
            follow_redirects=True,
        )
        return f"Status: {resp.status_code}\nHeaders: {dict(resp.headers)}\n\nBody:\n{resp.text[:5000]}"
    except Exception as e:
        return f"HTTP request failed: {e}"


shell_run.metadata = shell_run.metadata or {}
shell_run.metadata["category"] = "shell"
shell_which.metadata = shell_which.metadata or {}
shell_which.metadata["category"] = "shell"
shell_env_get.metadata = shell_env_get.metadata or {}
shell_env_get.metadata["category"] = "shell"
process_list.metadata = process_list.metadata or {}
process_list.metadata["category"] = "shell"
shell_curl.metadata = shell_curl.metadata or {}
shell_curl.metadata["category"] = "shell"

TOOLS = [shell_run, shell_which, shell_env_get, process_list, shell_curl]
