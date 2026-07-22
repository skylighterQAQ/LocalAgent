"""
Filesystem Tools - Read, write, list, search files

All relative paths are resolved against the active workspace directory
(if one has been configured via ``local_agent.workspace.context``).
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from local_agent.core.tools import tool
from local_agent.cli.workspace.context import resolve_workspace_path


def _resolve(path: str) -> Path:
    """Resolve a user-supplied path against the active workspace."""
    return Path(resolve_workspace_path(path)).expanduser()


@tool
def fs_read_file(path: str) -> str:
    """Read the contents of a file. Relative paths resolve to the active workspace."""
    try:
        p = _resolve(path)
        if not p.exists():
            return f"Error: File not found: {p}"
        if not p.is_file():
            return f"Error: Not a file: {path}"
        size = p.stat().st_size
        if size > 10 * 1024 * 1024:  # 10MB limit
            return f"Error: File too large ({size} bytes). Max 10MB."
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def fs_write_file(path: str, content: str, append: bool = False) -> str:
    """Write content to a file. Relative paths resolve to the active workspace. Set append=True to append."""
    try:
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if append:
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(content)
        else:
            # Write beside the destination and atomically replace it. A failed
            # generation/write must not leave a truncated or placeholder file.
            temp_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="",
                    dir=p.parent,
                    prefix=f".{p.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as fh:
                    fh.write(content)
                    fh.flush()
                    os.fsync(fh.fileno())
                    temp_path = Path(fh.name)
                os.replace(temp_path, p)
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()

            if p.read_text(encoding="utf-8") != content:
                return f"Error writing file: verification failed for {p}"
        return f"Successfully {'appended to' if append else 'wrote'} file: {p} ({len(content)} chars)"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def fs_append_file(path: str, content: str) -> str:
    """Append content to an existing file (or create it if not exists). Relative paths resolve to the active workspace."""
    try:
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(content)
        size = p.stat().st_size
        return f"Successfully appended to file: {p} ({len(content)} chars appended, total {size} bytes)"
    except Exception as e:
        return f"Error appending to file: {e}"


@tool
def fs_list_dir(path: str = ".", show_hidden: bool = False) -> str:
    """List files and directories in the given path. Relative paths resolve to the active workspace."""
    try:
        p = _resolve(path)
        if not p.exists():
            return f"Error: Path not found: {p}"
        if not p.is_dir():
            return f"Error: Not a directory: {p}"

        items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        lines = [f"Contents of {p.resolve()}:"]
        for item in items:
            if not show_hidden and item.name.startswith("."):
                continue
            if item.is_dir():
                lines.append(f"  📁 {item.name}/")
            else:
                size = item.stat().st_size
                size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB"
                lines.append(f"  📄 {item.name} ({size_str})")
        return "\n".join(lines)
    except Exception as e:
        return f"Error listing directory: {e}"


@tool
def fs_search_files(directory: str = ".", pattern: str = "*", recursive: bool = True) -> str:
    """Search for files matching a glob pattern. Relative directory resolves to the active workspace."""
    try:
        p = _resolve(directory)
        if not p.exists():
            return f"Error: Directory not found: {p}"

        if recursive:
            matches = list(p.rglob(pattern))
        else:
            matches = list(p.glob(pattern))

        if not matches:
            return f"No files found matching '{pattern}' in {directory}"

        lines = [f"Found {len(matches)} file(s) matching '{pattern}':"]
        for m in matches[:50]:  # Limit results
            lines.append(f"  {m}")
        if len(matches) > 50:
            lines.append(f"  ... and {len(matches) - 50} more")
        return "\n".join(lines)
    except Exception as e:
        return f"Error searching files: {e}"


@tool
def fs_create_dir(path: str) -> str:
    """Create a directory (and parent directories if needed). Relative paths resolve to the active workspace."""
    try:
        p = _resolve(path)
        p.mkdir(parents=True, exist_ok=True)
        return f"Directory created: {p.resolve()}"
    except Exception as e:
        return f"Error creating directory: {e}"


@tool
def fs_delete_file(path: str) -> str:
    """Delete a file or empty directory. Relative paths resolve to the active workspace."""
    try:
        p = _resolve(path)
        if not p.exists():
            return f"Error: Path not found: {p}"
        if p.is_file():
            p.unlink()
            return f"File deleted: {path}"
        elif p.is_dir():
            if any(p.iterdir()):
                return f"Error: Directory is not empty. Use fs_delete_dir for non-empty directories."
            p.rmdir()
            return f"Empty directory deleted: {path}"
    except Exception as e:
        return f"Error deleting: {e}"


@tool
def fs_move_file(source: str, destination: str) -> str:
    """Move or rename a file or directory. Relative paths resolve to the active workspace."""
    try:
        src = _resolve(source)
        dst = _resolve(destination)
        if not src.exists():
            return f"Error: Source not found: {src}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Moved: {src} → {dst}"
    except Exception as e:
        return f"Error moving: {e}"


@tool
def fs_copy_file(source: str, destination: str) -> str:
    """Copy a file to a new location. Relative paths resolve to the active workspace."""
    try:
        src = _resolve(source)
        dst = _resolve(destination)
        if not src.exists():
            return f"Error: Source not found: {src}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_file():
            shutil.copy2(str(src), str(dst))
        else:
            shutil.copytree(str(src), str(dst))
        return f"Copied: {src} → {dst}"
    except Exception as e:
        return f"Error copying: {e}"


@tool
def fs_get_file_info(path: str) -> str:
    """Get detailed information about a file or directory. Relative paths resolve to the active workspace."""
    try:
        p = _resolve(path)
        if not p.exists():
            return f"Error: Path not found: {p}"
        stat = p.stat()
        import datetime
        info = {
            "path": str(p.resolve()),
            "type": "directory" if p.is_dir() else "file",
            "size": f"{stat.st_size:,} bytes",
            "modified": datetime.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "created": datetime.datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        }
        if p.is_file():
            info["extension"] = p.suffix
        lines = [f"File Info: {path}"]
        for k, v in info.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error getting file info: {e}"


@tool
def fs_grep(path: str, pattern: str, recursive: bool = True, max_results: int = 50) -> str:
    """Search for a text pattern inside files. Relative paths resolve to the active workspace."""
    import re
    try:
        p = _resolve(path)
        results = []
        files = list(p.rglob("*")) if recursive and p.is_dir() else [p]

        for f in files:
            if not f.is_file():
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                for i, line in enumerate(content.splitlines(), 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        results.append(f"{f}:{i}: {line.strip()}")
                        if len(results) >= max_results:
                            break
            except Exception:
                continue
            if len(results) >= max_results:
                break

        if not results:
            return f"No matches found for '{pattern}' in {path}"
        return "\n".join([f"Found {len(results)} match(es) for '{pattern}':"] + results)
    except Exception as e:
        return f"Error searching: {e}"


# Assign categories
fs_read_file.metadata = fs_read_file.metadata or {}
fs_read_file.metadata["category"] = "filesystem"
fs_write_file.metadata = fs_write_file.metadata or {}
fs_write_file.metadata["category"] = "filesystem"
fs_list_dir.metadata = fs_list_dir.metadata or {}
fs_list_dir.metadata["category"] = "filesystem"
fs_search_files.metadata = fs_search_files.metadata or {}
fs_search_files.metadata["category"] = "filesystem"
fs_create_dir.metadata = fs_create_dir.metadata or {}
fs_create_dir.metadata["category"] = "filesystem"
fs_delete_file.metadata = fs_delete_file.metadata or {}
fs_delete_file.metadata["category"] = "filesystem"
fs_move_file.metadata = fs_move_file.metadata or {}
fs_move_file.metadata["category"] = "filesystem"
fs_copy_file.metadata = fs_copy_file.metadata or {}
fs_copy_file.metadata["category"] = "filesystem"
fs_get_file_info.metadata = fs_get_file_info.metadata or {}
fs_get_file_info.metadata["category"] = "filesystem"
fs_grep.metadata = fs_grep.metadata or {}
fs_grep.metadata["category"] = "filesystem"
fs_append_file.metadata = fs_append_file.metadata or {}
fs_append_file.metadata["category"] = "filesystem"

TOOLS = [
    fs_read_file,
    fs_write_file,
    fs_append_file,
    fs_list_dir,
    fs_search_files,
    fs_create_dir,
    fs_delete_file,
    fs_move_file,
    fs_copy_file,
    fs_get_file_info,
    fs_grep,
]
