"""File operations skill for LocalAgent - read/write local files."""
import os
from pathlib import Path
from typing import List, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.skill_base import OpenClawSkill


class ReadFileInput(BaseModel):
    path: str = Field(description="Path to the file to read")
    encoding: str = Field(default="utf-8", description="File encoding")


class WriteFileInput(BaseModel):
    path: str = Field(description="Path to write the file")
    content: str = Field(description="Content to write")
    append: bool = Field(default=False, description="If True, append to file; if False, overwrite")


class ListDirInput(BaseModel):
    path: str = Field(default=".", description="Directory path to list")


class MakeDirInput(BaseModel):
    path: str = Field(description="Path to the file to read")
    name: str = Field(description="Name of directory")


class ReadFileTool(BaseTool):
    name: str = "file_read"
    description: str = "Read the contents of a local file. Returns the file content as text."
    args_schema: Type[BaseModel] = ReadFileInput

    def _run(self, path: str, encoding: str = "utf-8") -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"Error: File not found: {path}"
            if not p.is_file():
                return f"Error: Not a file: {path}"
            content = p.read_text(encoding=encoding)
            size = len(content)
            if size > 10000:
                return content[:10000] + f"\n\n... (truncated, {size} total chars)"
            return content
        except Exception as e:
            return f"Error reading {path}: {e}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class WriteFileTool(BaseTool):
    name: str = "file_write"
    description: str = "Write or append content to a local file. Creates the file if it doesn't exist."
    args_schema: Type[BaseModel] = WriteFileInput

    def _run(self, path: str, content: str, append: bool = False) -> str:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with open(p, mode, encoding="utf-8") as f:
                f.write(content)
            return f"Successfully {'appended to' if append else 'wrote'} {path} ({len(content)} chars)"
        except Exception as e:
            return f"Error writing {path}: {e}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class ListDirTool(BaseTool):
    name: str = "file_list_dir"
    description: str = "List files and directories in a given path."
    args_schema: Type[BaseModel] = ListDirInput

    def _run(self, path: str = ".") -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"Error: Path not found: {path}"
            entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
            lines = []
            for e in entries:
                indicator = "/" if e.is_dir() else ""
                size = f" ({e.stat().st_size}B)" if e.is_file() else ""
                lines.append(f"  {e.name}{indicator}{size}")
            return f"Contents of {path}:\n" + "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class MakeDirTool(BaseTool):
    name: str = "make_dir"
    description: str = "Make local directories in a given path."
    args_schema: Type[BaseModel] = MakeDirInput

    def _run(self, path: str, name: str):
        try:
            p = Path(os.path.join(path, name))
            p.parent.mkdir(parents=True, exist_ok=True)
            return f"Successfully make directory {path})"
        except Exception as e:
            return f"Error writing {path}: {e}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class FileOpsSkill(OpenClawSkill):
    name = "file_ops"
    description = "Read, write, list local files and make new directories."
    version = "1.0.0"

    def get_tools(self) -> List[BaseTool]:
        return [ReadFileTool(), WriteFileTool(), ListDirTool(), MakeDirTool()]
