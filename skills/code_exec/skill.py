"""Code execution skill for LocalAgent - run Python scripts safely."""
import sys
import io
import traceback
import subprocess
import tempfile
import os
from typing import Any, List, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.skill_base import OpenClawSkill
from core.config_loader import get_config


class RunPythonInput(BaseModel):
    code: str = Field(description="Python code to execute")
    capture_output: bool = Field(default=True, description="If True, capture and return stdout/stderr")


class RunPythonFileTool(BaseTool):
    name: str = "code_run_python"
    description: str = (
        "Execute Python code and return the output. "
        "Can run any Python code including imports, calculations, file I/O, etc. "
        "Use this to perform computations, data processing, or run scripts."
    )
    args_schema: Type[BaseModel] = RunPythonInput

    def _run(self, code: str, capture_output: bool = True) -> str:
        cfg = get_config().code_exec
        try:
            # Write code to a temp file and run as subprocess for isolation
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=capture_output,
                timeout=cfg.timeout,
                text=True,
                cwd=os.getcwd(),
            )

            output_parts = []
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")
            if result.returncode != 0:
                output_parts.append(f"Exit code: {result.returncode}")

            return "\n".join(output_parts) if output_parts else "(no output)"

        except subprocess.TimeoutExpired:
            return f"Error: Code execution timed out after {cfg.timeout}s"
        except Exception as e:
            return f"Error: {e}\n{traceback.format_exc()}"
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class EvalExpressionInput(BaseModel):
    expression: str = Field(description="A single Python expression to evaluate (not a full script)")


class EvalExpressionTool(BaseTool):
    name: str = "code_eval_expression"
    description: str = (
        "Evaluate a single Python expression and return its value. "
        "Useful for quick calculations like '2 ** 32' or 'len(\"hello\")'. "
        "For multi-line code or imports, use code_run_python instead."
    )
    args_schema: Type[BaseModel] = EvalExpressionInput

    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, {"__builtins__": __builtins__})
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class CodeExecSkill(OpenClawSkill):
    name = "code_exec"
    description = "Execute Python code and scripts"
    version = "1.0.0"

    def get_tools(self) -> List[BaseTool]:
        return [RunPythonFileTool(), EvalExpressionTool()]
