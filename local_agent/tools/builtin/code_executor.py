"""
Code Execution Tools - Execute Python and other code safely

Supports:
  - Python    (subprocess / inline exec)
  - JavaScript / TypeScript  (node / ts-node)
  - Shell Script  (bash / zsh / sh)
  - Go        (go run)
  - Rust      (rustc + exec)
  - Generic   (unified multi-language entry point)
  - Lint      (multi-language static analysis)
  - Run file  (auto-detect by extension)
"""
import subprocess
import sys
import os
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional
from local_agent.core.tools import tool
from local_agent.cli.workspace.context import get_active_workspace_dir


# ── Helpers ───────────────────────────────────────────────────────────────────

def _effective_cwd(cwd: Optional[str]) -> Optional[str]:
    """Return *cwd* if explicitly provided, else fall back to the active workspace
    directory.  Returns ``None`` only when neither a cwd nor a workspace is set,
    which lets subprocess default to the process CWD."""
    if cwd:
        return cwd
    ws = get_active_workspace_dir()
    return str(ws) if ws is not None else None

def _run_subprocess(
    args: list,
    *,
    input_text: Optional[str] = None,
    timeout: int = 30,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
) -> dict:
    """Run a subprocess and return a dict with stdout, stderr, returncode, duration."""
    start = time.time()
    try:
        result = subprocess.run(
            args,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env={**os.environ, **(env or {})},
        )
        duration = time.time() - start
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "duration": round(duration, 3),
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds",
            "returncode": -1,
            "duration": timeout,
        }
    except FileNotFoundError as e:
        return {
            "stdout": "",
            "stderr": f"Command not found: {e}",
            "returncode": -1,
            "duration": 0.0,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {e}",
            "returncode": -1,
            "duration": 0.0,
        }


def _format_result(res: dict, label: str = "") -> str:
    """Format a subprocess result dict into a readable string."""
    parts = []
    if label:
        parts.append(f"[{label}]")
    if res["stdout"]:
        parts.append(f"Output:\n{res['stdout'].rstrip()}")
    if res["stderr"]:
        parts.append(f"Errors:\n{res['stderr'].rstrip()}")
    parts.append(f"Return code: {res['returncode']}  (took {res['duration']}s)")
    return "\n".join(parts) if parts else "(no output)"


def _which(cmd: str) -> Optional[str]:
    """Return the path of a command if found, else None."""
    import shutil
    return shutil.which(cmd)


# ── Python ────────────────────────────────────────────────────────────────────

def _execute_python(code: str, timeout: int = 30, cwd: Optional[str] = None) -> str:
    """Internal Python executor (used by code_execute_generic)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name
    try:
        res = _run_subprocess([sys.executable, tmp_path], timeout=timeout, cwd=cwd)
        return _format_result(res, "Python")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@tool
def code_execute_python(code: str, timeout: int = 30, cwd: Optional[str] = None) -> str:
    """
    Execute Python code and return the output.
    Code runs in a subprocess for safety.
    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default 30)
        cwd: Working directory for execution
    """
    return _execute_python(code, timeout=timeout, cwd=_effective_cwd(cwd))


@tool
def code_execute_python_inline(code: str) -> str:
    """
    Execute Python code inline (in the same process) and return the result.
    Captures print output and the result of the last expression.
    Args:
        code: Python code to execute
    """
    import io
    import contextlib

    stdout_capture = io.StringIO()
    local_vars = {}

    try:
        with contextlib.redirect_stdout(stdout_capture):
            with contextlib.redirect_stderr(stdout_capture):
                exec(compile(code, "<string>", "exec"), local_vars)

        stdout_output = stdout_capture.getvalue()
        output_parts = []
        if stdout_output:
            output_parts.append(f"Output:\n{stdout_output}")
        if "result" in local_vars:
            output_parts.append(f"result = {local_vars['result']}")
        return "\n".join(output_parts) if output_parts else "(executed successfully, no output)"
    except Exception:
        return f"Error:\n{traceback.format_exc()}"


# ── JavaScript / TypeScript ───────────────────────────────────────────────────

def _execute_js_internal(code: str, language: str = "javascript", timeout: int = 30, cwd: Optional[str] = None) -> str:
    """Internal JS/TS executor (used by code_execute_generic)."""
    lang = language.lower().strip()

    if lang in ("typescript", "ts"):
        runner = _which("ts-node") or _which("tsx")
        if runner is None:
            return (
                "Error: TypeScript runner not found.\n"
                "Install with: npm install -g ts-node  OR  npm install -g tsx"
            )
        suffix = ".ts"
        cmd_fn = lambda p: [runner, p]
    else:
        runner = _which("node")
        if runner is None:
            return "Error: Node.js not found. Install from https://nodejs.org"
        suffix = ".js"
        cmd_fn = lambda p: [runner, p]

    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name
    try:
        res = _run_subprocess(cmd_fn(tmp_path), timeout=timeout, cwd=cwd)
        return _format_result(res, lang.title())
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@tool
def code_execute_js(code: str, language: str = "javascript", timeout: int = 30, cwd: Optional[str] = None) -> str:
    """
    Execute JavaScript or TypeScript code.
    Args:
        code: JavaScript or TypeScript code to execute
        language: 'javascript' (default) or 'typescript'
        timeout: Maximum execution time in seconds (default 30)
        cwd: Working directory for execution
    """
    return _execute_js_internal(code, language=language, timeout=timeout, cwd=_effective_cwd(cwd))


# ── Shell Script ──────────────────────────────────────────────────────────────

def _execute_shell_internal(code: str, shell: str = "bash", timeout: int = 30, cwd: Optional[str] = None) -> str:
    """Internal shell executor (used by code_execute_generic)."""
    shell_path = _which(shell) or _which("bash") or _which("sh")
    if shell_path is None:
        return "Error: No shell found (tried bash, sh)"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name
    os.chmod(tmp_path, 0o755)

    try:
        res = _run_subprocess([shell_path, tmp_path], timeout=timeout, cwd=cwd)
        return _format_result(res, f"Shell ({shell})")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@tool
def code_execute_shell(code: str, shell: str = "bash", timeout: int = 30, cwd: Optional[str] = None) -> str:
    """
    Execute a shell script.
    Args:
        code: Shell script content
        shell: Shell to use: 'bash' (default), 'zsh', or 'sh'
        timeout: Maximum execution time in seconds (default 30)
        cwd: Working directory for execution
    """
    return _execute_shell_internal(code, shell=shell, timeout=timeout, cwd=_effective_cwd(cwd))


# ── Go ────────────────────────────────────────────────────────────────────────

def _execute_go_internal(code: str, timeout: int = 60, cwd: Optional[str] = None) -> str:
    """Internal Go executor (used by code_execute_generic)."""
    go_bin = _which("go")
    if go_bin is None:
        return "Error: Go not found. Install from https://go.dev/dl/"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".go", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name
    try:
        res = _run_subprocess([go_bin, "run", tmp_path], timeout=timeout, cwd=cwd)
        return _format_result(res, "Go")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@tool
def code_execute_go(code: str, timeout: int = 60, cwd: Optional[str] = None) -> str:
    """
    Execute Go code using 'go run'.
    Args:
        code: Go source code (must include package main and main function)
        timeout: Maximum execution time in seconds (default 60)
        cwd: Working directory for execution
    """
    return _execute_go_internal(code, timeout=timeout, cwd=_effective_cwd(cwd))


# ── Generic multi-language entry point ───────────────────────────────────────

# Map of language aliases → canonical name
_LANG_ALIASES: dict = {
    "python": "python",
    "py": "python",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "bash": "bash",
    "shell": "bash",
    "sh": "bash",
    "zsh": "zsh",
    "go": "go",
    "golang": "go",
    "rust": "rust",
    "rs": "rust",
}


@tool
def code_execute_generic(
    code: str,
    language: str,
    timeout: int = 30,
    cwd: Optional[str] = None,
) -> str:
    """
    Execute code in any supported language (unified entry point).
    Supported languages: python, javascript/js, typescript/ts, bash/shell/sh, zsh, go, rust.
    Args:
        code: Source code to execute
        language: Programming language (e.g. 'python', 'js', 'ts', 'bash', 'go', 'rust')
        timeout: Maximum execution time in seconds (default 30)
        cwd: Working directory for execution
    """
    lang = _LANG_ALIASES.get(language.lower().strip(), language.lower().strip())

    if lang == "python":
        return _execute_python(code, timeout=timeout, cwd=_effective_cwd(cwd))
    elif lang == "javascript":
        return _execute_js_internal(code, language="javascript", timeout=timeout, cwd=_effective_cwd(cwd))
    elif lang == "typescript":
        return _execute_js_internal(code, language="typescript", timeout=timeout, cwd=_effective_cwd(cwd))
    elif lang in ("bash", "zsh"):
        return _execute_shell_internal(code, shell=lang, timeout=timeout, cwd=_effective_cwd(cwd))
    elif lang == "go":
        return _execute_go_internal(code, timeout=timeout, cwd=_effective_cwd(cwd))
    elif lang == "rust":
        return _execute_rust(code, timeout=timeout, cwd=_effective_cwd(cwd))
    else:
        return (
            f"Unsupported language: '{language}'.\n"
            "Supported: python, javascript/js, typescript/ts, bash/shell/sh, zsh, go, rust"
        )


def _execute_rust(code: str, timeout: int = 60, cwd: Optional[str] = None) -> str:
    """Execute Rust code using rustc."""
    rustc = _which("rustc")
    if rustc is None:
        return "Error: rustc not found. Install Rust from https://rustup.rs"

    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "main.rs")
        out = os.path.join(tmpdir, "main")
        with open(src, "w", encoding="utf-8") as f:
            f.write(code)

        compile_res = _run_subprocess([rustc, src, "-o", out], timeout=30, cwd=tmpdir)
        if compile_res["returncode"] != 0:
            return _format_result(compile_res, "Rust (compile)")

        run_res = _run_subprocess([out], timeout=timeout, cwd=cwd or tmpdir)
        return _format_result(run_res, "Rust (run)")


# ── Run a file directly ───────────────────────────────────────────────────────

_EXT_TO_LANG: dict = {
    ".py": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".go": "go",
    ".rs": "rust",
}


@tool
def code_run_file(
    file_path: str,
    args: str = "",
    timeout: int = 60,
    cwd: Optional[str] = None,
) -> str:
    """
    Run a source code file directly, auto-detecting language from extension.
    Supports: .py .js .ts .sh .bash .zsh .go .rs
    Args:
        file_path: Path to the source file
        args: Optional command-line arguments string (e.g. '--verbose --count 3')
        timeout: Maximum execution time in seconds (default 60)
        cwd: Working directory (defaults to file's parent directory)
    """
    p = Path(file_path).expanduser()
    if not p.exists():
        return f"Error: File not found: {file_path}"
    if not p.is_file():
        return f"Error: Not a file: {file_path}"

    lang = _EXT_TO_LANG.get(p.suffix.lower())
    if lang is None:
        return (
            f"Error: Unknown file extension '{p.suffix}'.\n"
            f"Supported: {', '.join(sorted(_EXT_TO_LANG.keys()))}"
        )

    extra_args = args.split() if args.strip() else []
    run_cwd = cwd or _effective_cwd(None) or str(p.parent)

    if lang == "python":
        res = _run_subprocess([sys.executable, str(p)] + extra_args, timeout=timeout, cwd=run_cwd)
    elif lang == "javascript":
        node = _which("node")
        if node is None:
            return "Error: Node.js not found."
        res = _run_subprocess([node, str(p)] + extra_args, timeout=timeout, cwd=run_cwd)
    elif lang == "typescript":
        runner = _which("ts-node") or _which("tsx")
        if runner is None:
            return "Error: ts-node or tsx not found. Install: npm install -g ts-node"
        res = _run_subprocess([runner, str(p)] + extra_args, timeout=timeout, cwd=run_cwd)
    elif lang in ("bash", "zsh"):
        sh = _which(lang) or _which("bash") or _which("sh")
        res = _run_subprocess([sh, str(p)] + extra_args, timeout=timeout, cwd=run_cwd)
    elif lang == "go":
        go = _which("go")
        if go is None:
            return "Error: Go not found."
        res = _run_subprocess([go, "run", str(p)] + extra_args, timeout=timeout, cwd=run_cwd)
    elif lang == "rust":
        return _execute_rust(p.read_text(encoding="utf-8"), timeout=timeout, cwd=run_cwd)
    else:
        return f"Unsupported language: {lang}"

    return _format_result(res, f"{lang.title()} ({p.name})")


# ── Lint / static analysis ────────────────────────────────────────────────────

@tool
def code_lint_multi(code: str, language: str = "python") -> str:
    """
    Lint / statically analyze code for errors and style issues.
    Supports: python, javascript/js, typescript/ts, bash/shell.
    Args:
        code: Code to analyze
        language: Programming language
    """
    lang = _LANG_ALIASES.get(language.lower().strip(), language.lower().strip())
    results = []

    if lang == "python":
        results.extend(_lint_python(code))
    elif lang == "javascript":
        results.extend(_lint_js(code, is_ts=False))
    elif lang == "typescript":
        results.extend(_lint_js(code, is_ts=True))
    elif lang in ("bash", "sh", "shell"):
        results.extend(_lint_shell(code))
    else:
        return f"Linting for '{language}' is not yet supported. Supported: python, js, ts, bash"

    return "\n".join(results)


def _lint_python(code: str) -> list:
    results = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        # Try pyflakes
        res = _run_subprocess([sys.executable, "-m", "pyflakes", tmp], timeout=15)
        if res["returncode"] == 0 and not res["stdout"] and not res["stderr"]:
            results.append("Pyflakes: No issues found ✓")
        else:
            output = (res["stdout"] or res["stderr"]).replace(tmp, "<code>")
            results.append(f"Pyflakes:\n{output.rstrip()}")

        # Try flake8 if available
        flake8 = _which("flake8")
        if flake8:
            res2 = _run_subprocess([flake8, "--max-line-length=120", tmp], timeout=15)
            output2 = (res2["stdout"] or "").replace(tmp, "<code>").rstrip()
            if output2:
                results.append(f"Flake8:\n{output2}")
            else:
                results.append("Flake8: No issues found ✓")
    except Exception as e:
        # Fallback: syntax check
        try:
            compile(code, "<string>", "exec")
            results.append("Syntax: Valid Python ✓")
        except SyntaxError as se:
            results.append(f"Syntax Error: {se}")
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
    return results


def _lint_js(code: str, is_ts: bool = False) -> list:
    results = []
    suffix = ".ts" if is_ts else ".js"
    label = "TypeScript" if is_ts else "JavaScript"

    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        # Try node --check for JS (syntax only)
        node = _which("node")
        if node and not is_ts:
            res = _run_subprocess([node, "--check", tmp], timeout=15)
            if res["returncode"] == 0:
                results.append(f"{label} Syntax: Valid ✓")
            else:
                results.append(f"{label} Syntax Error:\n{(res['stderr'] or res['stdout']).rstrip()}")

        # Try tsc for TypeScript
        tsc = _which("tsc")
        if tsc and is_ts:
            res = _run_subprocess([tsc, "--noEmit", "--allowJs", tmp], timeout=20)
            if res["returncode"] == 0:
                results.append(f"TypeScript (tsc): No type errors ✓")
            else:
                results.append(f"TypeScript (tsc):\n{(res['stdout'] or res['stderr']).rstrip()}")

        if not results:
            results.append(f"{label}: No linter available (install node or tsc for checks)")
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
    return results


def _lint_shell(code: str) -> list:
    results = []
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        shellcheck = _which("shellcheck")
        if shellcheck:
            res = _run_subprocess([shellcheck, tmp], timeout=15)
            output = (res["stdout"] or "").replace(tmp, "<script>").rstrip()
            if output:
                results.append(f"Shellcheck:\n{output}")
            else:
                results.append("Shellcheck: No issues found ✓")
        else:
            # Fallback: bash -n (syntax check)
            bash = _which("bash")
            if bash:
                res = _run_subprocess([bash, "-n", tmp], timeout=10)
                if res["returncode"] == 0:
                    results.append("Shell Syntax (bash -n): Valid ✓")
                else:
                    results.append(f"Shell Syntax Error:\n{res['stderr'].rstrip()}")
            else:
                results.append("Shell lint: No checker available (install shellcheck)")
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
    return results


# ── Format ────────────────────────────────────────────────────────────────────

@tool
def code_format(code: str, language: str = "python") -> str:
    """
    Format source code using appropriate formatters.
    Supports: python (black / autopep8), javascript/typescript (prettier).
    Args:
        code: Source code to format
        language: Programming language
    """
    lang = _LANG_ALIASES.get(language.lower().strip(), language.lower().strip())

    if lang == "python":
        return _format_python(code)
    elif lang in ("javascript", "typescript"):
        return _format_js(code, is_ts=(lang == "typescript"))
    else:
        return f"Unsupported language for formatting: {language}. Supported: python, javascript, typescript"


def _format_python(code: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp = f.name
    try:
        # Try black
        black = _which("black")
        if black:
            res = _run_subprocess([black, "--quiet", tmp], timeout=15)
            if res["returncode"] == 0:
                return Path(tmp).read_text(encoding="utf-8")
        # Try autopep8
        autopep8 = _which("autopep8")
        if autopep8:
            res = _run_subprocess([autopep8, "-", "--aggressive"], timeout=15,
                                  input_text=code)
            if res["returncode"] == 0 and res["stdout"]:
                return res["stdout"]
        return code  # Return as-is if no formatter available
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def _format_js(code: str, is_ts: bool = False) -> str:
    prettier = _which("prettier")
    if prettier is None:
        return code  # Return as-is if prettier not available
    parser = "typescript" if is_ts else "babel"
    res = _run_subprocess([prettier, "--parser", parser], timeout=15, input_text=code)
    if res["returncode"] == 0 and res["stdout"]:
        return res["stdout"]
    return code


# ── Tests ─────────────────────────────────────────────────────────────────────

@tool
def code_run_tests(
    test_path: str,
    verbose: bool = True,
    cwd: Optional[str] = None,
    extra_args: str = "",
) -> str:
    """
    Run tests for a file or directory.
    Supports Python (pytest) by default; uses Jest for .js/.ts test files.
    Args:
        test_path: Path to test file or directory
        verbose: Show verbose output (default True)
        cwd: Working directory for test execution
        extra_args: Extra arguments string to pass to the test runner
    """
    p = Path(test_path).expanduser()
    ext = p.suffix.lower() if p.is_file() else ".py"
    run_cwd = cwd or _effective_cwd(None) or (str(p.parent) if p.is_file() else str(p))
    extra = extra_args.split() if extra_args.strip() else []

    if ext in (".js", ".ts", ".tsx"):
        jest = _which("jest")
        if jest is None:
            return "Error: jest not found. Install: npm install -g jest"
        args = [jest, str(test_path)] + extra
        if verbose:
            args.append("--verbose")
        res = _run_subprocess(args, timeout=120, cwd=run_cwd)
    else:
        # pytest
        args = [sys.executable, "-m", "pytest", str(test_path), "--tb=short"]
        if verbose:
            args.append("-v")
        args.extend(extra)
        res = _run_subprocess(args, timeout=120, cwd=run_cwd)

    return _format_result(res, "Tests")


# ── REPL eval ─────────────────────────────────────────────────────────────────

@tool
def repl_eval(expression: str) -> str:
    """
    Evaluate a Python expression and return the result.
    Useful for quick calculations or evaluations.
    """
    try:
        result = eval(expression, {"__builtins__": __builtins__})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ── Metadata ──────────────────────────────────────────────────────────────────

_CODE_TOOLS = [
    code_execute_python,
    code_execute_python_inline,
    code_execute_js,
    code_execute_shell,
    code_execute_go,
    code_execute_generic,
    code_run_file,
    code_lint_multi,
    code_format,
    code_run_tests,
    repl_eval,
]

for _t in _CODE_TOOLS:
    if not getattr(_t, "metadata", None):
        _t.metadata = {}
    _t.metadata["category"] = "code"

TOOLS = _CODE_TOOLS
