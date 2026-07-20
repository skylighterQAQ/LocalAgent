"""
Computer Tools – control mouse, keyboard, screen, clipboard, browser and apps.

These tools rely on ``pyautogui``, ``pyperclip``, ``webbrowser`` and
``subprocess`` to drive the local machine. Each tool fails gracefully when
the dependency is missing or the action raises an error.

Safety notes
------------
- ``pyautogui.FAILSAFE = True`` lets you abort by slamming the mouse into
  the top-left corner of the screen.
- A small global ``PAUSE`` (0.05s) is set so consecutive actions are
  visible/recoverable.

All file outputs (e.g. screenshots) are resolved relative to the active
workspace via :mod:`local_agent.workspace.context`.
"""
from __future__ import annotations

import datetime
import logging
import os
import platform
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Optional

from local_agent.core.tools import tool
from local_agent.cli.workspace.context import resolve_workspace_path

logger = logging.getLogger(__name__)


_PYAUTOGUI_HINT = (
    "Error: pyautogui not installed. Run: pip install pyautogui "
    "(macOS may also require: pip install pyobjc-core pyobjc)."
)
_PYPERCLIP_HINT = (
    "Error: pyperclip not installed. Run: pip install pyperclip."
)


def _import_pyautogui():
    """Lazy-import pyautogui and configure global safety settings."""
    try:
        import pyautogui  # type: ignore
    except Exception as exc:  # ImportError or display-related errors
        return None, f"{_PYAUTOGUI_HINT} ({exc})"
    try:
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
    except Exception:
        pass
    return pyautogui, None


def _import_pyperclip():
    try:
        import pyperclip  # type: ignore
        return pyperclip, None
    except Exception as exc:
        return None, f"{_PYPERCLIP_HINT} ({exc})"


def _default_screenshot_path() -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return resolve_workspace_path(f"screenshots/screenshot_{ts}.png")


# ── Mouse ────────────────────────────────────────────────────────────────

@tool
def computer_get_screen_size() -> str:
    """Return the primary screen size as 'width x height'."""
    pg, err = _import_pyautogui()
    if err:
        return err
    try:
        w, h = pg.size()
        return f"Screen size: {w} x {h}"
    except Exception as e:
        return f"Error getting screen size: {e}"


@tool
def computer_get_mouse_position() -> str:
    """Return the current mouse cursor position as 'x, y'."""
    pg, err = _import_pyautogui()
    if err:
        return err
    try:
        x, y = pg.position()
        return f"Mouse position: {x}, {y}"
    except Exception as e:
        return f"Error reading mouse position: {e}"


@tool
def computer_move_mouse(x: int, y: int, duration: float = 0.2) -> str:
    """Move the mouse cursor to absolute screen coordinates (x, y)."""
    pg, err = _import_pyautogui()
    if err:
        return err
    try:
        pg.moveTo(int(x), int(y), duration=float(duration))
        return f"Mouse moved to ({x}, {y})"
    except Exception as e:
        return f"Error moving mouse: {e}"


@tool
def computer_click(
    x: Optional[int] = None,
    y: Optional[int] = None,
    button: str = "left",
    clicks: int = 1,
    interval: float = 0.05,
) -> str:
    """
    Click the mouse. If x/y are omitted, click at the current cursor position.

    Args:
        x, y:   Optional screen coordinates.
        button: 'left', 'right' or 'middle'.
        clicks: Number of clicks (e.g. 2 for double-click).
    """
    pg, err = _import_pyautogui()
    if err:
        return err
    try:
        kwargs = {"button": button, "clicks": int(clicks), "interval": float(interval)}
        if x is not None and y is not None:
            kwargs.update({"x": int(x), "y": int(y)})
        pg.click(**kwargs)
        pos = f"({x}, {y})" if x is not None and y is not None else "current position"
        return f"Clicked {clicks}x with {button} button at {pos}"
    except Exception as e:
        return f"Error clicking: {e}"


@tool
def computer_double_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Double-click at given coordinates (or current position when omitted)."""
    pg, err = _import_pyautogui()
    if err:
        return err
    try:
        if x is not None and y is not None:
            pg.doubleClick(x=int(x), y=int(y))
        else:
            pg.doubleClick()
        return "Double-clicked successfully"
    except Exception as e:
        return f"Error double-clicking: {e}"


@tool
def computer_right_click(x: Optional[int] = None, y: Optional[int] = None) -> str:
    """Right-click at given coordinates (or current position when omitted)."""
    pg, err = _import_pyautogui()
    if err:
        return err
    try:
        if x is not None and y is not None:
            pg.rightClick(x=int(x), y=int(y))
        else:
            pg.rightClick()
        return "Right-clicked successfully"
    except Exception as e:
        return f"Error right-clicking: {e}"


@tool
def computer_drag(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    duration: float = 0.5,
    button: str = "left",
) -> str:
    """Drag from (start_x, start_y) to (end_x, end_y) with the given mouse button."""
    pg, err = _import_pyautogui()
    if err:
        return err
    try:
        pg.moveTo(int(start_x), int(start_y))
        pg.dragTo(int(end_x), int(end_y), duration=float(duration), button=button)
        return f"Dragged ({start_x},{start_y}) → ({end_x},{end_y})"
    except Exception as e:
        return f"Error dragging: {e}"


@tool
def computer_scroll(amount: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
    """
    Scroll the mouse wheel. Positive amount scrolls up, negative scrolls down.
    Optional x/y move the cursor before scrolling.
    """
    pg, err = _import_pyautogui()
    if err:
        return err
    try:
        if x is not None and y is not None:
            pg.moveTo(int(x), int(y))
        pg.scroll(int(amount))
        return f"Scrolled by {amount}"
    except Exception as e:
        return f"Error scrolling: {e}"


# ── Keyboard ─────────────────────────────────────────────────────────────

@tool
def computer_type_text(text: str, interval: float = 0.02) -> str:
    """
    Type *text* using the keyboard.

    For ASCII text pyautogui.write is used (fast). For non-ASCII content
    (e.g. Chinese), the text is copied to the clipboard and pasted via
    Ctrl/Cmd+V because pyautogui cannot emit IME characters directly.
    """
    pg, err = _import_pyautogui()
    if err:
        return err
    try:
        if text.isascii():
            pg.write(text, interval=float(interval))
            return f"Typed {len(text)} ASCII chars"
        # Non-ASCII fallback via clipboard paste
        clip, cerr = _import_pyperclip()
        if cerr:
            return f"{cerr} (needed for non-ASCII text input)"
        previous = ""
        try:
            previous = clip.paste()
        except Exception:
            previous = ""
        clip.copy(text)
        modifier = "command" if platform.system() == "Darwin" else "ctrl"
        pg.hotkey(modifier, "v")
        # Restore previous clipboard contents in the background-ish way
        try:
            clip.copy(previous)
        except Exception:
            pass
        return f"Pasted {len(text)} chars via clipboard"
    except Exception as e:
        return f"Error typing text: {e}"


@tool
def computer_press_key(key: str) -> str:
    """Press (and release) a single keyboard key, e.g. 'enter', 'esc', 'f1'."""
    pg, err = _import_pyautogui()
    if err:
        return err
    try:
        pg.press(str(key))
        return f"Pressed key: {key}"
    except Exception as e:
        return f"Error pressing key: {e}"


@tool
def computer_hotkey(keys: str) -> str:
    """
    Trigger a keyboard shortcut. Pass keys as a comma-separated string,
    e.g. 'ctrl,c' or 'cmd,shift,t'.
    """
    pg, err = _import_pyautogui()
    if err:
        return err
    try:
        parts = [k.strip() for k in str(keys).split(",") if k.strip()]
        if not parts:
            return "Error: no keys provided"
        pg.hotkey(*parts)
        return f"Hotkey pressed: {'+'.join(parts)}"
    except Exception as e:
        return f"Error pressing hotkey: {e}"


# ── Screen ──────────────────────────────────────────────────────────────

@tool
def computer_screenshot(output_path: str = "") -> str:
    """
    Capture a full-screen screenshot. If *output_path* is empty, save to
    ``screenshots/screenshot_<timestamp>.png`` inside the active workspace.
    """
    try:
        target = resolve_workspace_path(output_path) if output_path else _default_screenshot_path()
        p = Path(target).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)

        # Prefer pyautogui (already a dependency); fall back to mss if needed
        pg, err = _import_pyautogui()
        if not err and pg is not None:
            img = pg.screenshot()
            img.save(str(p))
            return f"Screenshot saved: {p}"

        try:
            import mss  # type: ignore
            with mss.mss() as sct:
                sct.shot(output=str(p))
            return f"Screenshot saved: {p}"
        except Exception as exc:
            return f"{err or 'pyautogui unavailable'} | mss fallback failed: {exc}"
    except Exception as e:
        return f"Error taking screenshot: {e}"


@tool
def computer_locate_on_screen(image_path: str, confidence: float = 0.9) -> str:
    """
    Find an image on the screen and return its bounding box.

    Args:
        image_path: Path to the reference image (PNG). Relative paths resolve
                    to the active workspace.
        confidence: 0-1 confidence threshold (requires opencv).
    """
    pg, err = _import_pyautogui()
    if err:
        return err
    try:
        path = resolve_workspace_path(image_path)
        try:
            box = pg.locateOnScreen(path, confidence=float(confidence))
        except TypeError:
            box = pg.locateOnScreen(path)
        if box is None:
            return f"Image not found on screen: {path}"
        return f"Found at left={box.left}, top={box.top}, width={box.width}, height={box.height}"
    except Exception as e:
        return f"Error locating image: {e}"


# ── Clipboard ───────────────────────────────────────────────────────────

@tool
def computer_clipboard_get() -> str:
    """Return the current text contents of the system clipboard."""
    clip, err = _import_pyperclip()
    if err:
        return err
    try:
        return clip.paste() or "(clipboard is empty)"
    except Exception as e:
        return f"Error reading clipboard: {e}"


@tool
def computer_clipboard_set(text: str) -> str:
    """Write *text* into the system clipboard."""
    clip, err = _import_pyperclip()
    if err:
        return err
    try:
        clip.copy(text)
        return f"Clipboard set ({len(text)} chars)"
    except Exception as e:
        return f"Error setting clipboard: {e}"


# ── Browser & apps ──────────────────────────────────────────────────────

@tool
def computer_open_url(url: str, browser: str = "") -> str:
    """
    Open *url* in the system default browser (or a specific browser when
    *browser* names a registered webbrowser, e.g. 'chrome', 'safari').
    """
    try:
        if browser:
            try:
                ctrl = webbrowser.get(browser)
            except webbrowser.Error:
                return f"Error: browser '{browser}' not registered. Try empty string for default."
            opened = ctrl.open(url, new=2)
        else:
            opened = webbrowser.open(url, new=2)
        return f"Opened URL: {url}" if opened else f"Failed to open URL: {url}"
    except Exception as e:
        return f"Error opening URL: {e}"


@tool
def computer_open_app(app_name: str) -> str:
    """
    Launch a native application by name.

    Examples:
      - macOS:   ``computer_open_app('Safari')``
      - Windows: ``computer_open_app('notepad')``
      - Linux:   ``computer_open_app('gedit')``
    """
    try:
        system = platform.system()
        if system == "Darwin":
            cmd = ["open", "-a", app_name]
        elif system == "Windows":
            cmd = ["cmd", "/c", "start", "", app_name]
        else:
            cmd = ["xdg-open", app_name]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f"Launched application: {app_name}"
    except FileNotFoundError as e:
        return f"Error: command not found ({e}). Is the OS launcher available?"
    except Exception as e:
        return f"Error launching app: {e}"


@tool
def computer_run_command(cmd: str, timeout: int = 30) -> str:
    """
    Execute a shell command and return its stdout/stderr. Use with care –
    this runs with the same permissions as LocalAgent.
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=int(timeout),
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        parts = [f"exit_code={result.returncode}"]
        if out:
            parts.append(f"stdout:\n{out}")
        if err:
            parts.append(f"stderr:\n{err}")
        return "\n".join(parts) if parts else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error running command: {e}"


@tool
def computer_alert(message: str, title: str = "LocalAgent") -> str:
    """
    Show a native confirmation dialog with *message*. Falls back to
    AppleScript on macOS or stdout on other systems if pyautogui is
    unavailable.
    """
    pg, err = _import_pyautogui()
    if not err and pg is not None:
        try:
            pg.alert(text=str(message), title=str(title))
            return "Alert dismissed"
        except Exception:
            pass
    if platform.system() == "Darwin":
        try:
            esc = str(message).replace('"', '\\"')
            ttl = str(title).replace('"', '\\"')
            subprocess.run(
                ["osascript", "-e", f'display dialog "{esc}" with title "{ttl}" buttons {{"OK"}}'],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return "Alert dismissed (osascript)"
        except Exception as exc:
            return f"Error displaying alert: {exc}"
    print(f"[ALERT][{title}] {message}", file=sys.stderr)
    return "Alert printed to stderr (no GUI available)"


# ── Category metadata ───────────────────────────────────────────────────

_ALL_TOOLS = [
    computer_get_screen_size,
    computer_get_mouse_position,
    computer_move_mouse,
    computer_click,
    computer_double_click,
    computer_right_click,
    computer_drag,
    computer_scroll,
    computer_type_text,
    computer_press_key,
    computer_hotkey,
    computer_screenshot,
    computer_locate_on_screen,
    computer_clipboard_get,
    computer_clipboard_set,
    computer_open_url,
    computer_open_app,
    computer_run_command,
    computer_alert,
]

for _t in _ALL_TOOLS:
    _t.metadata = _t.metadata or {}
    _t.metadata["category"] = "computer"

TOOLS = _ALL_TOOLS
