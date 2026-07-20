"""
System Utility Tools - datetime, clipboard, system info, calculator
"""
from local_agent.core.tools import tool


@tool
def datetime_get(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Get current date and time.
    Args:
        format: strftime format string (default: '%Y-%m-%d %H:%M:%S')
    """
    from datetime import datetime
    import time
    now = datetime.now()
    return (
        f"Current datetime: {now.strftime(format)}\n"
        f"Timezone: {time.tzname[0]}\n"
        f"Timestamp: {int(now.timestamp())}\n"
        f"Day of week: {now.strftime('%A')}\n"
        f"Week of year: {now.strftime('%W')}"
    )


@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations.
    Supports: +, -, *, /, **, //, %, and math functions (sin, cos, sqrt, log, etc.)
    Args:
        expression: Mathematical expression to evaluate
    """
    import math
    try:
        # Safe math evaluation
        safe_dict = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "int": int, "float": float, "pow": pow,
            "pi": math.pi, "e": math.e, "inf": math.inf,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
            "exp": math.exp, "ceil": math.ceil, "floor": math.floor,
            "factorial": math.factorial, "gcd": math.gcd,
        }
        result = eval(expression, safe_dict)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def clipboard_copy(text: str) -> str:
    """Copy text to the system clipboard."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return f"Copied to clipboard: {text[:100]}{'...' if len(text) > 100 else ''}"
    except ImportError:
        return "Error: pyperclip not installed. Run: pip install pyperclip"
    except Exception as e:
        return f"Clipboard error: {e}"


@tool
def clipboard_paste() -> str:
    """Get text from the system clipboard."""
    try:
        import pyperclip
        content = pyperclip.paste()
        return f"Clipboard content:\n{content}"
    except ImportError:
        return "Error: pyperclip not installed"
    except Exception as e:
        return f"Clipboard error: {e}"


@tool
def system_info() -> str:
    """Get information about the current system (OS, CPU, memory, disk)."""
    try:
        import platform
        import psutil

        cpu_freq = psutil.cpu_freq()
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        info = [
            f"OS: {platform.system()} {platform.release()} ({platform.machine()})",
            f"Python: {platform.python_version()}",
            f"CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count()} logical)",
            f"CPU Speed: {cpu_freq.current:.0f} MHz" if cpu_freq else "CPU Speed: N/A",
            f"CPU Usage: {psutil.cpu_percent(interval=1):.1f}%",
            f"Memory: {mem.used / 1e9:.1f} GB used / {mem.total / 1e9:.1f} GB total ({mem.percent:.1f}%)",
            f"Disk: {disk.used / 1e9:.1f} GB used / {disk.total / 1e9:.1f} GB total ({disk.percent:.1f}%)",
            f"Hostname: {platform.node()}",
        ]
        return "\n".join(info)
    except ImportError:
        import platform
        return f"OS: {platform.system()} {platform.release()}\nPython: {platform.python_version()}"
    except Exception as e:
        return f"Error getting system info: {e}"


@tool
def network_info() -> str:
    """Get network interface information."""
    try:
        import psutil
        import socket

        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
        except Exception:
            local_ip = "N/A"

        lines = [f"Hostname: {hostname}", f"Local IP: {local_ip}", "\nNetwork interfaces:"]

        for iface, addrs in psutil.net_if_addrs().items():
            for addr in addrs:
                if addr.family.name in ("AF_INET", "AF_INET6"):
                    lines.append(f"  {iface}: {addr.address} ({addr.family.name})")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


@tool
def sleep_seconds(seconds: float) -> str:
    """Wait for a specified number of seconds."""
    import time
    if seconds > 60:
        return "Error: Maximum sleep is 60 seconds"
    time.sleep(seconds)
    return f"Waited {seconds} seconds"


@tool
def generate_uuid() -> str:
    """Generate a random UUID."""
    import uuid
    return str(uuid.uuid4())


@tool
def hash_text(text: str, algorithm: str = "sha256") -> str:
    """
    Hash text using a cryptographic hash function.
    Supported algorithms: md5, sha1, sha256, sha512
    """
    import hashlib
    algos = {"md5": hashlib.md5, "sha1": hashlib.sha1,
             "sha256": hashlib.sha256, "sha512": hashlib.sha512}
    if algorithm not in algos:
        return f"Unknown algorithm. Use: {', '.join(algos.keys())}"
    h = algos[algorithm](text.encode()).hexdigest()
    return f"{algorithm}({text[:20]}...) = {h}" if len(text) > 20 else f"{algorithm}({text}) = {h}"


@tool
def format_json(json_string: str, indent: int = 2) -> str:
    """Parse and pretty-print a JSON string."""
    import json
    try:
        data = json.loads(json_string)
        return json.dumps(data, indent=indent, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"


datetime_get.metadata = datetime_get.metadata or {}
datetime_get.metadata["category"] = "system"
calculator.metadata = calculator.metadata or {}
calculator.metadata["category"] = "system"
clipboard_copy.metadata = clipboard_copy.metadata or {}
clipboard_copy.metadata["category"] = "system"
clipboard_paste.metadata = clipboard_paste.metadata or {}
clipboard_paste.metadata["category"] = "system"
system_info.metadata = system_info.metadata or {}
system_info.metadata["category"] = "system"
network_info.metadata = network_info.metadata or {}
network_info.metadata["category"] = "system"
sleep_seconds.metadata = sleep_seconds.metadata or {}
sleep_seconds.metadata["category"] = "system"
generate_uuid.metadata = generate_uuid.metadata or {}
generate_uuid.metadata["category"] = "system"
hash_text.metadata = hash_text.metadata or {}
hash_text.metadata["category"] = "system"
format_json.metadata = format_json.metadata or {}
format_json.metadata["category"] = "system"

TOOLS = [
    datetime_get,
    calculator,
    clipboard_copy,
    clipboard_paste,
    system_info,
    network_info,
    sleep_seconds,
    generate_uuid,
    hash_text,
    format_json,
]
