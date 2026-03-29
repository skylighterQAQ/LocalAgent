#!/usr/bin/env python3
"""
OpenClaw - Local AI Agent powered by Ollama + LangGraph

Usage:
  python main.py                          # Interactive mode
  python main.py "your task here"         # Single-shot mode
  python main.py --config my_config.yaml  # Custom config
  python main.py --model llama3.2:3b      # Override model
  python main.py --list-tools            # List available tools
"""

import sys
import os
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def setup_logging(level: str, log_file: str) -> None:
    """
    日志配置
    args:
        level: 日志级别 DEBUG INFO WARNING ERROR
        log_file: 日志文件
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stderr)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    # Suppress noisy loggers
    for noisy in ("httpx", "httpcore", "urllib3", "langchain"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="openclaw",
        description="OpenClaw — Local AI Agent powered by Ollama + LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("task", nargs="?", help="Task to execute (single-shot mode)")
    parser.add_argument("--config", "-c", help="Path to config YAML file")
    parser.add_argument("--model", "-m", help="Ollama model to use (overrides config)")
    parser.add_argument("--list_tools", action="store_true", help="List available tools and exit")
    parser.add_argument("--no-memory", action="store_true", help="Disable conversation memory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ──────────────────────────────────────────────────────────
    from core.config_loader import load_config, get_config
    cfg = load_config(args.config)

    if args.model:
        cfg.ollama.model = args.model

    # 配置日志输出
    setup_logging(
        "DEBUG" if args.verbose else cfg.logging.level,
        cfg.logging.file,
    )

    # ── Load tools ───────────────────────────────────────────────────────────
    from core.tool_loader import load_tools
    from core.tool_base import get_registry

    load_tools(cfg.tools)
    registry = get_registry()
    tools_info = registry.list_tools()

    if args.list_tools:
        from ui.cli import print_tools_table
        from rich.console import Console
        Console().print(f"\nRegistered tools ({len(tools_info)}):")
        print_tools_table(tools_info)
        return

    # ── Build agent ───────────────────────────────────────────────────────────
    from rich.console import Console
    console = Console()

    try:
        with console.status("[cyan]Connecting to Ollama and building agent...[/cyan]"):
            from core.agent import create_agent
            graph, llm = create_agent()
    except Exception as e:
        console.print(f"[bold red]Failed to create agent:[/bold red] {e}")
        console.print("\n[dim]Make sure Ollama is running: ollama serve[/dim]")
        console.print(f"[dim]And the model is pulled: ollama pull {cfg.ollama.model}[/dim]")
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.task:
        # Single-shot mode
        from core.agent import run_agent
        from ui.cli import print_response, show_thinking
        console.print(f"[dim]Task: {args.task}[/dim]")
        with show_thinking():
            response = run_agent(graph, args.task)
        print_response(response)
    else:
        # Interactive REPL
        from ui.cli import run_interactive
        run_interactive(graph, cfg.ollama.model, tools_info)


if __name__ == "__main__":
    main()
