#!/usr/bin/env python3
"""
Quick start script for LocalAgent
Tests basic functionality and provides setup guidance
"""
import sys
import subprocess


def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 11):
        print(f"❌ Python 3.11+ required, found {v.major}.{v.minor}")
        return False
    print(f"✅ Python {v.major}.{v.minor}.{v.micro}")
    return True


def check_ollama():
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"✅ Ollama running ({len(models)} models: {', '.join(models[:3]) or 'none'})")
        if not models:
            print("   ⚠️  Pull a model: ollama pull qwen2.5:7b")
        return True
    except Exception:
        print("❌ Ollama not running. Start with: ollama serve")
        print("   Install: https://ollama.ai")
        return False


def check_deps():
    required = ["langchain", "langchain_ollama", "langgraph", "fastapi", "typer", "rich"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("   Run: pip install -e .")
        return False
    print("✅ Core dependencies installed")
    return True


def main():
    print("=" * 50)
    print("🤖 LocalAgent - Setup Check")
    print("=" * 50)

    ok = True
    ok = check_python() and ok
    ok = check_deps() and ok
    check_ollama()  # Not blocking

    print()
    if ok:
        print("🚀 Ready! Try:")
        print("   la                      # Interactive chat")
        print("   la chat run 'Hello!'    # Single message")
        print("   la server start         # Start web UI")
        print("   la skills list          # List skills")
        print("   la tools list           # List tools")
    else:
        print("⚠️  Fix the issues above, then run: pip install -e .")


if __name__ == "__main__":
    main()
