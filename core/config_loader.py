"""Configuration management for OpenClaw."""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    temperature: float = 0.1
    num_ctx: int = 8192


class AgentConfig(BaseModel):
    max_iterations: int = 20
    verbose: bool = True
    memory_enabled: bool = True


class CodeExecConfig(BaseModel):
    timeout: int = 30
    sandbox: bool = False
    allowed_modules: list[str] = Field(default_factory=list)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "openclaw.log"


class AppConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: list[str] = Field(default_factory=lambda: ["browser", "code_exec", "web_search", "file_ops"])
    code_exec: CodeExecConfig = Field(default_factory=CodeExecConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


_config: Optional[AppConfig] = None


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML file."""
    global _config

    if config_path is None:
        # Search common locations
        search_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path.home() / ".LocalAgent" / "config.yaml",
        ]
        for p in search_paths:
            if p.exists():
                config_path = str(p)
                break

    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _config = AppConfig(**data)
    else:
        _config = AppConfig()

    # Allow env var overrides
    if model := os.getenv("OLLAMA_MODEL"):
        _config.ollama.model = model
    if url := os.getenv("OLLAMA_BASE_URL"):
        _config.ollama.base_url = url

    return _config


def get_config() -> AppConfig:
    """Get the current configuration (loads default if not yet loaded)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
