"""
Configuration management for LocalAgent.

Priority (high → low): environment variables → config.yaml → defaults.
All config is accessed via the singleton `get_settings()`.

LLM configuration is now unified in config.yaml (config/llm.yaml has been removed).
Each provider supports a `models` list that controls which models are available
for the /model command.
"""
from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configuration must be found relative to the installed/copied project, not
# relative to whichever directory happened to launch Python.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Settings(BaseSettings):
    """
    Application settings.

    Loaded from (in priority order):
      1. Environment variables (e.g. OLLAMA_BASE_URL)
      2. .env file
      3. config.yaml
      4. Hard-coded defaults below
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ── LLM Provider ────────────────────────────────────────────────────────
    llm_provider: str = Field(
        default="",
        alias="LLM_PROVIDER",
        description="LLM provider: 'ollama' or 'openai' (auto-selected if empty)",
    )

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
        description="Ollama server URL",
    )
    ollama_default_model: str = Field(
        default="qwen2.5:7b",
        alias="OLLAMA_DEFAULT_MODEL",
        description="Default Ollama model name",
    )
    ollama_models: List[str] = Field(
        default_factory=list,
        alias="OLLAMA_MODELS",
        description="List of Ollama models available for /model switching",
    )
    ollama_timeout: int = Field(
        default=120,
        alias="OLLAMA_TIMEOUT",
        description="Ollama request timeout in seconds",
    )

    # ── OpenAI ──────────────────────────────────────────────────────────────
    openai_api_key: str = Field(
        default="",
        alias="OPENAI_API_KEY",
        description="OpenAI API key",
    )
    openai_base_url: str = Field(
        default="",
        alias="OPENAI_BASE_URL",
        description="OpenAI API base URL (for compatible APIs)",
    )
    openai_default_model: str = Field(
        default="gpt-3.5-turbo",
        alias="OPENAI_DEFAULT_MODEL",
        description="Default OpenAI model name",
    )
    openai_models: List[str] = Field(
        default_factory=list,
        alias="OPENAI_MODELS",
        description="List of OpenAI models available for /model switching",
    )

    # ── Wanqing ─────────────────────────────────────────────────────────────
    wanqing_api_key: str = Field(
        default="",
        alias="WANQING_API_KEY",
        description="Wanqing API key",
    )
    wanqing_base_url: str = Field(
        default="https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints",
        alias="WANQING_BASE_URL",
        description="Wanqing API base URL",
    )
    wanqing_default_model: str = Field(
        default="ep-116jmc-1778070434284074729",
        alias="WANQING_DEFAULT_MODEL",
        description="Default Wanqing model name",
    )
    wanqing_models: List[str] = Field(
        default_factory=list,
        alias="WANQING_MODELS",
        description="List of Wanqing models available for /model switching",
    )
    wanqing_timeout: int = Field(
        default=120,
        alias="WANQING_TIMEOUT",
        description="Wanqing request timeout in seconds",
    )

    # ── Claude Code (Anthropic) ──────────────────────────────────────────
    claude_code_api_key: str = Field(
        default="",
        alias="CLAUDE_CODE_API_KEY",
        description="Anthropic API key for Claude Code",
    )
    claude_code_base_url: str = Field(
        default="",
        alias="CLAUDE_CODE_BASE_URL",
        description="Claude Code API base URL (leave empty for default Anthropic endpoint)",
    )
    claude_code_default_model: str = Field(
        default="claude-opus-4-5",
        alias="CLAUDE_CODE_DEFAULT_MODEL",
        description="Default Claude Code model name",
    )
    claude_code_models: List[str] = Field(
        default_factory=list,
        alias="CLAUDE_CODE_MODELS",
        description="List of Claude Code models available for /model switching",
    )
    claude_code_timeout: int = Field(
        default=120,
        alias="CLAUDE_CODE_TIMEOUT",
        description="Claude Code request timeout in seconds",
    )

    # ── Agent ───────────────────────────────────────────────────────────────
    agent_max_iterations: int = Field(
        default=50,
        alias="AGENT_MAX_ITERATIONS",
        description="Maximum ReAct loop iterations",
    )
    agent_verbose: bool = Field(
        default=True,
        alias="AGENT_VERBOSE",
    )
    # 本地模型工具调用增强配置
    agent_tool_call_retry: bool = Field(
        default=True,
        alias="AGENT_TOOL_CALL_RETRY",
        description="Enable tool-call guidance retry for weak local models",
    )
    agent_max_tool_retry: int = Field(
        default=2,
        alias="AGENT_MAX_TOOL_RETRY",
        description="Max number of tool-call guidance retry attempts",
    )
    agent_max_tools_for_local_model: int = Field(
        default=20,
        alias="AGENT_MAX_TOOLS_FOR_LOCAL_MODEL",
        description="Max number of tools to pass to local models (0 = no limit)",
    )
    
    # ── Context Management (防止上下文爆炸) ──────────────────────────────────
    agent_max_tool_result_length: int = Field(
        default=0,
        alias="AGENT_MAX_TOOL_RESULT_LENGTH",
        description="Maximum character length for tool results (truncate if longer)",
    )
    agent_enable_tool_result_summarization: bool = Field(
        default=True,
        alias="AGENT_ENABLE_TOOL_RESULT_SUMMARIZATION",
        description="Enable LLM summarization for long tool results (>8000 chars)",
    )
    agent_message_sliding_window: int = Field(
        default=10,
        alias="AGENT_MESSAGE_SLIDING_WINDOW",
        description="Keep only last N messages in history (0 = no limit). System + recent user/AI messages are always kept.",
    )
    agent_reuse_system_prompt: bool = Field(
        default=True,
        alias="AGENT_REUSE_SYSTEM_PROMPT",
        description="Reuse system prompt across iterations instead of repeating it",
    )

    # ── Memory ──────────────────────────────────────────────────────────────
    memory_enable_long_term: bool = Field(
        default=True,
        alias="MEMORY_ENABLE_LONG_TERM",
    )
    memory_chroma_path: str = Field(
        default="./.local_agent/memory",
        alias="MEMORY_CHROMA_PATH",
    )
    memory_max_short_term: int = Field(
        default=20,
        alias="MEMORY_MAX_SHORT_TERM",
        description="Maximum messages kept in short-term context",
    )

    # ── Server ──────────────────────────────────────────────────────────────
    server_host: str = Field(default="0.0.0.0", alias="SERVER_HOST")
    server_port: int = Field(default=8080, alias="SERVER_PORT")

    # ── Security ────────────────────────────────────────────────────────────
    require_confirmation: bool = Field(
        default=False,
        alias="REQUIRE_CONFIRMATION",
        description="Require user confirmation for destructive operations",
    )
    max_file_size_mb: int = Field(
        default=100,
        alias="MAX_FILE_SIZE_MB",
    )

    # ── Skills / Tools directories ──────────────────────────────────────────
    extra_skill_dirs: List[str] = Field(
        default_factory=list,
        alias="EXTRA_SKILL_DIRS",
        description="Additional directories to scan for custom skills",
    )
    extra_tool_dirs: List[str] = Field(
        default_factory=list,
        alias="EXTRA_TOOL_DIRS",
        description="Additional directories to scan for custom tools",
    )

    # ── MCP ─────────────────────────────────────────────────────────────────
    mcp_config_path: str = Field(
        default="./config/mcp.json",
        alias="MCP_CONFIG_PATH",
        description="Path to the MCP configuration file",
    )
    mcp_enabled: bool = Field(
        default=True,
        alias="MCP_ENABLED",
        description="Set to False to disable MCP tool loading entirely",
    )
    mcp_servers: Dict[str, object] = Field(
        default_factory=dict,
        description=(
            "Inline MCP server definitions from config.yaml mcp.servers section. "
            "Merged with mcp_config_path file; same-name keys here take precedence."
        ),
    )

    # ── Workspace ───────────────────────────────────────────────────────────
    workspace_config: str = Field(
        default="./workspace.yaml",
        alias="WORKSPACE_CONFIG",
        description="Path to the workspace.yaml configuration file",
    )
    workspace_dir: str = Field(
        default="",
        alias="WORKSPACE_DIR",
        description="Override workspace default directory (overrides workspace.yaml)",
    )
    workspace_skill: str = Field(
        default="",
        alias="WORKSPACE_SKILL",
        description="Override workspace skill (overrides workspace.yaml)",
    )
    workspace_auto_load: bool = Field(
        default=True,
        alias="WORKSPACE_AUTO_LOAD",
        description="Automatically load workspace.yaml if found in current or parent directories",
    )

    # ── Debug ───────────────────────────────────────────────────────────────
    debug_print_mode: bool = Field(
        default=False,
        alias="DEBUG_PRINT_MODE",
        description="Master switch: enable debug print output",
    )
    debug_print_llm: bool = Field(
        default=True,
        alias="DEBUG_PRINT_LLM",
        description="Print LLM-related info: model input, output, and model selection",
    )
    debug_print_tools: bool = Field(
        default=True,
        alias="DEBUG_PRINT_TOOLS",
        description="Print tool-related info: tool calls, tools binding, and retry guidance",
    )
    debug_print_agent: bool = Field(
        default=True,
        alias="DEBUG_PRINT_AGENT",
        description="Print agent-state info: messages state and skill activation",
    )
    debug_log_file: str = Field(
        default="",
        alias="DEBUG_LOG_FILE",
        description="File path to write debug output to (empty = terminal only)",
    )

    # ────────────────────────────────────────────────────────────────────────
    # Pydantic v2: use model_post_init for side-effects after validation
    # ────────────────────────────────────────────────────────────────────────
    def model_post_init(self, __context: object) -> None:
        """Apply config.yaml overrides for values not set via environment."""
        _apply_yaml_overrides(self)

    # ── Convenience read-only properties ────────────────────────────────────
    @cached_property
    def recursion_limit(self) -> int:
        """LangGraph recursion limit = 2× max_iterations."""
        return self.agent_max_iterations * 2

    def get_configured_models(self, provider: Optional[str] = None) -> List[str]:
        """
        Return the list of models configured for the given provider.

        If the provider's `models` list is non-empty, that list is returned.
        Otherwise falls back to [default_model] so there is always at least one entry.

        Args:
            provider: Provider name ('ollama', 'openai', 'wanqing', 'claude_code').
                      Defaults to the currently active llm_provider.
        """
        target = provider or self.llm_provider or "ollama"
        if target == "ollama":
            return self.ollama_models if self.ollama_models else [self.ollama_default_model]
        elif target == "openai":
            return self.openai_models if self.openai_models else [self.openai_default_model]
        elif target == "wanqing":
            return self.wanqing_models if self.wanqing_models else [self.wanqing_default_model]
        elif target == "claude_code":
            return self.claude_code_models if self.claude_code_models else [self.claude_code_default_model]
        return [self.ollama_default_model]

    def get_all_configured_models(self) -> dict[str, list[str]]:
        """
        Return a mapping of provider → configured model list.
        Only includes providers that have either an API key (cloud) or
        at least one model configured.
        """
        result: dict[str, list[str]] = {}
        # Ollama: always include (local provider)
        result["ollama"] = self.get_configured_models("ollama")
        # Cloud providers: only include if API key is set
        if self.openai_api_key:
            result["openai"] = self.get_configured_models("openai")
        if self.wanqing_api_key:
            result["wanqing"] = self.get_configured_models("wanqing")
        if self.claude_code_api_key:
            result["claude_code"] = self.get_configured_models("claude_code")
        return result


def _apply_yaml_overrides(settings: Settings) -> None:
    """Read config.yaml and apply values that were *not* set by env vars."""
    if not CONFIG_PATH.exists():
        return

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return

    # Only override if the env var was NOT explicitly set
    def _env_not_set(env_key: str) -> bool:
        return os.environ.get(env_key) is None

    llm = data.get("llm", {})
    if _env_not_set("LLM_PROVIDER") and "provider" in llm:
        settings.llm_provider = llm["provider"]

    ollama = data.get("ollama", {})
    if _env_not_set("OLLAMA_BASE_URL") and "base_url" in ollama:
        settings.ollama_base_url = ollama["base_url"]
    if _env_not_set("OLLAMA_DEFAULT_MODEL") and "default_model" in ollama:
        settings.ollama_default_model = ollama["default_model"]
    if _env_not_set("OLLAMA_TIMEOUT") and "timeout" in ollama:
        settings.ollama_timeout = int(ollama["timeout"])
    if _env_not_set("OLLAMA_MODELS") and "models" in ollama:
        settings.ollama_models = list(ollama["models"])

    openai = data.get("openai", {})
    if _env_not_set("OPENAI_API_KEY") and "api_key" in openai:
        settings.openai_api_key = openai["api_key"]
    if _env_not_set("OPENAI_BASE_URL") and "base_url" in openai:
        settings.openai_base_url = openai["base_url"]
    if _env_not_set("OPENAI_DEFAULT_MODEL") and "default_model" in openai:
        settings.openai_default_model = openai["default_model"]
    if _env_not_set("OPENAI_MODELS") and "models" in openai:
        settings.openai_models = list(openai["models"])

    wanqing = data.get("wanqing", {})
    if _env_not_set("WANQING_API_KEY") and "api_key" in wanqing:
        settings.wanqing_api_key = wanqing["api_key"]
    if _env_not_set("WANQING_BASE_URL") and "base_url" in wanqing:
        settings.wanqing_base_url = wanqing["base_url"]
    if _env_not_set("WANQING_DEFAULT_MODEL") and "default_model" in wanqing:
        settings.wanqing_default_model = wanqing["default_model"]
    if _env_not_set("WANQING_TIMEOUT") and "timeout" in wanqing:
        settings.wanqing_timeout = int(wanqing["timeout"])
    if _env_not_set("WANQING_MODELS") and "models" in wanqing:
        settings.wanqing_models = list(wanqing["models"])

    claude_code = data.get("claude_code", {})
    if _env_not_set("CLAUDE_CODE_API_KEY") and "api_key" in claude_code:
        settings.claude_code_api_key = claude_code["api_key"]
    if _env_not_set("CLAUDE_CODE_BASE_URL") and "base_url" in claude_code:
        settings.claude_code_base_url = claude_code["base_url"]
    if _env_not_set("CLAUDE_CODE_DEFAULT_MODEL") and "default_model" in claude_code:
        settings.claude_code_default_model = claude_code["default_model"]
    if _env_not_set("CLAUDE_CODE_TIMEOUT") and "timeout" in claude_code:
        settings.claude_code_timeout = int(claude_code["timeout"])
    if _env_not_set("CLAUDE_CODE_MODELS") and "models" in claude_code:
        settings.claude_code_models = list(claude_code["models"])

    agent = data.get("agent", {})
    if _env_not_set("AGENT_MAX_ITERATIONS") and "max_iterations" in agent:
        settings.agent_max_iterations = int(agent["max_iterations"])
    if _env_not_set("AGENT_TOOL_CALL_RETRY") and "tool_call_retry" in agent:
        settings.agent_tool_call_retry = bool(agent["tool_call_retry"])
    if _env_not_set("AGENT_MAX_TOOL_RETRY") and "max_tool_retry" in agent:
        settings.agent_max_tool_retry = int(agent["max_tool_retry"])
    if _env_not_set("AGENT_MAX_TOOLS_FOR_LOCAL_MODEL") and "max_tools_for_local_model" in agent:
        settings.agent_max_tools_for_local_model = int(agent["max_tools_for_local_model"])

    memory = data.get("memory", {})
    if _env_not_set("MEMORY_CHROMA_PATH") and "chroma_path" in memory:
        settings.memory_chroma_path = memory["chroma_path"]

    server = data.get("server", {})
    if _env_not_set("SERVER_PORT") and "port" in server:
        settings.server_port = int(server["port"])
    if _env_not_set("SERVER_HOST") and "host" in server:
        settings.server_host = server["host"]

    mcp = data.get("mcp", {})
    if _env_not_set("MCP_CONFIG_PATH") and "config_path" in mcp:
        settings.mcp_config_path = mcp["config_path"]
    if _env_not_set("MCP_ENABLED") and "enabled" in mcp:
        settings.mcp_enabled = bool(mcp["enabled"])
    # Inline server definitions (config.yaml mcp.servers section)
    if "servers" in mcp and isinstance(mcp["servers"], dict):
        settings.mcp_servers = dict(mcp["servers"])

    workspace = data.get("workspace", {})
    if _env_not_set("WORKSPACE_CONFIG") and "config" in workspace:
        settings.workspace_config = str(workspace["config"])
    if _env_not_set("WORKSPACE_DIR") and "default_dir" in workspace:
        settings.workspace_dir = str(workspace["default_dir"])
    if _env_not_set("WORKSPACE_SKILL") and "skill" in workspace:
        settings.workspace_skill = str(workspace["skill"])
    if _env_not_set("WORKSPACE_AUTO_LOAD") and "auto_load" in workspace:
        settings.workspace_auto_load = bool(workspace["auto_load"])

    debug = data.get("debug", {})
    if _env_not_set("DEBUG_PRINT_MODE") and "print_mode" in debug:
        settings.debug_print_mode = bool(debug["print_mode"])
    if _env_not_set("DEBUG_PRINT_LLM") and "print_llm" in debug:
        settings.debug_print_llm = bool(debug["print_llm"])
    if _env_not_set("DEBUG_PRINT_TOOLS") and "print_tools" in debug:
        settings.debug_print_tools = bool(debug["print_tools"])
    if _env_not_set("DEBUG_PRINT_AGENT") and "print_agent" in debug:
        settings.debug_print_agent = bool(debug["print_agent"])
    if _env_not_set("DEBUG_LOG_FILE") and "log_file" in debug:
        settings.debug_log_file = str(debug["log_file"])


# ── Singleton ────────────────────────────────────────────────────────────────
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return the global Settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the singleton (useful in tests)."""
    global _settings
    _settings = None
