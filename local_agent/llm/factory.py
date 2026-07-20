"""
local_agent.llm.factory
========================
LLM Provider 单例工厂，提供全局共享的 Provider 实例。

原 local_agent.shared.providers 中的 get_llm_provider() 已迁移至此。
"""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from local_agent.llm.providers.ollama import OllamaProvider
    from local_agent.llm.providers.openai import OpenAIProvider
    from local_agent.llm.providers.wanqing import WanqingProvider
    from local_agent.llm.providers.claude_code import ClaudeCodeProvider

logger = logging.getLogger(__name__)

# Singleton instance
_llm_provider: Optional[Union["OllamaProvider", "OpenAIProvider", "WanqingProvider", "ClaudeCodeProvider"]] = None


def get_llm_provider(
    model: Optional[str] = None,
    provider: Optional[str] = None
) -> Union["OllamaProvider", "OpenAIProvider", "WanqingProvider", "ClaudeCodeProvider"]:
    """
    Get or create the shared LLM Provider instance.
    
    Args:
        model: Optional model name. If provided, creates a new provider with this model.
              If None, returns existing provider or creates one with default model.
        provider: Optional provider type ('ollama', 'openai', 'wanqing', or 'claude_code').
                  If None, uses config default.
    
    Returns:
        OllamaProvider, OpenAIProvider, WanqingProvider, or ClaudeCodeProvider instance
    """
    global _llm_provider
    
    from local_agent.core.config import get_settings
    
    settings = get_settings()
    provider_type = provider or settings.llm_provider
    
    if model is not None or provider is not None:
        # Create new provider if model or provider is specified
        if provider_type == "openai":
            from local_agent.llm.providers.openai import OpenAIProvider
            return OpenAIProvider(model=model or settings.openai_default_model)
        elif provider_type == "wanqing":
            from local_agent.llm.providers.wanqing import WanqingProvider
            return WanqingProvider(model=model or settings.wanqing_default_model)
        elif provider_type == "claude_code":
            from local_agent.llm.providers.claude_code import ClaudeCodeProvider
            return ClaudeCodeProvider(model=model or settings.claude_code_default_model)
        else:
            from local_agent.llm.providers.ollama import OllamaProvider
            return OllamaProvider(model=model or settings.ollama_default_model)
    
    if _llm_provider is None:
        if provider_type == "openai":
            from local_agent.llm.providers.openai import OpenAIProvider
            _llm_provider = OpenAIProvider(model=settings.openai_default_model)
            logger.debug(f"Created shared OpenAI provider with model: {settings.openai_default_model}")
        elif provider_type == "wanqing":
            from local_agent.llm.providers.wanqing import WanqingProvider
            _llm_provider = WanqingProvider(model=settings.wanqing_default_model)
            logger.debug(f"Created shared Wanqing provider with model: {settings.wanqing_default_model}")
        elif provider_type == "claude_code":
            from local_agent.llm.providers.claude_code import ClaudeCodeProvider
            _llm_provider = ClaudeCodeProvider(model=settings.claude_code_default_model)
            logger.debug(f"Created shared Claude Code provider with model: {settings.claude_code_default_model}")
        else:
            from local_agent.llm.providers.ollama import OllamaProvider
            _llm_provider = OllamaProvider(model=settings.ollama_default_model)
            logger.debug(f"Created shared Ollama provider with model: {settings.ollama_default_model}")
    
    return _llm_provider


def reset_llm_provider() -> None:
    """Reset the shared LLM provider singleton (useful for testing or config changes)."""
    global _llm_provider
    _llm_provider = None
    logger.debug("Reset shared LLM provider")
