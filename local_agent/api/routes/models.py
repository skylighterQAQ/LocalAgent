"""Models API Routes"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_models():
    """
    List models configured in config.yaml, grouped by provider.
    
    Only models explicitly listed under each provider's `models` key (or 
    default_model if no list is configured) are returned. This ensures the 
    /model endpoint reflects exactly what the user has configured.
    """
    from local_agent.llm.providers.ollama import OllamaProvider
    from local_agent.core.config import get_settings

    settings = get_settings()
    providers: dict[str, dict[str, object]] = {}
    result: dict[str, object] = {
        "current_provider": settings.llm_provider,
        "providers": providers,
    }

    # Ollama: check live connection; models come from config.yaml
    ollama_provider = OllamaProvider()
    ollama_connected = ollama_provider.check_connection()
    configured_ollama = settings.get_configured_models("ollama")
    providers["ollama"] = {
        "available": ollama_connected,
        "models": configured_ollama,
        "default_model": settings.ollama_default_model,
    }

    # OpenAI: available if API key is set; models from config.yaml
    if settings.openai_api_key:
        providers["openai"] = {
            "available": True,
            "models": settings.get_configured_models("openai"),
            "default_model": settings.openai_default_model,
        }
    else:
        providers["openai"] = {
            "available": False,
            "models": [],
            "default_model": settings.openai_default_model,
        }

    # Wanqing: available if API key is set; models from config.yaml
    if settings.wanqing_api_key:
        providers["wanqing"] = {
            "available": True,
            "models": settings.get_configured_models("wanqing"),
            "default_model": settings.wanqing_default_model,
        }
    else:
        providers["wanqing"] = {
            "available": False,
            "models": [],
            "default_model": settings.wanqing_default_model,
        }

    # Claude Code: available if API key is set; models from config.yaml
    if settings.claude_code_api_key:
        providers["claude_code"] = {
            "available": True,
            "models": settings.get_configured_models("claude_code"),
            "default_model": settings.claude_code_default_model,
        }
    else:
        providers["claude_code"] = {
            "available": False,
            "models": [],
            "default_model": settings.claude_code_default_model,
        }

    return result


@router.get("/status")
async def providers_status():
    """Check all providers connection status"""
    from local_agent.llm.providers.ollama import OllamaProvider
    from local_agent.llm.providers.claude_code import ClaudeCodeProvider
    from local_agent.core.config import get_settings

    settings = get_settings()
    ollama_provider = OllamaProvider()
    ollama_connected = ollama_provider.check_connection()

    claude_provider = ClaudeCodeProvider()

    return {
        "current_provider": settings.llm_provider,
        "providers": {
            "ollama": {
                "connected": ollama_connected,
                "base_url": settings.ollama_base_url,
                "default_model": settings.ollama_default_model,
                "models": settings.get_configured_models("ollama"),
            },
            "openai": {
                "connected": bool(settings.openai_api_key),
                "base_url": settings.openai_base_url or "https://api.openai.com/v1",
                "default_model": settings.openai_default_model,
                "models": settings.get_configured_models("openai") if settings.openai_api_key else [],
            },
            "wanqing": {
                "connected": bool(settings.wanqing_api_key),
                "base_url": settings.wanqing_base_url,
                "default_model": settings.wanqing_default_model,
                "models": settings.get_configured_models("wanqing") if settings.wanqing_api_key else [],
            },
            "claude_code": {
                "connected": claude_provider.check_connection(),
                "base_url": settings.claude_code_base_url or "https://api.anthropic.com",
                "default_model": settings.claude_code_default_model,
                "models": settings.get_configured_models("claude_code") if settings.claude_code_api_key else [],
            },
        }
    }
