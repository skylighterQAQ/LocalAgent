"""
LocalAgent – Public API
"""
from local_agent.core.agent import LocalAgent
from local_agent.core.config import Settings, get_settings, reset_settings

__version__ = "0.1.0"
__all__ = ["LocalAgent", "Settings", "get_settings", "reset_settings"]
