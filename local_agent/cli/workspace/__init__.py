"""
LocalAgent CLI Workspace Module

工作区管理模块，已迁移至 local_agent.cli.workspace。
"""
from local_agent.cli.workspace.config import WorkspaceConfig
from local_agent.cli.workspace.manager import WorkspaceManager, find_workspace, WORKSPACE_FILENAME
from local_agent.cli.workspace.state import (
    save_last_workspace,
    load_last_workspace,
    clear_last_workspace,
)
from local_agent.cli.workspace.context import (
    set_active_workspace,
    get_active_workspace,
    get_active_workspace_dir,
    resolve_workspace_path,
)

__all__ = [
    "WorkspaceConfig",
    "WorkspaceManager",
    "find_workspace",
    "WORKSPACE_FILENAME",
    "save_last_workspace",
    "load_last_workspace",
    "clear_last_workspace",
    "set_active_workspace",
    "get_active_workspace",
    "get_active_workspace_dir",
    "resolve_workspace_path",
]
