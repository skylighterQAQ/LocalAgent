"""
Tool Step - Call registered tools in SubAgent
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from local_agent.core.subagent.context import ExecutionContext
from local_agent.core.subagent.steps.base import BaseStep

logger = logging.getLogger(__name__)


class ToolStep(BaseStep):
    """
    工具调用步骤
    
    调用已注册的工具（通过 ToolRegistry）。支持：
    - 本地工具（LocalAgent 内置工具）
    - MCP 工具（通过 MCP 协议加载的外部工具）
    - 传递参数（支持变量替换）
    - 处理工具返回值
    
    示例::
    
        # 本地工具 - 文件读取
        step = ToolStep(
            name="read_file",
            tool="file_reader",
            params={"path": "{file_path}"},
            output_key="file_content"
        )
        
        # 本地工具 - 文件写入
        step = ToolStep(
            tool="file_writer",
            params={
                "path": "output.txt",
                "content": "{result}"
            }
        )
        
        # MCP 工具 - 天气查询（假设已配置 MCP 服务）
        step = ToolStep(
            tool="weather_get_current",
            params={"city": "北京"}
        )
        
        # Shell 命令
        step = ToolStep(
            tool="shell_executor",
            params={"command": "ls -la"}
        )
    
    注意::
        - MCP 工具需要先在 config/mcp.json 中配置并加载
        - 工具名称必须在 ToolRegistry 中已注册
    """
    
    def __init__(
        self,
        tool: str,
        params: Optional[Dict[str, Any]] = None,
        mcp_server: Optional[str] = None,
        auto_load_mcp: bool = True,
        **kwargs,
    ):
        """
        初始化工具步骤
        
        Args:
            tool: 工具名称（必须已注册在 ToolRegistry 中）
            params: 工具参数字典（支持变量替换）
            mcp_server: 可选，指定 MCP 服务器名称（用于 MCP 工具）
            auto_load_mcp: 如果为 True，当工具未找到时自动尝试加载 MCP 工具
            **kwargs: 其他 BaseStep 参数
        """
        super().__init__(**kwargs)
        
        self.tool_name = tool
        self.params = params or {}
        self.mcp_server = mcp_server
        self.auto_load_mcp = auto_load_mcp
        self._tool = None  # Cache the tool instance
    
    def _get_tool(self):
        """
        获取工具实例，支持自动加载 MCP 工具
        
        Returns:
            工具实例
        """
        if self._tool is not None:
            return self._tool
        
        from local_agent.tools.registry import ToolRegistry
        registry = ToolRegistry()
        
        # 尝试从 registry 获取工具
        try:
            self._tool = registry.get_tool(self.tool_name)
            if self._tool:
                return self._tool
        except KeyError:
            pass
        
        # 如果工具不存在且启用了 auto_load_mcp，尝试加载 MCP 工具
        if self.auto_load_mcp:
            logger.info(f"Tool '{self.tool_name}' not found, trying to load from MCP...")
            try:
                from local_agent.mcp.manager import MCPManager
                mcp_manager = MCPManager()
                
                # 如果指定了 MCP 服务器，只从该服务器加载
                if self.mcp_server:
                    mcp_manager.load_mcp_tools(server=self.mcp_server)
                else:
                    # 否则加载所有 MCP 工具
                    mcp_manager.load_all_mcp_tools()
                
                # 再次尝试获取工具
                self._tool = registry.get_tool(self.tool_name)
                if self._tool:
                    logger.info(f"Successfully loaded MCP tool: {self.tool_name}")
                    return self._tool
            except Exception as e:
                logger.warning(f"Failed to load MCP tool '{self.tool_name}': {e}")
        
        # 如果还是找不到工具，抛出异常
        raise ValueError(
            f"Tool '{self.tool_name}' not found in registry. "
            f"Available tools: {registry.list_tools()}"
        )
    
    def execute(self, context: ExecutionContext) -> Any:
        """
        执行工具调用
        
        Args:
            context: 执行上下文
            
        Returns:
            工具执行结果
        """
        # 获取工具实例
        tool = self._get_tool()
        
        # 解析参数（变量替换）
        resolved_params = context.resolve(self.params)
        
        logger.debug(f"ToolStep '{self.name}' calling tool '{self.tool_name}' with params: {resolved_params}")
        
        # 调用工具
        try:
            # 检查工具类型
            if hasattr(tool, 'invoke'):
                # LangChain Tool
                result = tool.invoke(resolved_params)
            elif callable(tool):
                # 普通函数
                result = tool(**resolved_params)
            else:
                raise TypeError(f"Tool '{self.tool_name}' is not callable")
            
            logger.debug(f"ToolStep '{self.name}' result: {str(result)[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"Tool '{self.tool_name}' execution failed: {e}")
            raise
    
    def __repr__(self) -> str:
        return f"ToolStep(name='{self.name}', tool='{self.tool_name}')"
