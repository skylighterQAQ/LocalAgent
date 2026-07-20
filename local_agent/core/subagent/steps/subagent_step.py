"""
SubAgent Step - Call another SubAgent in a SubAgent
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

from local_agent.core.subagent.context import ExecutionContext
from local_agent.core.subagent.steps.base import BaseStep

if TYPE_CHECKING:
    from local_agent.core.subagent.subagent import SubAgent

logger = logging.getLogger(__name__)


class SubAgentStep(BaseStep):
    """
    SubAgent 嵌套调用步骤
    
    在一个 SubAgent 中调用另一个 SubAgent，支持：
    - 传递 SubAgent 实例或配置文件路径
    - 输入映射：将当前上下文的变量映射到子 SubAgent 的输入
    - 输出提取：从子 SubAgent 的结果中提取需要的数据
    
    示例::
    
        # 使用 SubAgent 实例
        summarizer = SubAgent(name="summarizer")
        summarizer.add_step(LLMStep(model="qwen2.5:7b", prompt="总结: {text}"))
        
        step = SubAgentStep(
            name="call_summarizer",
            subagent=summarizer,
            input_mapping={"text": "content"},
            output_key="summary"
        )
        
        # 使用配置文件路径
        step = SubAgentStep(
            name="call_analyzer",
            subagent="config/analyzer_agent.yaml",
            input_mapping={"doc": "file_content"},
            output_key="analysis"
        )
    """
    
    def __init__(
        self,
        subagent: Union["SubAgent", str],
        input_mapping: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        """
        初始化 SubAgent 步骤
        
        Args:
            subagent: SubAgent 实例或配置文件路径
            input_mapping: 输入映射字典
                - key: 子 SubAgent 的输入变量名
                - value: 当前上下文中的变量名
                例如: {"text": "content"} 表示将当前上下文的 content 变量
                     映射为子 SubAgent 的 text 输入
            **kwargs: 其他 BaseStep 参数
        """
        super().__init__(**kwargs)
        
        self._subagent_source = subagent
        self.input_mapping = input_mapping or {}
        self._subagent = None
    
    def _get_subagent(self) -> "SubAgent":
        """
        获取 SubAgent 实例（懒加载）
        
        Returns:
            SubAgent 实例
        """
        if self._subagent is not None:
            return self._subagent
        
        # 如果已经是 SubAgent 实例，直接返回
        if isinstance(self._subagent_source, str):
            # 从配置文件加载
            from local_agent.core.subagent.config import load_subagent_from_file
            
            try:
                self._subagent = load_subagent_from_file(self._subagent_source)
                logger.info(f"Loaded SubAgent from config: {self._subagent_source}")
            except Exception as e:
                logger.error(f"Failed to load SubAgent from '{self._subagent_source}': {e}")
                raise
        else:
            # 假设是 SubAgent 实例
            self._subagent = self._subagent_source
        
        return self._subagent
    
    def execute(self, context: ExecutionContext) -> Any:
        """
        执行 SubAgent 调用
        
        Args:
            context: 执行上下文
            
        Returns:
            SubAgent 执行结果
        """
        # 获取 SubAgent 实例
        subagent = self._get_subagent()
        
        # 准备输入数据（根据映射）
        input_data = {}
        for sub_key, ctx_key in self.input_mapping.items():
            value = context.get(ctx_key)
            if value is not None:
                input_data[sub_key] = value
            else:
                logger.warning(
                    f"SubAgentStep '{self.name}': input mapping '{ctx_key}' not found in context"
                )
        
        logger.debug(f"SubAgentStep '{self.name}' calling SubAgent '{subagent.name}' with input: {input_data}")
        
        # 调用 SubAgent
        try:
            result = subagent.run(input_data=input_data)
            logger.debug(f"SubAgentStep '{self.name}' completed with result")
            return result
            
        except Exception as e:
            logger.error(f"SubAgent '{subagent.name}' execution failed: {e}")
            raise
    
    def __repr__(self) -> str:
        subagent_name = (
            self._subagent.name if self._subagent 
            else (self._subagent_source if isinstance(self._subagent_source, str) else "unknown")
        )
        return f"SubAgentStep(name='{self.name}', subagent='{subagent_name}')"
