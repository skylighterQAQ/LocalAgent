"""
SubAgent - Main class for defining and executing linear workflows
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from local_agent.core.subagent.context import ExecutionContext
from local_agent.core.subagent.steps.base import BaseStep

logger = logging.getLogger(__name__)


class SubAgent:
    """
    SubAgent 主类，用于定义和执行线性工作流
    
    支持功能：
    - 添加步骤（LLM/Tool/SubAgent）
    - 顺序执行步骤
    - 上下文管理和数据传递
    - 执行历史和监控
    
    示例::
    
        from local_agent.core.subagent import SubAgent, LLMStep, ToolStep
        
        # 创建 SubAgent
        agent = SubAgent(name="document_analyzer")
        
        # 添加步骤（支持链式调用）
        agent.add_step(ToolStep(
            tool="file_reader",
            params={"path": "doc.txt"},
            output_key="content"
        )).add_step(LLMStep(
            model="qwen2.5:7b",
            prompt="分析: {content}",
            output_key="analysis"
        )).add_step(ToolStep(
            tool="file_writer",
            params={"path": "result.txt", "content": "{analysis}"}
        ))
        
        # 执行
        result = agent.run(input_data={"filename": "doc.txt"})
        
        # 查看执行摘要
        print(agent.get_execution_summary())
    """
    
    def __init__(self, name: str = "SubAgent", description: str = ""):
        """
        初始化 SubAgent
        
        Args:
            name: SubAgent 名称
            description: SubAgent 描述
        """
        self.name = name
        self.description = description
        self.steps: List[BaseStep] = []
        self.context: Optional[ExecutionContext] = None
        
        logger.info(f"Created SubAgent: {name}")
    
    def add_step(self, step: BaseStep) -> "SubAgent":
        """
        添加步骤到 SubAgent（支持链式调用）
        
        Args:
            step: 要添加的步骤
            
        Returns:
            self（支持链式调用）
        """
        if not isinstance(step, BaseStep):
            raise TypeError(f"Expected BaseStep, got {type(step)}")
        
        self.steps.append(step)
        logger.debug(f"Added step to SubAgent '{self.name}': {step}")
        return self
    
    def add_steps(self, steps: List[BaseStep]) -> "SubAgent":
        """
        批量添加步骤
        
        Args:
            steps: 步骤列表
            
        Returns:
            self（支持链式调用）
        """
        for step in steps:
            self.add_step(step)
        return self
    
    def run(
        self,
        input_data: Optional[Dict[str, Any]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> Dict[str, Any]:
        """
        执行 SubAgent
        
        Args:
            input_data: 输入数据字典
            context: 可选的执行上下文（如果不提供会自动创建）
            
        Returns:
            执行结果字典，包含：
            - 所有上下文变量
            - 执行摘要
            - 最后一个步骤的输出（如果有）
        """
        # 创建或使用提供的上下文
        if context is None:
            context = ExecutionContext(initial_data=input_data)
        else:
            if input_data:
                context.update(input_data)
        
        self.context = context
        
        logger.info(f"Starting SubAgent '{self.name}' with {len(self.steps)} steps")
        
        # 顺序执行所有步骤
        last_result = None
        for i, step in enumerate(self.steps, 1):
            logger.info(f"Executing step {i}/{len(self.steps)}: {step.name}")
            
            try:
                result = step.run(context)
                last_result = result
                
            except Exception as e:
                logger.error(f"SubAgent '{self.name}' failed at step {i} ('{step.name}'): {e}")
                # 如果步骤配置为抛出异常，则中断执行
                if step.on_error == "raise":
                    raise
                # 否则继续执行下一步
        
        # 构建返回结果
        result = {
            "variables": context.get_all(),
            "summary": context.get_execution_summary(),
            "last_result": last_result,
        }
        
        logger.info(
            f"SubAgent '{self.name}' completed: "
            f"{result['summary']['success']}/{result['summary']['total_steps']} steps succeeded"
        )
        
        return result
    
    def validate(self) -> bool:
        """
        验证 SubAgent 配置
        
        Returns:
            是否有效
        """
        if not self.steps:
            logger.warning(f"SubAgent '{self.name}' has no steps")
            return False
        
        for i, step in enumerate(self.steps, 1):
            if not isinstance(step, BaseStep):
                logger.error(f"Step {i} is not a BaseStep instance: {type(step)}")
                return False
        
        logger.info(f"SubAgent '{self.name}' validation passed: {len(self.steps)} steps")
        return True
    
    def get_execution_summary(self) -> Optional[Dict[str, Any]]:
        """
        获取最近一次执行的摘要
        
        Returns:
            执行摘要字典，如果尚未执行则返回 None
        """
        if self.context is None:
            return None
        
        return self.context.get_execution_summary()
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        获取最近一次执行的详细历史
        
        Returns:
            执行历史列表，如果尚未执行则返回空列表
        """
        if self.context is None:
            return []
        
        return self.context.get_history()
    
    def __len__(self) -> int:
        """返回步骤数量"""
        return len(self.steps)
    
    def __repr__(self) -> str:
        return f"SubAgent(name='{self.name}', steps={len(self.steps)})"
    
    def __str__(self) -> str:
        """生成可读的字符串表示"""
        lines = [f"SubAgent: {self.name}"]
        if self.description:
            lines.append(f"Description: {self.description}")
        lines.append(f"Steps ({len(self.steps)}):")
        
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  {i}. {step}")
        
        return "\n".join(lines)
