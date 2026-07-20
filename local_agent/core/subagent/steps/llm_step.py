"""
LLM Step - Call LLM models in SubAgent
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from local_agent.core.messages import HumanMessage, SystemMessage

from local_agent.core.subagent.context import ExecutionContext
from local_agent.core.subagent.steps.base import BaseStep

logger = logging.getLogger(__name__)


class LLMStep(BaseStep):
    """
    大模型调用步骤
    
    支持通过 Ollama/OpenAI 调用不同的大语言模型，可以自由选择模型、配置参数。
    
    主要参数：
    - model: 模型名称（如 "qwen2.5:7b", "gpt-3.5-turbo"）
    - prompt: 提示词模板（支持 {variable} 格式的变量替换）
    - temperature: 温度参数（0-1，控制随机性）
    - max_tokens: 最大生成长度
    - system_prompt: 系统提示词
    
    示例::
    
        # 基础用法
        step = LLMStep(
            name="summarizer",
            model="qwen2.5:7b",
            prompt="请总结以下内容: {content}",
            output_key="summary"
        )
        
        # 高级用法
        step = LLMStep(
            model="qwen2.5:14b",
            prompt="分析文本: {text}",
            temperature=0.7,
            max_tokens=1000,
            output_key="analysis"
        )
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:7b",
        prompt: str = "",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs,
    ):
        """
        初始化 LLM 步骤
        
        Args:
            model: 模型名称
            prompt: 提示词模板（支持变量替换）
            temperature: 温度参数（0-1）
            max_tokens: 最大生成长度
            system_prompt: 系统提示词
            provider: LLM 提供商（"ollama" 或 "openai"，默认自动检测）
            **kwargs: 其他 BaseStep 参数（name, output_key 等）
        """
        super().__init__(**kwargs)
        
        self.model = model
        self.prompt_template = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.provider = provider
        
        self._llm = None
    
    def _get_llm(self):
        """
        获取 LLM 实例（懒加载）
        
        Returns:
            LLM 实例
        """
        if self._llm is not None:
            return self._llm
        
        # 导入 LLM Provider
        from local_agent.llm.factory import get_llm_provider
        try:
            self._llm = get_llm_provider(model=self.model, provider=self.provider)
            return self._llm
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}")
            raise
    
    def execute(self, context: ExecutionContext) -> Any:
        """
        执行 LLM 调用
        
        Args:
            context: 执行上下文
            
        Returns:
            LLM 生成的文本结果
        """
        # 解析提示词模板
        prompt = context.resolve(self.prompt_template)
        
        logger.debug(f"LLMStep '{self.name}' prompt: {prompt[:100]}...")
        
        # 获取 LLM 实例
        llm = self._get_llm()
        
        # 构建消息
        messages = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        # 调用 LLM
        try:
            response = llm.invoke(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # 提取响应文本
            if hasattr(response, 'content'):
                result = response.content
            else:
                result = str(response)
            
            logger.debug(f"LLMStep '{self.name}' response: {result[:100]}...")
            return result
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def __repr__(self) -> str:
        return f"LLMStep(name='{self.name}', model='{self.model}')"
