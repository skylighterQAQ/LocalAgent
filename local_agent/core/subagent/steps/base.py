"""
Base class for all SubAgent steps
"""
from __future__ import annotations

import logging
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from local_agent.core.subagent.context import ExecutionContext

logger = logging.getLogger(__name__)


class StepExecutionError(RuntimeError):
    """Raised when a step fails after exhausting all retries."""


class BaseStep(ABC):
    """
    SubAgent 步骤的抽象基类
    
    所有步骤类型（LLM/Tool/SubAgent）都必须继承这个类并实现 execute 方法。
    
    主要功能：
    - 定义统一的执行接口
    - 提供输入输出管理
    - 支持错误处理和重试
    - 自动记录执行时间和状态
    
    属性:
        name: 步骤名称（用于标识和日志）
        output_key: 保存输出到上下文的键名
        enabled: 是否启用此步骤
        retry_count: 失败时的重试次数
        retry_delay: 重试之间的延迟（秒）
        on_error: 错误处理策略（'raise', 'skip', 'continue'）
    
    示例::
    
        class MyStep(BaseStep):
            def execute(self, context: ExecutionContext) -> Any:
                # 实现具体逻辑
                input_data = context.get("input")
                result = do_something(input_data)
                return result
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        output_key: Optional[str] = None,
        enabled: bool = True,
        retry_count: int = 0,
        retry_delay: float = 1.0,
        on_error: str = "raise",
    ):
        """
        初始化步骤
        
        Args:
            name: 步骤名称（默认使用类名）
            output_key: 保存输出到上下文的键名
            enabled: 是否启用此步骤
            retry_count: 失败时的重试次数
            retry_delay: 重试之间的延迟（秒）
            on_error: 错误处理策略
                - 'raise': 抛出异常（默认）
                - 'skip': 跳过步骤，继续执行
                - 'continue': 返回 None，继续执行
        """
        self.name = name or self.__class__.__name__
        self.output_key = output_key
        self.enabled = enabled
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.on_error = on_error
        
        if on_error not in ("raise", "skip", "continue"):
            raise ValueError(f"Invalid on_error value: {on_error}")
    
    @abstractmethod
    def execute(self, context: ExecutionContext) -> Any:
        """
        执行步骤逻辑（子类必须实现）
        
        Args:
            context: 执行上下文
            
        Returns:
            步骤执行结果
            
        Raises:
            Exception: 执行失败时抛出异常
        """
        pass
    
    def run(self, context: ExecutionContext) -> Any:
        """
        执行步骤（包含错误处理和重试逻辑）
        
        Args:
            context: 执行上下文
            
        Returns:
            步骤执行结果
        """
        if not self.enabled:
            logger.info(f"Step '{self.name}' is disabled, skipping")
            context.add_history(self.name, "skipped", 0.0)
            return None
        
        logger.info(f"Running step: {self.name}")
        start_time = time.time()
        
        # 重试循环
        last_error = None
        for attempt in range(self.retry_count + 1):
            try:
                # 执行步骤
                result = self.execute(context)
                
                # 保存结果到上下文
                if self.output_key:
                    context.set(self.output_key, result)
                    logger.debug(f"Step '{self.name}' output saved to key: {self.output_key}")
                
                # 记录步骤结果
                context.set_step_result(self.name, result)
                
                # 记录执行历史
                duration = time.time() - start_time
                context.add_history(self.name, "success", duration)
                
                logger.info(f"Step '{self.name}' completed successfully in {duration:.2f}s")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Step '{self.name}' failed (attempt {attempt + 1}/{self.retry_count + 1}): {e}"
                )
                
                # 如果还有重试机会，延迟后重试
                if attempt < self.retry_count:
                    time.sleep(self.retry_delay)
                    continue
                
                # 已用完重试次数，处理错误
                duration = time.time() - start_time

                final_msg = (
                    f"Step '{self.name}' failed after "
                    f"{self.retry_count + 1} attempt(s): {e}"
                )
                print(
                    f"\n\033[1;31m[Step Fatal Error]\033[0m {final_msg}\n"
                    f"\033[33mExiting agent loop.\033[0m\n",
                    file=sys.stderr,
                )
                context.add_history(self.name, "failed", duration, str(e))
                raise StepExecutionError(final_msg) from e
        
        # 理论上不会到达这里
        return None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
