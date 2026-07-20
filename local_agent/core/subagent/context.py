"""
Execution context - manages data flow between steps in SubAgent
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExecutionContext:
    """
    SubAgent 执行上下文，负责在步骤之间传递数据和状态。
    
    主要功能：
    - 变量存取：通过 set/get 方法管理变量
    - 模板替换：自动替换 {variable} 格式的变量引用
    - 结果缓存：记录每个步骤的执行结果
    - 执行历史：追踪步骤执行顺序和状态
    
    示例::
    
        ctx = ExecutionContext()
        ctx.set("user_name", "Alice")
        ctx.set("greeting", "Hello {user_name}")
        
        # 自动替换变量
        result = ctx.resolve("{greeting}")  # "Hello Alice"
    """
    
    def __init__(self, initial_data: Optional[Dict[str, Any]] = None):
        """
        初始化上下文
        
        Args:
            initial_data: 初始变量字典
        """
        self._variables: Dict[str, Any] = initial_data or {}
        self._step_results: Dict[str, Any] = {}
        self._execution_history: List[Dict[str, Any]] = []
        
    def set(self, key: str, value: Any) -> None:
        """
        设置变量值
        
        Args:
            key: 变量名
            value: 变量值
        """
        self._variables[key] = value
        logger.debug(f"Context set: {key} = {value}")
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取变量值
        
        Args:
            key: 变量名
            default: 默认值（如果变量不存在）
            
        Returns:
            变量值或默认值
        """
        return self._variables.get(key, default)
    
    def has(self, key: str) -> bool:
        """
        检查变量是否存在
        
        Args:
            key: 变量名
            
        Returns:
            是否存在
        """
        return key in self._variables
    
    def update(self, data: Dict[str, Any]) -> None:
        """
        批量更新变量
        
        Args:
            data: 变量字典
        """
        self._variables.update(data)
        logger.debug(f"Context updated with {len(data)} variables")
    
    def get_all(self) -> Dict[str, Any]:
        """
        获取所有变量
        
        Returns:
            变量字典的副本
        """
        return self._variables.copy()
    
    def resolve(self, value: Any) -> Any:
        """
        递归解析值中的变量引用
        
        支持格式：
        - 字符串: "{variable}" -> 替换为变量值
        - 字典: {"key": "{variable}"} -> 递归替换
        - 列表: ["{variable}"] -> 递归替换
        - 其他类型: 直接返回
        
        Args:
            value: 要解析的值
            
        Returns:
            解析后的值
        """
        if isinstance(value, str):
            return self._resolve_string(value)
        elif isinstance(value, dict):
            return {k: self.resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [self.resolve(item) for item in value]
        else:
            return value
    
    def _resolve_string(self, template: str) -> Any:
        """
        解析字符串模板中的变量引用
        
        Args:
            template: 模板字符串
            
        Returns:
            解析后的值（可能是字符串或其他类型）
        """
        # 如果整个字符串就是一个变量引用，直接返回变量值（保留类型）
        if re.match(r'^\{[^}]+\}$', template):
            var_name = template[1:-1]
            return self._variables.get(var_name, template)
        
        # 否则进行字符串替换
        def replace_var(match):
            var_name = match.group(1)
            value = self._variables.get(var_name)
            return str(value) if value is not None else match.group(0)
        
        return re.sub(r'\{([^}]+)\}', replace_var, template)
    
    def set_step_result(self, step_name: str, result: Any) -> None:
        """
        记录步骤执行结果
        
        Args:
            step_name: 步骤名称
            result: 执行结果
        """
        self._step_results[step_name] = result
        logger.debug(f"Step result saved: {step_name}")
    
    def get_step_result(self, step_name: str) -> Optional[Any]:
        """
        获取步骤执行结果
        
        Args:
            step_name: 步骤名称
            
        Returns:
            步骤结果或 None
        """
        return self._step_results.get(step_name)
    
    def add_history(self, step_name: str, status: str, duration: float, error: Optional[str] = None):
        """
        添加执行历史记录
        
        Args:
            step_name: 步骤名称
            status: 执行状态 (success/failed/skipped)
            duration: 执行耗时（秒）
            error: 错误信息（如果失败）
        """
        self._execution_history.append({
            "step": step_name,
            "status": status,
            "duration": duration,
            "error": error,
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        获取执行历史
        
        Returns:
            历史记录列表
        """
        return self._execution_history.copy()
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        获取执行摘要
        
        Returns:
            包含统计信息的字典
        """
        total = len(self._execution_history)
        success = sum(1 for h in self._execution_history if h["status"] == "success")
        failed = sum(1 for h in self._execution_history if h["status"] == "failed")
        skipped = sum(1 for h in self._execution_history if h["status"] == "skipped")
        total_time = sum(h["duration"] for h in self._execution_history)
        
        return {
            "total_steps": total,
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "total_time": total_time,
        }
    
    def __repr__(self) -> str:
        return f"ExecutionContext(variables={len(self._variables)}, steps={len(self._step_results)})"
