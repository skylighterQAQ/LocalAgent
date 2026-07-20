"""
Configuration loader for SubAgent
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from local_agent.core.subagent.subagent import SubAgent
from local_agent.core.subagent.steps.llm_step import LLMStep
from local_agent.core.subagent.steps.tool_step import ToolStep
from local_agent.core.subagent.steps.subagent_step import SubAgentStep
from local_agent.core.subagent.steps.agent_step import AgentStep

logger = logging.getLogger(__name__)


def load_config_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    从文件加载配置
    
    支持 YAML 和 JSON 格式
    
    Args:
        file_path: 配置文件路径
        
    Returns:
        配置字典
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 文件格式不支持
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    if suffix in (".yaml", ".yml"):
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    elif suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")


def create_step_from_config(step_config: Dict[str, Any]) -> Any:
    """
    从配置字典创建步骤实例
    
    Args:
        step_config: 步骤配置字典
        
    Returns:
        步骤实例
        
    Raises:
        ValueError: 配置无效
    """
    step_type = step_config.get("type")
    if not step_type:
        raise ValueError("Step config missing 'type' field")
    
    # 通用参数
    common_params = {
        "name": step_config.get("name"),
        "output_key": step_config.get("output_key"),
        "enabled": step_config.get("enabled", True),
        "retry_count": step_config.get("retry_count", 0),
        "retry_delay": step_config.get("retry_delay", 1.0),
        "on_error": step_config.get("on_error", "raise"),
    }
    
    if step_type == "llm":
        return LLMStep(
            model=step_config.get("model", "qwen2.5:7b"),
            prompt=step_config.get("prompt", ""),
            temperature=step_config.get("temperature", 0.1),
            max_tokens=step_config.get("max_tokens"),
            system_prompt=step_config.get("system_prompt"),
            provider=step_config.get("provider"),
            **common_params,
        )
    
    elif step_type == "tool":
        return ToolStep(
            tool=step_config["tool"],
            params=step_config.get("params", {}),
            mcp_server=step_config.get("mcp_server"),
            auto_load_mcp=step_config.get("auto_load_mcp", True),
            **common_params,
        )
    
    elif step_type == "subagent":
        subagent_source = step_config.get("subagent")
        if not subagent_source:
            raise ValueError("SubAgent step config missing 'subagent' field")
        
        # 如果是字符串，假设是配置文件路径
        # 否则假设是 SubAgent 配置字典
        if isinstance(subagent_source, str):
            subagent_instance = subagent_source
        elif isinstance(subagent_source, dict):
            subagent_instance = create_subagent_from_config(subagent_source)
        else:
            raise ValueError(f"Invalid subagent source type: {type(subagent_source)}")
        
        return SubAgentStep(
            subagent=subagent_instance,
            input_mapping=step_config.get("input_mapping", {}),
            **common_params,
        )
    
    elif step_type == "agent":
        tools = step_config.get("tools", [])
        if isinstance(tools, str):
            # 支持逗号分隔的字符串格式
            tools = [t.strip() for t in tools.split(",") if t.strip()]
        
        return AgentStep(
            prompt=step_config.get("prompt", ""),
            tools=tools,
            system_prompt=step_config.get("system_prompt"),
            model=step_config.get("model"),
            provider=step_config.get("provider"),
            max_iterations=step_config.get("max_iterations", 20),
            tool_call_retry=step_config.get("tool_call_retry", True),
            max_tool_retry=step_config.get("max_tool_retry", 2),
            **common_params,
        )
    
    else:
        raise ValueError(f"Unknown step type: {step_type}")


def create_subagent_from_config(config: Dict[str, Any]) -> SubAgent:
    """
    从配置字典创建 SubAgent 实例
    
    配置格式::
    
        {
            "name": "document_analyzer",
            "description": "分析文档内容",
            "steps": [
                {
                    "type": "tool",
                    "name": "read_file",
                    "tool": "file_reader",
                    "params": {"path": "{input.file_path}"},
                    "output_key": "content"
                },
                {
                    "type": "llm",
                    "name": "analyze",
                    "model": "qwen2.5:7b",
                    "prompt": "分析: {content}",
                    "output_key": "analysis"
                }
            ]
        }
    
    Args:
        config: SubAgent 配置字典
        
    Returns:
        SubAgent 实例
        
    Raises:
        ValueError: 配置无效
    """
    name = config.get("name", "SubAgent")
    description = config.get("description", "")
    
    agent = SubAgent(name=name, description=description)
    
    steps_config = config.get("steps", [])
    if not steps_config:
        logger.warning(f"SubAgent '{name}' has no steps defined")
    
    for i, step_config in enumerate(steps_config, 1):
        try:
            step = create_step_from_config(step_config)
            agent.add_step(step)
        except Exception as e:
            logger.error(f"Failed to create step {i} in SubAgent '{name}': {e}")
            raise ValueError(f"Failed to create step {i}: {e}")
    
    logger.info(f"Created SubAgent '{name}' from config with {len(steps_config)} steps")
    return agent


def load_subagent_from_file(file_path: Union[str, Path]) -> SubAgent:
    """
    从配置文件加载 SubAgent
    
    Args:
        file_path: 配置文件路径（YAML 或 JSON）
        
    Returns:
        SubAgent 实例
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 配置无效
    """
    logger.info(f"Loading SubAgent from file: {file_path}")
    
    config = load_config_file(file_path)
    agent = create_subagent_from_config(config)
    
    return agent


def save_subagent_to_file(
    agent: SubAgent,
    file_path: Union[str, Path],
    format: str = "yaml",
) -> None:
    """
    将 SubAgent 保存为配置文件
    
    注意：只保存基本配置，不保存运行时状态
    
    Args:
        agent: SubAgent 实例
        file_path: 保存路径
        format: 文件格式（"yaml" 或 "json"）
        
    Raises:
        ValueError: 格式不支持
    """
    # 构建配置字典
    config = {
        "name": agent.name,
        "description": agent.description,
        "steps": [],
    }
    
    for step in agent.steps:
        step_config = {
            "name": step.name,
            "output_key": step.output_key,
            "enabled": step.enabled,
            "retry_count": step.retry_count,
            "retry_delay": step.retry_delay,
            "on_error": step.on_error,
        }
        
        if isinstance(step, LLMStep):
            step_config.update({
                "type": "llm",
                "model": step.model,
                "prompt": step.prompt_template,
                "temperature": step.temperature,
                "max_tokens": step.max_tokens,
                "system_prompt": step.system_prompt,
            })
        
        elif isinstance(step, ToolStep):
            step_config.update({
                "type": "tool",
                "tool": step.tool_name,
                "params": step.params,
                "mcp_server": step.mcp_server,
                "auto_load_mcp": step.auto_load_mcp,
            })
        
        elif isinstance(step, SubAgentStep):
            # SubAgentStep 比较复杂，暂时跳过序列化
            step_config.update({
                "type": "subagent",
                "subagent": "# SubAgent instance (not serializable)",
                "input_mapping": step.input_mapping,
            })
        
        elif isinstance(step, AgentStep):
            step_config.update({
                "type": "agent",
                "prompt": step.prompt_template,
                "tools": step.tool_names,
                "system_prompt": step.system_prompt,
                "model": step.model,
                "provider": step.provider,
                "max_iterations": step.max_iterations,
                "tool_call_retry": step.tool_call_retry,
                "max_tool_retry": step.max_tool_retry,
            })
        
        config["steps"].append(step_config)
    
    # 保存文件
    file_path = Path(file_path)
    
    if format == "yaml":
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    elif format == "json":
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved SubAgent '{agent.name}' to file: {file_path}")
