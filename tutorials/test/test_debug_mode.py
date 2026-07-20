#!/usr/bin/env python3
"""
测试脚本：验证debug模式和中文输入功能
"""
import sys
from pathlib import Path

# Make the test runnable from any checkout location.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from local_agent.core.config import get_settings

# 测试配置读取
settings = get_settings()

print("=== 配置测试 ===")
print(f"debug_print_mode: {settings.debug_print_mode}")
print(f"debug_print_model_input: {settings.debug_print_model_input}")
print(f"debug_print_model_output: {settings.debug_print_model_output}")
print(f"debug_print_tool_calls: {settings.debug_print_tool_calls}")

# 测试动态修改配置
print("\n=== 动态修改配置 ===")
settings.debug_print_mode = True
print(f"启用debug后: {settings.debug_print_mode}")

# 测试debug工具函数
print("\n=== 测试debug工具函数 ===")
from local_agent.core.debug import (
    should_print_debug,
    should_print_model_input,
    should_print_model_output,
    should_print_tool_calls,
    print_debug_separator,
    print_iteration_info,
)

print(f"should_print_debug: {should_print_debug()}")
print(f"should_print_model_input: {should_print_model_input()}")
print(f"should_print_model_output: {should_print_model_output()}")
print(f"should_print_tool_calls: {should_print_tool_calls()}")

print("\n=== 测试Rich格式化输出 ===")
print_debug_separator()
print_iteration_info(1, 50)

# 测试消息格式化
from local_agent.core.messages import HumanMessage, AIMessage, SystemMessage
from local_agent.core.debug import (
    print_model_input,
    print_model_output,
    print_tool_call,
    format_message_for_debug,
)

print("\n=== 测试消息格式化 ===")
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="你好，请介绍一下自己"),
]

print_model_input(messages, system_prompt="测试系统提示词")

ai_msg = AIMessage(content="你好！我是LocalAgent，一个基于Ollama的本地AI助手。")
print_model_output(ai_msg)

print("\n=== 测试工具调用打印 ===")
print_tool_call(
    tool_name="read_file",
    tool_input={"path": "README.md", "encoding": "utf-8"},
    tool_output="# LocalAgent\n\n一个基于Ollama的本地AI Agent框架..."
)

print("\n=== 测试中文输入 ===")
print("请输入中文文本（测试input()函数）：")
user_input = input(">> ")
print(f"你输入的是：{user_input}")
print(f"字符长度：{len(user_input)}")

print("\n✅ 所有测试完成！")
