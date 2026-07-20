#!/usr/bin/env python3
"""
测试 LocalAgent 流式输出修复
验证工具调用不会导致提前退出
"""
from main import create_agent

def test_simple_chat():
    """测试简单对话(无工具调用)"""
    print("=" * 60)
    print("测试 1: 简单对话(无工具调用)")
    print("=" * 60)
    
    agent = create_agent(load_mcp=False)
    print("\n用户: 你好,介绍一下自己\n")
    print("助手: ", end="", flush=True)
    
    for token in agent.stream("你好,介绍一下自己"):
        print(token, end="", flush=True)
    print("\n")


def test_tool_call():
    """测试工具调用(文件系统)"""
    print("=" * 60)
    print("测试 2: 工具调用(列出当前目录)")
    print("=" * 60)
    
    agent = create_agent(load_mcp=False)
    print("\n用户: 列出当前目录下的文件\n")
    print("助手: ", end="", flush=True)
    
    for token in agent.stream("列出当前目录下的文件"):
        if token.startswith("\n[Tool"):
            print(f"\n  {token.strip()}", flush=True)
        else:
            print(token, end="", flush=True)
    print("\n")


def test_mcp_tool():
    """测试 MCP 工具调用(高德地图搜索)"""
    print("=" * 60)
    print("测试 3: MCP 工具调用(高德地图搜索)")
    print("=" * 60)
    
    agent = create_agent(load_mcp=True)
    print("\n用户: 帮我找一下北京好吃的自助\n")
    print("助手: ", end="", flush=True)
    
    token_count = 0
    for token in agent.stream("帮我找一下北京好吃的自助"):
        token_count += 1
        if token.startswith("\n[Tool"):
            print(f"\n  {token.strip()}", flush=True)
        else:
            print(token, end="", flush=True)
    
    print(f"\n\n总共收到 {token_count} 个 token")
    print()


if __name__ == "__main__":
    import sys
    
    # 允许选择测试
    tests = {
        "1": ("简单对话", test_simple_chat),
        "2": ("工具调用", test_tool_call),
        "3": ("MCP工具", test_mcp_tool),
    }
    
    if len(sys.argv) > 1:
        test_id = sys.argv[1]
        if test_id in tests:
            name, func = tests[test_id]
            print(f"\n运行测试: {name}\n")
            func()
        else:
            print(f"未知测试: {test_id}")
            print("可用测试: 1=简单对话, 2=工具调用, 3=MCP工具")
    else:
        # 运行所有测试
        print("\n运行所有测试...\n")
        for name, func in tests.values():
            try:
                func()
            except Exception as e:
                print(f"\n❌ 测试失败: {e}\n")
