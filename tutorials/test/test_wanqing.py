"""
测试万擎 Provider 功能
用于验证 Wanqing API 集成是否正常工作
"""
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

from local_agent.core.config import get_settings
from local_agent.llm.factory import get_llm_provider


def test_config_loading():
    """测试配置加载"""
    print("=" * 60)
    print("测试 1: 配置加载")
    print("=" * 60)
    
    settings = get_settings()
    print(f"✓ LLM Provider: {settings.llm_provider}")
    print(f"✓ Wanqing Base URL: {settings.wanqing_base_url}")
    print(f"✓ Wanqing Model: {settings.wanqing_default_model}")
    print(f"✓ Wanqing Timeout: {settings.wanqing_timeout}")
    
    # 检查 API Key 是否配置（不完整显示）
    if settings.wanqing_api_key:
        masked_key = settings.wanqing_api_key[:8] + "..." + settings.wanqing_api_key[-4:]
        print(f"✓ Wanqing API Key: {masked_key}")
    else:
        print("⚠ Wanqing API Key: 未配置")
    
    print()


def test_provider_creation():
    """测试 Provider 创建"""
    print("=" * 60)
    print("测试 2: Provider 创建")
    print("=" * 60)
    
    try:
        provider = get_llm_provider(provider="wanqing")
        print(f"✓ 成功创建 WanqingProvider")
        print(f"✓ Provider 类型: {type(provider).__name__}")
        print(f"✓ 配置的模型: {provider.model}")
        print()
        return provider
    except Exception as e:
        print(f"✗ Provider 创建失败: {e}")
        return None


def test_llm_instance():
    """测试 LLM 实例创建"""
    print("=" * 60)
    print("测试 3: LLM 实例创建")
    print("=" * 60)
    
    try:
        provider = get_llm_provider(provider="wanqing")
        llm = provider.get_llm()
        print(f"✓ 成功创建 LLM 实例")
        print(f"✓ LLM 类型: {type(llm).__name__}")
        print()
        return llm
    except Exception as e:
        print(f"✗ LLM 实例创建失败: {e}")
        return None


def test_simple_invoke():
    """测试简单调用"""
    print("=" * 60)
    print("测试 4: 简单调用（非流式）")
    print("=" * 60)
    
    try:
        provider = get_llm_provider(provider="wanqing")
        llm = provider.get_llm()
        
        print("发送测试消息: '你好，请简单介绍一下你自己'")
        response = llm.invoke("你好，请简单介绍一下你自己")
        
        print("✓ 调用成功")
        print(f"✓ 响应内容:\n{response.content}")
        print()
        return True
    except Exception as e:
        print(f"✗ 调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stream_invoke():
    """测试流式调用"""
    print("=" * 60)
    print("测试 5: 流式调用")
    print("=" * 60)
    
    try:
        provider = get_llm_provider(provider="wanqing")
        llm = provider.get_llm()
        
        print("发送测试消息: '请列举太阳系的八大行星'")
        print("✓ 流式响应:\n")
        
        for chunk in llm.stream("请列举太阳系的八大行星"):
            if chunk.content:
                print(chunk.content, end="", flush=True)
        
        print("\n")
        print("✓ 流式调用成功")
        print()
        return True
    except Exception as e:
        print(f"\n✗ 流式调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "万擎 Provider 功能测试" + " " * 15 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # 运行所有测试
    results = []
    
    # 测试 1: 配置加载
    try:
        test_config_loading()
        results.append(("配置加载", True))
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        results.append(("配置加载", False))
    
    # 测试 2: Provider 创建
    provider = test_provider_creation()
    results.append(("Provider 创建", provider is not None))
    
    # 测试 3: LLM 实例创建
    llm = test_llm_instance()
    results.append(("LLM 实例创建", llm is not None))
    
    # 测试 4: 简单调用
    if llm:
        result = test_simple_invoke()
        results.append(("简单调用", result))
    else:
        results.append(("简单调用", False))
    
    # 测试 5: 流式调用
    if llm:
        result = test_stream_invoke()
        results.append(("流式调用", result))
    else:
        results.append(("流式调用", False))
    
    # 打印测试总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:20s} : {status}")
    
    print()
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n✓ 所有测试通过！万擎 Provider 集成成功！")
        return 0
    else:
        print(f"\n⚠ {total - passed} 个测试失败，请检查配置和网络连接")
        return 1


if __name__ == "__main__":
    sys.exit(main())
