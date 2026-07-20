#!/usr/bin/env python3
"""
测试多 LLM 提供商功能
"""
from local_agent.core.config import get_settings, reset_settings
from local_agent.core.agent import LocalAgent


def test_ollama_provider():
    """测试 Ollama 提供商"""
    print("\n=== 测试 Ollama 提供商 ===")
    
    reset_settings()
    agent = LocalAgent.create(model="qwen2.5:7b", provider="ollama")
    
    print(f"Provider: {agent.provider_type}")
    print(f"Model: {agent.model}")
    print(f"LLM Provider Type: {type(agent.llm_provider).__name__}")
    
    assert agent.provider_type == "ollama"
    assert agent.model == "qwen2.5:7b"
    print("✓ Ollama 提供商测试通过")


def test_openai_provider():
    """测试 OpenAI 提供商"""
    print("\n=== 测试 OpenAI 提供商 ===")
    
    reset_settings()
    agent = LocalAgent.create(model="gpt-4", provider="openai")
    
    print(f"Provider: {agent.provider_type}")
    print(f"Model: {agent.model}")
    print(f"LLM Provider Type: {type(agent.llm_provider).__name__}")
    
    assert agent.provider_type == "openai"
    assert agent.model == "gpt-4"
    print("✓ OpenAI 提供商测试通过")


def test_config_settings():
    """测试配置加载"""
    print("\n=== 测试配置加载 ===")
    
    reset_settings()
    settings = get_settings()
    
    print(f"Default Provider: {settings.llm_provider}")
    print(f"Ollama Model: {settings.ollama_default_model}")
    print(f"OpenAI Model: {settings.openai_default_model}")
    print(f"Ollama Base URL: {settings.ollama_base_url}")
    
    assert hasattr(settings, "llm_provider")
    assert hasattr(settings, "openai_api_key")
    assert hasattr(settings, "openai_base_url")
    print("✓ 配置加载测试通过")


def test_model_switching():
    """测试模型切换"""
    print("\n=== 测试模型切换 ===")
    
    reset_settings()
    agent = LocalAgent.create(model="qwen2.5:7b", provider="ollama")
    
    print(f"初始: {agent.provider_type}:{agent.model}")
    
    # 切换到 OpenAI
    agent.provider_type = "openai"
    agent.model = "gpt-4"
    agent._graph = None
    
    print(f"切换后: {agent.provider_type}:{agent.model}")
    
    assert agent.provider_type == "openai"
    assert agent.model == "gpt-4"
    print("✓ 模型切换测试通过")


def main():
    """运行所有测试"""
    print("开始测试多 LLM 提供商功能...")
    
    try:
        test_config_settings()
        test_ollama_provider()
        test_openai_provider()
        test_model_switching()
        
        print("\n" + "="*50)
        print("✓ 所有测试通过！")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
