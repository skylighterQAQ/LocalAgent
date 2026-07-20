#!/usr/bin/env python3
"""
LocalAgent SubAgent 使用示例

本示例展示如何使用 SubAgent 系统创建和执行工作流。
包含以下场景：
1. 基础工作流 - 文本处理
2. 从 YAML 加载配置
3. 嵌套 SubAgent 调用
4. 错误处理和调试

运行方式：
    python tutorials/demo/subagent_basic.py
"""

import os
import sys
import tempfile
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

temp_dir = Path(tempfile.gettempdir())

from local_agent.core.subagent import (
    SubAgent,
    LLMStep,
    ToolStep,
    SubAgentStep,
    ExecutionContext,
    load_subagent_from_file,
)


def demo_basic_workflow():
    """
    演示 1: 基础工作流 - 文本分析
    
    工作流程：
    1. 创建测试文本
    2. 使用 LLM 分析文本
    3. 使用 LLM 生成摘要
    4. 保存结果到文件
    """
    print("\n" + "=" * 60)
    print("演示 1: 基础工作流 - 文本分析")
    print("=" * 60)
    
    # 创建 SubAgent
    agent = SubAgent(
        name="text_analyzer",
        description="分析文本内容并生成摘要"
    )
    
    # 添加步骤（链式调用）
    agent.add_step(
        # 步骤 1: 准备输入数据（使用上下文变量）
        ToolStep(
            name="prepare_text",
            tool="file_writer",
            params={
                "path": str(temp_dir / "test_input.txt"),
                "content": "LocalAgent 是一个强大的 AI Agent 框架，支持 SubAgent 编排、工具调用和 LLM 集成。"
            },
            output_key="input_file"
        )
    ).add_step(
        # 步骤 2: 读取文件
        ToolStep(
            name="read_file",
            tool="file_reader",
            params={"path": str(temp_dir / "test_input.txt")},
            output_key="text_content"
        )
    ).add_step(
        # 步骤 3: LLM 分析
        LLMStep(
            name="analyze_text",
            model="qwen2.5:7b",
            provider="ollama",
            prompt="请分析以下文本的主要内容和特点：\n\n{text_content}\n\n请用简洁的语言回答。",
            output_key="analysis"
        )
    ).add_step(
        # 步骤 4: 生成摘要
        LLMStep(
            name="generate_summary",
            model="qwen2.5:7b",
            prompt="基于以下分析，生成一个 1-2 句的摘要：\n\n{analysis}",
            output_key="summary"
        )
    ).add_step(
        # 步骤 5: 保存结果
        ToolStep(
            name="save_result",
            tool="file_writer",
            params={
                "path": str(temp_dir / "analysis_result.md"),
                "content": "# 文本分析报告\n\n## 详细分析\n{analysis}\n\n## 摘要\n{summary}"
            }
        )
    )
    
    # 验证 SubAgent
    print(f"\n📝 SubAgent 配置:")
    print(agent)
    print(f"\n✓ 配置有效: {agent.validate()}")
    
    # 执行工作流
    print("\n🚀 开始执行工作流...")
    try:
        result = agent.run()
        
        # 显示结果
        print("\n✓ 执行完成!")
        print(f"\n📊 执行摘要:")
        summary = result['summary']
        print(f"  - 总步骤: {summary['total_steps']}")
        print(f"  - 成功: {summary['success']}")
        print(f"  - 失败: {summary['failed']}")
        print(f"  - 总耗时: {summary['total_time']:.2f} 秒")
        
        # 显示部分结果
        print(f"\n📄 生成的摘要:")
        print(f"  {result['variables'].get('summary', 'N/A')}")
        
        print(f"\n💾 完整报告已保存到: {temp_dir / 'analysis_result.md'}")
        
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        print("💡 提示: 确保 Ollama 正在运行并已拉取 qwen2.5:7b 模型")


def demo_yaml_loading():
    """
    演示 2: 从 YAML 配置文件加载 SubAgent
    
    展示如何使用配置文件定义工作流，便于维护和复用。
    """
    print("\n" + "=" * 60)
    print("演示 2: 从 YAML 配置文件加载")
    print("=" * 60)
    
    # YAML 配置文件路径
    yaml_path = project_root / "tutorials" / "demo" / "subagent_advanced.yaml"
    
    if not yaml_path.exists():
        print(f"\n⚠️  配置文件未找到: {yaml_path}")
        print("   跳过此演示。")
        return
    
    print(f"\n📂 加载配置文件: {yaml_path}")
    
    try:
        # 从 YAML 加载 SubAgent
        agent = load_subagent_from_file(str(yaml_path))
        
        print(f"\n✓ 成功加载 SubAgent: {agent.name}")
        print(f"  描述: {agent.description}")
        print(f"  步骤数: {len(agent.steps)}")
        
        # 显示步骤列表
        print(f"\n📋 工作流步骤:")
        for i, step in enumerate(agent.steps, 1):
            print(f"  {i}. {step.name} ({step.__class__.__name__})")
        
        # 准备输入数据
        print(f"\n📝 准备测试数据...")
        test_input = str(temp_dir / "yaml_test_input.txt")
        test_output = str(temp_dir / "yaml_test_output.md")
        
        with open(test_input, 'w', encoding='utf-8') as f:
            f.write("SubAgent 系统支持通过 YAML 配置文件定义工作流，使得工作流的管理和维护更加便捷。")
        
        # 执行工作流
        print(f"\n🚀 执行工作流...")
        result = agent.run(input_data={
            "input_file": test_input,
            "output_file": test_output
        })
        
        print(f"\n✓ 执行完成!")
        print(f"  成功步骤: {result['summary']['success']}/{result['summary']['total_steps']}")
        print(f"  总耗时: {result['summary']['total_time']:.2f} 秒")
        print(f"\n💾 结果已保存到: {test_output}")
        
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        print("💡 提示: 确保 Ollama 正在运行并已拉取所需模型")


def demo_nested_subagent():
    """
    演示 3: 嵌套 SubAgent 调用
    
    展示如何在一个 SubAgent 中调用另一个 SubAgent，
    实现复杂工作流的模块化组合。
    """
    print("\n" + "=" * 60)
    print("演示 3: 嵌套 SubAgent 调用")
    print("=" * 60)
    
    # 创建子 SubAgent：文本摘要器
    summarizer = SubAgent(name="summarizer", description="生成文本摘要")
    summarizer.add_step(
        LLMStep(
            name="create_summary",
            model="qwen2.5:7b",
            prompt="请为以下内容生成简短摘要（1-2句话）：\n\n{text}",
            output_key="summary"
        )
    )
    
    # 创建子 SubAgent：关键词提取器
    keyword_extractor = SubAgent(name="keyword_extractor", description="提取关键词")
    keyword_extractor.add_step(
        LLMStep(
            name="extract_keywords",
            model="qwen2.5:7b",
            prompt="请从以下文本中提取 3-5 个关键词，用逗号分隔：\n\n{text}",
            output_key="keywords"
        )
    )
    
    # 创建主 SubAgent：文档处理器
    main_agent = SubAgent(name="document_processor", description="完整的文档处理流程")
    
    # 添加步骤
    main_agent.add_step(
        # 步骤 1: 准备文档
        ToolStep(
            name="prepare_document",
            tool="file_writer",
            params={
                "path": str(temp_dir / "nested_test.txt"),
                "content": "人工智能技术正在快速发展。SubAgent 系统提供了灵活的编排能力，支持工作流的模块化组合和复用。"
            },
            output_key="doc_path"
        )
    ).add_step(
        # 步骤 2: 读取文档
        ToolStep(
            name="read_document",
            tool="file_reader",
            params={"path": str(temp_dir / "nested_test.txt")},
            output_key="document_text"
        )
    ).add_step(
        # 步骤 3: 调用摘要器 SubAgent
        SubAgentStep(
            name="summarize_document",
            subagent=summarizer,
            input_mapping={"text": "document_text"},  # 映射输入变量
            output_key="doc_summary"
        )
    ).add_step(
        # 步骤 4: 调用关键词提取器 SubAgent
        SubAgentStep(
            name="extract_document_keywords",
            subagent=keyword_extractor,
            input_mapping={"text": "document_text"},
            output_key="doc_keywords"
        )
    ).add_step(
        # 步骤 5: 保存结果
        ToolStep(
            name="save_analysis",
            tool="file_writer",
            params={
                "path": str(temp_dir / "nested_result.md"),
                "content": "# 文档分析结果\n\n## 摘要\n{doc_summary}\n\n## 关键词\n{doc_keywords}"
            }
        )
    )
    
    # 显示 SubAgent 结构
    print(f"\n🏗️  SubAgent 结构:")
    print(f"\n主 Agent: {main_agent.name}")
    print(f"  - {len(main_agent.steps)} 个步骤")
    print(f"\n子 Agent 1: {summarizer.name}")
    print(f"  - {len(summarizer.steps)} 个步骤")
    print(f"\n子 Agent 2: {keyword_extractor.name}")
    print(f"  - {len(keyword_extractor.steps)} 个步骤")
    
    # 执行主 SubAgent
    print(f"\n🚀 执行嵌套工作流...")
    try:
        result = main_agent.run()
        
        print(f"\n✓ 执行完成!")
        print(f"  成功步骤: {result['summary']['success']}/{result['summary']['total_steps']}")
        print(f"  总耗时: {result['summary']['total_time']:.2f} 秒")
        
        # 显示结果
        print(f"\n📊 分析结果:")
        print(f"  摘要: {result['variables'].get('doc_summary', 'N/A')}")
        print(f"  关键词: {result['variables'].get('doc_keywords', 'N/A')}")
        print(f"\n💾 完整结果已保存到: {temp_dir / 'nested_result.md'}")
        
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        print("💡 提示: 确保 Ollama 正在运行并已拉取 qwen2.5:7b 模型")


def demo_execution_context():
    """
    演示 4: ExecutionContext 使用
    
    展示如何使用执行上下文管理变量、模板替换和执行历史。
    """
    print("\n" + "=" * 60)
    print("演示 4: ExecutionContext 变量管理")
    print("=" * 60)
    
    # 创建执行上下文
    context = ExecutionContext(initial_data={
        "user_name": "Alice",
        "topic": "AI Agent"
    })
    
    print(f"\n🔧 创建执行上下文")
    print(f"  初始变量: {context.get_all()}")
    
    # 设置和获取变量
    print(f"\n📝 变量操作:")
    context.set("project", "LocalAgent")
    print(f"  设置 project = {context.get('project')}")
    
    # 模板替换
    print(f"\n🔄 模板替换:")
    template = "Hello {user_name}! Welcome to {project}."
    rendered = context.render_template(template)
    print(f"  模板: {template}")
    print(f"  结果: {rendered}")
    
    # 嵌套模板替换
    context.set("greeting", "Hello {user_name}")
    nested_template = "{greeting}, let's discuss {topic}."
    rendered_nested = context.render_template(nested_template)
    print(f"\n  嵌套模板: {nested_template}")
    print(f"  结果: {rendered_nested}")
    
    # 执行历史
    context.add_history({
        "step": "demo_step",
        "status": "success",
        "time": 0.5
    })
    
    print(f"\n📚 执行历史:")
    history = context.get_history()
    for i, record in enumerate(history, 1):
        print(f"  {i}. {record}")
    
    # 执行摘要
    print(f"\n📊 执行摘要:")
    summary = context.get_execution_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main():
    """主函数：运行所有演示"""
    print("=" * 60)
    print("LocalAgent SubAgent 系统使用示例")
    print("=" * 60)
    print("\n本示例展示 SubAgent 系统的核心功能：")
    print("  1. 基础工作流 - 文本分析")
    print("  2. 从 YAML 配置文件加载")
    print("  3. 嵌套 SubAgent 调用")
    print("  4. ExecutionContext 变量管理")
    print("\n⚠️  注意:")
    print("  - 演示 1-3 需要 Ollama 运行并拉取 qwen2.5:7b 模型")
    print("  - 演示 4 无需 LLM，可直接运行")
    
    # 运行演示 4（无需 LLM）
    demo_execution_context()
    
    # 询问是否运行需要 LLM 的演示
    print("\n" + "=" * 60)
    user_input = input("\n是否运行需要 LLM 的演示？(y/n): ").strip().lower()
    
    if user_input == 'y':
        try:
            demo_basic_workflow()
            demo_yaml_loading()
            demo_nested_subagent()
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断执行")
    else:
        print("\n⏭️  跳过 LLM 演示")
    
    print("\n" + "=" * 60)
    print("✅ 所有演示完成!")
    print("=" * 60)
    print("\n📖 更多信息:")
    print("  - 查看 tutorials/demo/README.md 了解详细说明")
    print("  - 查看 tutorials/demo/subagent_advanced.yaml 了解配置格式")
    print("  - 查看 README.md 了解完整文档")
    print()


if __name__ == "__main__":
    main()
