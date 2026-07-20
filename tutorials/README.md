# 📚 LocalAgent 示例文件说明

本目录包含 LocalAgent 的示例文件，帮助你快速上手 SubAgent 编排系统。

---

## 📁 文件列表

### 0. `demo/` - SubAgent 使用示例集合

**作用：** 完整的 SubAgent 使用教程，展示各种使用场景和最佳实践

**包含文件：**
- `subagent_basic.py` - 完整的 Python 示例代码（4 个演示场景）
- `subagent_advanced.yaml` - 真实可用的 YAML 配置示例
- `README.md` - 详细的使用说明文档

**演示内容：**
1. **基础工作流** - 文本分析流程（读取→分析→摘要→保存）
2. **YAML 配置加载** - 从配置文件加载和运行工作流
3. **嵌套 SubAgent** - 在主 Agent 中调用子 Agent
4. **ExecutionContext** - 变量管理和模板替换

**快速开始：**
```bash
# 运行 Python 示例（包含 4 个演示）
python tutorials/demo/subagent_basic.py

# 使用 YAML 配置
python -c "
from local_agent.subagent import load_subagent_from_file
agent = load_subagent_from_file('tutorials/demo/subagent_advanced.yaml')
result = agent.run(input_data={'input_file': 'test.txt', 'output_file': 'result.md'})
print('✓ 执行完成！查看 result.md 文件')
"
```

**特点：**
- ✅ 代码注释详细，易于理解
- ✅ 支持交互式运行
- ✅ 演示 4 无需 LLM 可直接运行
- ✅ 包含完整的使用说明和常见问题

**详细文档：** 查看 `demo/README.md` 获取完整说明

---

### 1. `subagent_demo.py` - SubAgent 系统快速入门演示

**作用：** 完整的教学示例，展示 SubAgent 系统的各种用法

**特点：**
- ✅ **无需 Ollama 运行**（纯 API 演示）
- ✅ 包含 5 个独立演示场景
- ✅ 详细的代码注释和输出说明

**演示内容：**
1. **ExecutionContext 演示** - 变量管理和模板替换
2. **SubAgent 创建演示** - 链式添加步骤
3. **步骤类型演示** - LLMStep、ToolStep、SubAgentStep
4. **SubAgent 嵌套调用演示** - 组合多个 SubAgent
5. **配置文件格式演示** - YAML 配置示例

**运行方式：**
```bash
python tutorials/demo/subagent_demo.py
```

**预期输出：**
```
============================================================
LocalAgent SubAgent 系统 - 快速入门演示
============================================================

1. ExecutionContext 演示
2. SubAgent 创建演示
3. 步骤类型演示
4. SubAgent 嵌套调用演示
5. 配置文件格式演示

✓ 所有演示完成!
```

---

### 2. `document_analyzer.yaml` - 文档分析器配置示例

**作用：** 真实可用的 SubAgent 配置文件，展示如何通过 YAML 定义工作流

**工作流程：**
```
读取文件 → LLM 分析 → 生成摘要 → 保存报告
```

**包含步骤：**
1. **读取文件** - 使用 `file_reader` 工具
2. **分析内容** - 使用 LLM 分析文档（主题、关键信息、改进建议）
3. **生成摘要** - 使用 LLM 生成 3-5 句简短摘要
4. **保存结果** - 使用 `file_writer` 工具保存 Markdown 格式报告

**使用方式：**

```python
from local_agent.subagent import load_subagent_from_file

# 从配置文件加载 SubAgent
agent = load_subagent_from_file("tutorials/demo/document_analyzer.yaml")

# 运行 SubAgent
result = agent.run(input_data={
    "file_path": "your_document.txt",
    "output_path": "analysis_report.md"
})

# 查看执行结果
print(f"分析完成！")
print(f"成功步骤: {result['summary']['success']}/{result['summary']['total_steps']}")
print(f"总耗时: {result['summary']['total_time']:.2f}秒")
```

**前置条件：**
- ✅ Ollama 已安装并运行
- ✅ 已拉取 `qwen2.5:7b` 模型（或修改配置中的模型名称）

**测试命令：**
```bash
# 1. 创建测试文档
echo "人工智能正在改变世界。机器学习和深度学习技术广泛应用于各个领域。" > test_doc.txt

# 2. 运行分析
python -c "
from local_agent.subagent import load_subagent_from_file
agent = load_subagent_from_file('tutorials/demo/document_analyzer.yaml')
result = agent.run(input_data={'file_path': 'test_doc.txt', 'output_path': 'result.md'})
print('分析完成！查看 result.md 文件')
"

# 3. 查看结果
cat result.md
```

---

### 3. `test_multi_provider.py` - 多 LLM 提供商功能测试

**作用：** 测试和验证 LocalAgent 的多提供商支持（Ollama 和 OpenAI）

**测试内容：**
- ✅ Ollama 提供商初始化测试
- ✅ OpenAI 提供商初始化测试
- ✅ 配置文件加载测试
- ✅ 自动提供商选择测试
- ✅ 运行时模型切换测试

**运行方式：**
```bash
python tutorials/test/test_multi_provider.py
```

**预期输出：**
```
=== 测试 Ollama 提供商 ===
Provider: ollama
Model: qwen2.5:7b
✓ Ollama 提供商测试通过

=== 测试 OpenAI 提供商 ===
Provider: openai
Model: gpt-4
✓ OpenAI 提供商测试通过

=== 测试配置加载 ===
Default Provider: ollama
✓ 配置加载测试通过

...
✓ 所有测试通过
```

**前置条件：**
- ⚠️ 测试 Ollama: 需要 Ollama 运行
- ⚠️ 测试 OpenAI: 需要设置 `OPENAI_API_KEY` 环境变量

---

### 4. `test_debug_mode.py` - Debug 模式和中文输入测试

**作用：** 测试和验证 LocalAgent 的 debug 模式配置和中文输入功能

**测试内容：**
- ✅ 配置读取测试（debug_print_mode、debug_print_model_input 等）
- ✅ 动态配置修改测试
- ✅ Debug 工具函数测试（should_print_debug、should_print_model_input 等）
- ✅ Rich 格式化输出测试
- ✅ 消息格式化测试（SystemMessage、HumanMessage、AIMessage）
- ✅ 工具调用打印测试
- ✅ 中文输入功能测试

**运行方式：**
```bash
python tutorials/test/test_debug_mode.py
```

**预期输出：**
```
=== 配置测试 ===
debug_print_mode: False
...
=== 测试中文输入 ===
请输入中文文本（测试input()函数）：
>> 你好世界
你输入的是：你好世界
字符长度：4

✅ 所有测试完成！
```

**前置条件：**
- ✅ LocalAgent 已正确安装
- ✅ 配置文件（config.yaml）存在

---

## 🚀 快速开始指南

### 步骤 0: 运行 SubAgent 示例（推荐入口）

```bash
# 运行完整的 SubAgent 使用示例
python tutorials/demo/subagent_basic.py
```

这是最全面的入门示例，包含 4 个演示场景：
- **演示 4** 无需 LLM，可立即运行（展示变量管理）
- **演示 1-3** 需要 Ollama，展示完整工作流

详细说明请查看 `tutorials/demo/README.md`

### 步骤 1: 了解基本 API（无需 Ollama）

```bash
# 运行 API 演示，了解 SubAgent 系统的基本概念
python tutorials/demo/subagent_demo.py
```

这会展示如何创建 SubAgent、添加步骤、使用上下文等核心概念。

### 步骤 2: 启动 Ollama（用于运行实际工作流）

```bash
# macOS/Linux
brew install ollama
ollama serve

# 拉取模型
ollama pull qwen2.5:7b
```

### 步骤 3: 运行真实的工作流

```bash
# 创建测试文档
echo "LocalAgent 是一个强大的 AI Agent 框架。" > test.txt

# 运行文档分析
python -c "
from local_agent.subagent import load_subagent_from_file
agent = load_subagent_from_file('tutorials/demo/document_analyzer.yaml')
result = agent.run(input_data={'file_path': 'test.txt', 'output_path': 'analysis.md'})
"

# 查看分析结果
cat analysis.md
```

### 步骤 4: 创建自己的 SubAgent

参考 `subagent_demo.py` 中的示例，创建自定义工作流：

```python
from local_agent.subagent import SubAgent, LLMStep, ToolStep

# 创建自定义 SubAgent
my_agent = SubAgent("my_workflow")

# 添加步骤
my_agent.add_step(LLMStep(
    model="qwen2.5:7b",
    prompt="你的提示词",
    output_key="result"
)).add_step(ToolStep(
    tool="file_writer",
    params={"path": "output.txt", "content": "{result}"}
))

# 运行
result = my_agent.run(input_data={})
```

---

## 📖 更多学习资源

- **[README.md](../README.md)** - 项目完整文档
- **local_agent/subagent/** - SubAgent 模块源代码
- **配置示例** - `demo/document_analyzer.yaml`
- **测试工具** - `test/test_debug_mode.py`, `test/test_multi_provider.py`

---

## ❓ 常见问题

### Q1: 运行 `document_analyzer.yaml` 时报错 "Tool 'file_reader' not found"

**A:** 确保已正确安装 LocalAgent：
```bash
pip install -e .
```

### Q2: LLM 调用失败 "Connection refused"

**A:** 确保 Ollama 服务正在运行：
```bash
ollama serve
```

### Q3: 如何修改 YAML 配置中的模型？

**A:** 编辑 `document_analyzer.yaml`，将 `model: "qwen2.5:7b"` 改为你拥有的模型名称。

### Q4: 如何使用 OpenAI 而不是 Ollama？

**A:** 在步骤配置中添加 `provider: "openai"`：
```yaml
- type: "llm"
  model: "gpt-4"
  provider: "openai"
  prompt: "你的提示词"
```

并设置环境变量：
```bash
export OPENAI_API_KEY="sk-..."
```

---

## 🎯 下一步

1. ✅ **推荐**: 运行 `tutorials/demo/subagent_basic.py` 学习完整用法
2. ✅ 查看 `tutorials/demo/README.md` 了解详细说明
3. ✅ 查看 `tutorials/demo/subagent_advanced.yaml` 学习配置格式
4. ✅ 运行 `demo/subagent_demo.py` 了解 API（无需 Ollama）
5. ✅ 查看 `demo/document_analyzer.yaml` 学习配置格式
6. ✅ 运行 `test/test_debug_mode.py` 测试 debug 功能
7. ✅ 创建自己的 SubAgent 工作流
8. ✅ 阅读完整文档：[README.md](../README.md)

祝你使用愉快！🚀
