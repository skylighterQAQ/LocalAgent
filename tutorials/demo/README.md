# 🚀 SubAgent 使用示例

欢迎来到 LocalAgent SubAgent 系统的演示教程！本目录包含完整的示例代码和配置文件，帮助你快速掌握 SubAgent 的使用方法。

## 📦 文件列表

### 1. `subagent_basic.py` - 完整的 Python 示例

**内容：** 包含 4 个独立的演示场景，展示 SubAgent 的核心功能

**演示场景：**
- **演示 1: 基础工作流** - 文本分析（读取→分析→摘要→保存）
- **演示 2: YAML 配置加载** - 从配置文件加载和运行工作流
- **演示 3: 嵌套调用** - 在主 SubAgent 中调用子 SubAgent
- **演示 4: ExecutionContext** - 变量管理和模板替换

**特点：**
- ✅ 代码注释详细
- ✅ 支持交互式运行
- ✅ 演示 4 无需 LLM 可直接运行
- ✅ 包含错误处理和用户提示

### 2. `subagent_advanced.yaml` - YAML 配置示例

**内容：** 真实可用的工作流配置文件

**工作流程：**
```
读取文件 → LLM 分析 → 生成摘要 → 提取关键词 → 保存报告
```

**特点：**
- ✅ 完整的配置注释
- ✅ 支持变量替换
- ✅ 错误处理配置
- ✅ 多个 LLM 步骤演示
- ✅ 可直接使用

---

## 🎯 快速开始

### 前置条件

#### 选项 A: 使用 Ollama（本地）

```bash
# 1. 安装 Ollama
brew install ollama  # macOS

# 2. 启动 Ollama 服务
ollama serve

# 3. 拉取模型（在新终端）
ollama pull qwen2.5:7b
```

#### 选项 B: 使用 OpenAI（云端）

```bash
# 设置 API Key
export OPENAI_API_KEY="sk-..."

# 修改配置文件中的 provider
# provider: "openai"
# model: "gpt-4"
```

### 运行示例

#### 方式 1: 运行 Python 示例（推荐）

```bash
# 进入项目根目录
cd LocalAgent

# 运行所有演示
python tutorials/demo/subagent_basic.py
```

**预期输出：**
```
============================================================
LocalAgent SubAgent 系统使用示例
============================================================

本示例展示 SubAgent 系统的核心功能：
  1. 基础工作流 - 文本分析
  2. 从 YAML 配置文件加载
  3. 嵌套 SubAgent 调用
  4. ExecutionContext 变量管理

⚠️  注意:
  - 演示 1-3 需要 Ollama 运行并拉取 qwen2.5:7b 模型
  - 演示 4 无需 LLM，可直接运行

============================================================
演示 4: ExecutionContext 变量管理
============================================================

🔧 创建执行上下文
  初始变量: {'user_name': 'Alice', 'topic': 'AI Agent'}

📝 变量操作:
  设置 project = LocalAgent

🔄 模板替换:
  模板: Hello {user_name}! Welcome to {project}.
  结果: Hello Alice! Welcome to LocalAgent.

...

是否运行需要 LLM 的演示？(y/n): y

============================================================
演示 1: 基础工作流 - 文本分析
============================================================

📝 SubAgent 配置:
SubAgent: text_analyzer
Description: 分析文本内容并生成摘要
Steps (5):
  1. ToolStep(name='prepare_text', tool='file_writer')
  2. ToolStep(name='read_file', tool='file_reader')
  3. LLMStep(name='analyze_text', model='qwen2.5:7b')
  4. LLMStep(name='generate_summary', model='qwen2.5:7b')
  5. ToolStep(name='save_result', tool='file_writer')

✓ 配置有效: True

🚀 开始执行工作流...

✓ 执行完成!

📊 执行摘要:
  - 总步骤: 5
  - 成功: 5
  - 失败: 0
  - 总耗时: 3.45 秒

📄 生成的摘要:
  LocalAgent 是一个功能强大的 AI Agent 开发框架...

💾 完整报告已保存到: /tmp/analysis_result.md

...

✅ 所有演示完成!
```

#### 方式 2: 使用 YAML 配置

```bash
# 创建测试文件
echo "SubAgent 系统支持灵活的工作流编排" > /tmp/test_input.txt

# 运行 YAML 配置
python -c "
from local_agent.subagent import load_subagent_from_file

# 加载配置
agent = load_subagent_from_file('tutorials/demo/subagent_advanced.yaml')

# 执行工作流
result = agent.run(input_data={
    'input_file': '/tmp/test_input.txt',
    'output_file': '/tmp/test_output.md'
})

# 查看结果
print(f'✓ 执行完成！')
print(f'成功步骤: {result[\"summary\"][\"success\"]}/{result[\"summary\"][\"total_steps\"]}')
print(f'总耗时: {result[\"summary\"][\"total_time\"]:.2f} 秒')
print(f'结果已保存到: /tmp/test_output.md')
"

# 查看生成的报告
cat /tmp/test_output.md
```

---

## 📚 详细说明

### SubAgent 是什么？

SubAgent 是 LocalAgent 提供的工作流编排系统，允许你定义和执行**线性工作流**。

**核心概念：**
- **SubAgent**: 工作流容器，包含多个步骤
- **Step**: 工作流步骤，支持三种类型：
  - `LLMStep`: 调用大语言模型
  - `ToolStep`: 调用工具（文件操作、API 调用等）
  - `SubAgentStep`: 调用另一个 SubAgent
- **ExecutionContext**: 执行上下文，管理变量和数据传递

### 核心功能

#### 1. 链式调用

```python
from local_agent.subagent import SubAgent, LLMStep, ToolStep

agent = SubAgent("my_workflow")

# 支持链式添加步骤
agent.add_step(
    ToolStep(tool="file_reader", params={"path": "input.txt"}, output_key="text")
).add_step(
    LLMStep(model="qwen2.5:7b", prompt="分析: {text}", output_key="analysis")
).add_step(
    ToolStep(tool="file_writer", params={"path": "output.txt", "content": "{analysis}"})
)
```

#### 2. 变量系统

```python
from local_agent.subagent import ExecutionContext

# 创建上下文
context = ExecutionContext(initial_data={"name": "Alice"})

# 设置变量
context.set("project", "LocalAgent")

# 模板替换
result = context.render_template("Hello {name}, welcome to {project}!")
# 输出: Hello Alice, welcome to LocalAgent!
```

#### 3. 嵌套调用

```python
from local_agent.subagent import SubAgent, SubAgentStep, LLMStep

# 创建子 SubAgent
summarizer = SubAgent("summarizer")
summarizer.add_step(LLMStep(model="qwen2.5:7b", prompt="总结: {text}"))

# 在主 SubAgent 中调用
main = SubAgent("main")
main.add_step(
    SubAgentStep(
        subagent=summarizer,
        input_mapping={"text": "input_text"},
        output_key="summary"
    )
)
```

#### 4. 配置文件支持

**YAML 配置：**
```yaml
name: "my_workflow"
description: "工作流描述"
steps:
  - type: "tool"
    tool: "file_reader"
    params:
      path: "{input_file}"
    output_key: "content"
    
  - type: "llm"
    model: "qwen2.5:7b"
    prompt: "分析: {content}"
    output_key: "analysis"
```

**加载和运行：**
```python
from local_agent.subagent import load_subagent_from_file

agent = load_subagent_from_file("config.yaml")
result = agent.run(input_data={"input_file": "test.txt"})
```

---

## 💡 使用场景

### 场景 1: 文档分析

```python
# 读取 → 分析 → 总结 → 保存
agent = SubAgent("doc_analyzer")
agent.add_step(ToolStep(tool="file_reader", ...))
agent.add_step(LLMStep(prompt="分析文档...", ...))
agent.add_step(LLMStep(prompt="生成摘要...", ...))
agent.add_step(ToolStep(tool="file_writer", ...))
```

### 场景 2: 数据处理流水线

```python
# 提取 → 转换 → 加载 (ETL)
agent = SubAgent("data_pipeline")
agent.add_step(ToolStep(tool="data_extractor", ...))
agent.add_step(LLMStep(prompt="数据清洗...", ...))
agent.add_step(ToolStep(tool="database_writer", ...))
```

### 场景 3: 自动化报告生成

```python
# 收集数据 → 分析 → 生成图表 → 编写报告
agent = SubAgent("report_generator")
agent.add_step(ToolStep(tool="data_collector", ...))
agent.add_step(LLMStep(prompt="分析趋势...", ...))
agent.add_step(ToolStep(tool="chart_generator", ...))
agent.add_step(LLMStep(prompt="撰写报告...", ...))
```

---

## ❓ 常见问题

### Q1: 如何调试 SubAgent？

**A:** 使用 `debug_print_mode` 配置或查看执行历史：

```python
# 方法 1: 启用 debug 模式（在 config.yaml 中）
debug:
  debug_print_mode: true
  debug_print_model_input: true

# 方法 2: 查看执行历史
result = agent.run(input_data=...)
history = agent.get_execution_history()
for record in history:
    print(record)

# 方法 3: 查看执行摘要
summary = agent.get_execution_summary()
print(f"成功: {summary['success']}, 失败: {summary['failed']}")
```

### Q2: 如何处理步骤失败？

**A:** 使用 `on_error` 参数控制错误处理策略：

```python
# 失败后抛出异常（默认）
step1 = LLMStep(..., on_error="raise")

# 失败后继续执行
step2 = LLMStep(..., on_error="continue")
```

在 YAML 中：
```yaml
- type: "llm"
  prompt: "..."
  on_error: "continue"  # 或 "raise"
```

### Q3: 如何在步骤间传递数据？

**A:** 使用 `output_key` 和变量替换：

```python
# 步骤 1: 保存结果到 "text"
step1 = ToolStep(..., output_key="text")

# 步骤 2: 使用 {text} 引用结果
step2 = LLMStep(prompt="分析: {text}", ...)
```

### Q4: 支持哪些工具？

**A:** SubAgent 支持所有注册的工具，包括：
- **本地工具**: `file_reader`, `file_writer`, `http_client` 等
- **MCP 工具**: 通过 `mcp.json` 配置的外部工具

查看可用工具：
```python
from local_agent.tools import ToolRegistry
registry = ToolRegistry()
print(registry.list_tools())
```

### Q5: 如何切换 LLM 提供商？

**A:** 在步骤配置中指定 `provider` 参数：

```python
# 使用 Ollama
LLMStep(model="qwen2.5:7b", provider="ollama", ...)

# 使用 OpenAI
LLMStep(model="gpt-4", provider="openai", ...)
```

在 YAML 中：
```yaml
- type: "llm"
  model: "gpt-4"
  provider: "openai"
  prompt: "..."
```

### Q6: YAML 配置文件支持哪些格式？

**A:** 支持 YAML 和 JSON 两种格式：

```python
# YAML
agent = load_subagent_from_file("config.yaml")

# JSON
agent = load_subagent_from_file("config.json")
```

### Q7: 如何保存 SubAgent 配置？

**A:** 使用 `save_subagent_to_file` 函数：

```python
from local_agent.subagent import save_subagent_to_file

# 创建 SubAgent
agent = SubAgent("my_workflow")
agent.add_step(...)

# 保存为 YAML
save_subagent_to_file(agent, "my_workflow.yaml")

# 保存为 JSON
save_subagent_to_file(agent, "my_workflow.json")
```

---

## 🎓 进阶学习

### 1. 阅读源码

- **SubAgent 核心**: `local_agent/subagent/subagent.py`
- **执行上下文**: `local_agent/subagent/context.py`
- **步骤类型**: `local_agent/subagent/steps/`
- **配置加载**: `local_agent/subagent/config.py`

### 2. 查看更多示例

- **测试文件**: `tutorials/test/`
- **文档分析器**: `tutorials/demo/document_analyzer.yaml`（如果存在）
- **主 README**: 项目根目录的 `README.md`

### 3. 自定义步骤类型

继承 `BaseStep` 创建自定义步骤：

```python
from local_agent.subagent.steps.base import BaseStep

class CustomStep(BaseStep):
    def run(self, context):
        # 实现自定义逻辑
        result = self.do_something(context)
        
        # 保存结果到上下文
        if self.output_key:
            context.set(self.output_key, result)
        
        return result
```

### 4. 集成 MCP 工具

在 `config/mcp.json` 中配置 MCP 服务器：

```json
{
  "mcpServers": {
    "my_server": {
      "command": "npx",
      "args": ["-y", "@my/mcp-server"]
    }
  }
}
```

在 SubAgent 中使用：

```python
ToolStep(
    tool="mcp_tool_name",
    params={"arg1": "value1"},
    output_key="result"
)
```

---

## 📞 获取帮助

- **项目文档**: 查看主 README.md
- **示例代码**: 运行 `tutorials/demo/subagent_basic.py`
- **测试文件**: 查看 `tutorials/test/` 目录
- **源码**: 浏览 `local_agent/subagent/` 目录

---

## 🚀 下一步

1. ✅ 运行 `python tutorials/demo/subagent_basic.py` 了解基本用法
2. ✅ 查看 `subagent_advanced.yaml` 学习配置格式
3. ✅ 修改 YAML 配置，创建自己的工作流
4. ✅ 探索更多工具和 LLM 提供商
5. ✅ 阅读主 README 了解完整系统架构

祝你使用愉快！🎉
