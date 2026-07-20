# 🤖 LocalAgent

> 一个基于 **Ollama/OpenAI/万擎/Claude Code** 的本地/云端 AI Agent 框架，支持丰富的工具、技能和工作流编排系统。核心运行时引擎完全自主实现，不依赖 LangChain / LangGraph。

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![Ollama](https://img.shields.io/badge/Ollama-local-orange)](https://ollama.ai)
[![OpenAI](https://img.shields.io/badge/OpenAI-compatible-blue)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🔄 **多模式支持** | 支持 Ollama（本地）、OpenAI（云端）、万擎（快手内部）、Claude Code 四种 LLM 提供商 |
| 🏠 **完全本地化** | 基于 Ollama，无需 API Key，保护隐私 |
| ☁️ **云端增强** | 支持 OpenAI API 及兼容接口（Azure OpenAI、万擎等）|
| 🔧 **丰富工具** | 40+ 开箱即用的工具 |
| 🎯 **技能系统** | 8 个内置专业技能 |
| 🔗 **SubAgent 编排** | 用代码或配置文件定义线性工作流（LLM/Tool/SubAgent） ⭐ |
| 🔌 **MCP 支持** | 兼容 Model Context Protocol，ToolStep 无缝集成 MCP 工具 |
| 🌊 **流式响应** | 实时 Token 流式输出 |
| 💬 **多种界面** | CLI、Python API、Web UI |
| 🧠 **长期记忆** | 基于 ChromaDB 的向量记忆 |
| 🏗️ **共享架构** | 统一的 Provider 层，避免重复实例化 |
| ⚙️ **自主运行时** | ReAct 引擎、消息类型、LLM 客户端全部自主实现，零第三方 Agent 框架依赖 |

---

## 🚀 快速开始

### 1. 选择 LLM 提供商

#### 选项 A：使用 Ollama（本地）

```bash
brew install ollama  # macOS
ollama serve         # 启动服务
ollama pull qwen2.5:7b  # 拉取模型
```

#### 选项 B：使用 OpenAI（云端）

创建 `.env` 文件或设置环境变量：
```bash
export OPENAI_API_KEY="sk-..."
# 可选：用于兼容 API
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

#### 选项 C：使用万擎（快手内部）

万擎是快手内部部署的大模型服务，支持 OpenAI 兼容接口。

创建 `.env` 文件或设置环境变量：
```bash
export WANQING_API_KEY="your-api-key"
export WANQING_BASE_URL="https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints"
export WANQING_DEFAULT_MODEL="ep-116jmc-1778070434284074729"
```

或在 `config.yaml` 中配置：
```yaml
provider: "wanqing"

wanqing:
  api_key: "your-api-key"
  base_url: "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints"
  default_model: "ep-116jmc-1778070434284074729"
  timeout: 120
```

### 2. 安装 LocalAgent

```bash
git clone https://github.com/your-name/LocalAgent.git
cd LocalAgent
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### 安装 Playwright 浏览器（网页搜索与浏览工具必需）

`pip install -e .` 只会安装 Playwright 的 Python 库；浏览器程序需要在每台机器、每个操作系统上单独下载。完成依赖安装后，执行：

```bash
python -m playwright install chromium
```

Windows 用户请在**运行 LocalAgent 的同一个 Python/Conda 环境**中执行该命令。例如：

```powershell
E:\conda\python.exe -m playwright install chromium
```

若未执行此步骤，使用 `search_web`、浏览器访问等功能时可能会报 `BrowserType.launch: Executable doesn't exist`。不要从 macOS 复制浏览器缓存到 Windows；Playwright 浏览器二进制文件与操作系统有关，需分别安装。

### 3. 配置 LLM 提供商

编辑 `config.yaml`：

```yaml
llm:
  provider: "ollama"  # 或 "openai"、"wanqing"

ollama:
  base_url: "http://localhost:11434"
  default_model: "qwen2.5:7b"
  timeout: 120

openai:
  api_key: ""  # 通过环境变量 OPENAI_API_KEY
  base_url: ""  # 可选，用于兼容 API
  default_model: "gpt-3.5-turbo"

wanqing:
  api_key: ""  # 通过环境变量 WANQING_API_KEY
  base_url: "https://wanqing-api.corp.kuaishou.com/api/gateway/v1/endpoints"
  default_model: "ep-116jmc-1778070434284074729"
  timeout: 120
```

### 4. 开始使用

```bash
# 交互式对话（使用配置的默认提供商）
python main.py

# 使用 Ollama
python main.py -m qwen2.5:7b

# 使用 OpenAI
python main.py -m gpt-4

# 单次问答
python main.py --mode once -q "列出当前目录的文件"

# 启动 Web 界面
python main.py --mode server
```

### 5. 在终端查看和切换模型

#### 查看所有可用模型

使用 `/models` 命令列出所有 Provider 的可用模型：

```bash
/models  # 显示 Ollama、OpenAI、万擎的所有模型及连接状态
```

输出示例：
```
Available Models (Current: wanqing / ep-116jmc-1778070434284074729)
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Provider ┃ Model                        ┃ Status    ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Ollama   │ qwen2.5:7b                   │           │
│ OpenAI   │ gpt-4                        │           │
│          │ gpt-4-turbo                  │           │
│ Wanqing  │ ep-116jmc-1778070434284074729│ ◀ active  │
└──────────┴──────────────────────────────┴───────────┘
```

#### 切换模型

在交互式聊天中，使用 `/model` 命令：

```bash
# 查看当前模型
/model

# 切换模型（同一提供商）
/model qwen2.5:14b

# 切换提供商和模型
/model openai:gpt-4
/model ollama:llama3.1:8b
/model wanqing:ep-116jmc-1778070434284074729
```

---

## 📦 配置系统

LocalAgent 以 `config.yaml` 作为唯一配置中心，所有模块（LLM、技能、MCP、Agent、Memory 等）均在此文件中集中管理。

### 配置结构

```
LocalAgent/
├── config.yaml          # 主配置文件（单一真相来源，所有设置都在此）
├── .env                 # API Keys 等敏感信息（不提交到版本控制）
└── config/              # 向后兼容目录（可选）
    └── mcp.json         # 历史 MCP 配置（仍被加载，优先级低于 config.yaml）
```

> **推荐做法**：只维护 `config.yaml` + `.env`，无需其他配置文件。

### 配置优先级

1. **环境变量**（最高优先级）
2. **.env 文件**
3. **config.yaml**（主配置，单一真相来源）
4. **代码默认值**（最低优先级）

### 向后兼容性

✅ **完全向后兼容**，旧的配置文件仍然有效：
- `config/mcp.json` 若存在，仍会被加载，与 `config.yaml` 中的 `mcp.servers` 合并（同名服务器以 `config.yaml` 为准）
- `config/skills.yaml` 中的内容已与 `config.yaml` 中的 `skills:` 配置段等效，不再需要单独维护
- 可以通过环境变量 `MCP_CONFIG_PATH` 指定自定义 MCP JSON 文件路径

### 完整配置示例

`config.yaml` 涵盖以下所有模块：

```yaml
# LLM 提供商
llm:
  provider: "ollama"

ollama:
  base_url: "http://localhost:11434"
  default_model: "qwen3:8b"
  models: ["qwen3:8b", "qwen2.5:7b"]
  timeout: 120

# Agent 行为
agent:
  max_iterations: 50
  verbose: true
  stream: true

# 技能管理
skills:
  auto_load: true
  directories:
    - "./local_agent/skills/builtin"
    - "./skills"

# MCP 服务器（内联定义，无需 mcp.json）
mcp:
  enabled: true
  servers:
    filesystem:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
      description: "本地文件系统访问"
    weather:
      url: "https://your-mcp-server/sse"
      description: "天气信息服务"

# 记忆系统
memory:
  enable_long_term: true
  chroma_path: "./.local_agent/memory"

# 服务器
server:
  host: "0.0.0.0"
  port: 8080

# 调试
debug:
  print_mode: false
```

---

## 🔗 SubAgent 编排系统 ⭐

LocalAgent 提供强大的工作流编排能力，可以用 Python 代码或配置文件定义线性执行流程。

> 💡 **完整教程**：查看 [tutorials/demo/](tutorials/demo/) 目录获取详细示例和使用说明

### 快速示例

```python
from local_agent.subagent import SubAgent, LLMStep, ToolStep

# 创建工作流
agent = SubAgent("document_analyzer")

# 添加步骤（链式调用）
agent.add_step(ToolStep(
    tool="file_reader",
    params={"path": "doc.txt"},
    output_key="content"
)).add_step(LLMStep(
    model="qwen2.5:7b",
    prompt="分析：{content}",
    output_key="analysis"
)).add_step(ToolStep(
    tool="file_writer",
    params={"path": "result.txt", "content": "{analysis}"}
))

# 执行
result = agent.run(input_data={"filename": "doc.txt"})
```

### 支持的步骤类型

| 步骤 | 说明 | 用途 |
|------|------|------|
| `LLMStep` | 大模型调用 | 文本生成、分析 |
| `ToolStep` | 工具调用（**支持 MCP**） | 文件操作、数据处理、API 调用 |
| `SubAgentStep` | SubAgent 嵌套调用 | 复杂工作流组合 |

### 配置文件支持

SubAgent 支持通过 YAML/JSON 配置文件定义工作流：

```yaml
name: "document_analyzer"
description: "分析文档内容"
steps:
  - type: "tool"
    name: "read_file"
    tool: "file_reader"
    params:
      path: "{file_path}"
    output_key: "content"
    
  - type: "llm"
    name: "analyze"
    model: "qwen2.5:7b"
    prompt: "分析: {content}"
    output_key: "analysis"
    
  - type: "tool"
    name: "save"
    tool: "file_writer"
    params:
      path: "result.txt"
      content: "{analysis}"
```

从配置文件加载：

```python
from local_agent.subagent import load_subagent_from_file

agent = load_subagent_from_file("config/analyzer.yaml")
result = agent.run(input_data={"file_path": "doc.txt"})
```

### SubAgent 嵌套调用

支持在 SubAgent 中调用其他 SubAgent：

```python
from local_agent.subagent import SubAgent, SubAgentStep, LLMStep

# 创建子 SubAgent
summarizer = SubAgent("summarizer")
summarizer.add_step(LLMStep(
    model="qwen2.5:7b",
    prompt="总结: {text}"
))

# 在主 SubAgent 中调用
analyzer = SubAgent("analyzer")
analyzer.add_step(LLMStep(
    model="qwen2.5:7b",
    prompt="分析: {input_text}",
    output_key="analysis"
)).add_step(SubAgentStep(
    subagent=summarizer,
    input_mapping={"text": "analysis"},
    output_key="summary"
))

result = analyzer.run(input_data={"input_text": "..."})
```

### ToolStep + MCP 集成示例

```python
# ToolStep 自动支持 MCP 工具
agent.add_step(ToolStep(
    tool="weather_get_current",  # MCP 工具
    params={"city": "北京"},
    auto_load_mcp=True  # 自动加载 MCP
))
```

### 📚 更多示例和文档

- **完整教程**: [tutorials/demo/README.md](tutorials/demo/README.md)
- **Python 示例**: [tutorials/demo/subagent_basic.py](tutorials/demo/subagent_basic.py)
  - 包含 4 个演示场景（基础工作流、YAML 加载、嵌套调用、上下文管理）
- **YAML 配置**: [tutorials/demo/subagent_advanced.yaml](tutorials/demo/subagent_advanced.yaml)
  - 真实可用的工作流配置文件
- **快速入门**: [tutorials/README.md](tutorials/README.md)

**运行示例：**
```bash
# 运行完整的 SubAgent 教程
python tutorials/demo/subagent_basic.py
```

---

## 🔧 内置工具（40+）

| 分类 | 工具数量 | 典型工具 |
|------|---------|---------|
| 📁 文件系统 | 10 | 读写文件、目录操作、搜索 |
| 💻 代码执行 | 6 | Python 执行、格式化、测试 |
| 🌐 网络搜索 | 5 | DuckDuckGo、Wikipedia、arXiv |
| 🖥️ 浏览器 | 7 | 导航、截图、提取文本 |
| 📊 数据分析 | 7 | CSV 处理、统计、可视化 |
| 🔧 Shell | 5 | 命令执行、进程管理 |
| ⚙️ 系统 | 10 | 时间、计算器、系统信息 |

**了解更多**：查看源代码注释和示例

---

## 🎯 内置技能（8个）

| 技能 | 说明 | 激活 |
|------|------|------|
| `code_assistant` | 代码编写和调试 | `--skill code_assistant` |
| `data_analyst` | 数据分析和可视化 | `--skill data_analyst` |
| `web_researcher` | 网络信息研究 | `--skill web_researcher` |
| `file_manager` | 文件管理 | `--skill file_manager` |
| `browser_automation` | 浏览器自动化 | `--skill browser_automation` |
| `task_planner` | 任务规划 | `--skill task_planner` |
| `sql_analyst` | SQL 查询和分析 | `--skill sql_analyst` |
| `system_admin` | 系统管理 | `--skill system_admin` |

**了解更多**：查看 `local_agent/skills/builtin/` 目录

### 技能配置（config.yaml）

在 `config.yaml` 的 `skills` 段中管理技能加载行为：

```yaml
skills:
  auto_load: true          # 是否自动加载技能，默认 true
  directories:
    - "./local_agent/skills/builtin"  # 内置技能（随 LocalAgent 发布，勿删）
    - "./skills"                       # 用户自定义技能目录
  # 未来保留字段
  # reload_on_change: false
  # hot_reload: false
```

**添加自定义技能目录**：只需在 `directories` 列表中追加新路径，LocalAgent 会自动扫描该目录下的子目录并加载符合约定的技能文件。

---

## 💻 使用方式

### 方式一：命令行

```bash
# 交互式聊天（使用默认配置）
python main.py

# 使用指定模型
python main.py -m qwen2.5:7b

# 使用 OpenAI
python main.py -m gpt-4

# 单次问答
python main.py --mode once -q "你的问题"

# 激活技能
python main.py --skill data_analyst

# 启动 Web 服务
python main.py --mode server
```

**终端命令**：
- `/models` - 列举所有 Provider 的可用模型及状态
- `/model` - 查看当前提供商和模型
- `/model <name>` - 切换模型（如：`/model qwen2.5:14b`）
- `/model <provider>:<model>` - 切换提供商（如：`/model openai:gpt-4`, `/model wanqing:ep-...`）
- `/help` - 查看所有可用命令

### 方式二：Python API

```python
from local_agent.core.agent import LocalAgent

# 使用 Ollama
agent = LocalAgent.create(model="qwen2.5:7b")

# 使用 OpenAI
agent = LocalAgent.create(model="gpt-4", provider="openai")

# 对话
response = agent.chat("列出当前目录的文件")
print(response)

# 流式输出
for token in agent.stream("写一个快速排序"):
    print(token, end="", flush=True)
```

### 方式三：SubAgent 编排

```python
from local_agent.subagent import SubAgent, LLMStep, ToolStep

# 创建复杂工作流
agent = SubAgent("workflow")
agent.add_step(LLMStep(...))
agent.add_step(ToolStep(...))
result = agent.run(input_data={...})
```

---

## 🏗️ 架构改进

LocalAgent 实现了统一的共享架构，并将底层运行时引擎完整合并至 `core/` 包。

### 模块合并：engine → core/engine

原 `local_agent/engine/` 已完整合并至 `local_agent/core/engine/`，实现了更清晰的包结构：

```
local_agent/core/
├── agent.py          # LocalAgent 主类
├── config.py         # 配置管理
├── graph.py          # ReAct 图工厂
├── state.py          # AgentState 状态
├── debug.py          # 调试工具
└── engine/           # ← 原 engine/ 整体移入
    ├── messages.py   # 消息类型（HumanMessage, AIMessage, ToolMessage…）
    ├── tools.py      # 工具基类（BaseTool）与 @tool 装饰器
    ├── react.py      # ReAct 循环引擎
    └── llm/          # LLM HTTP 客户端
        ├── base.py
        ├── ollama.py
        ├── openai_compat.py
        └── anthropic.py
```

**新推荐 import 路径：**
```python
from local_agent.core.engine.messages import HumanMessage, AIMessage
from local_agent.core.engine.tools import BaseTool, tool
from local_agent.core.engine.react import ReActEngine
from local_agent.core.engine.llm import OllamaLLM, OpenAILLM, AnthropicLLM

# 或直接从 core 包顶层导入（已再导出）
from local_agent.core import BaseTool, tool, HumanMessage, AIMessage
```

> **向后兼容**：原 `local_agent.engine.*` 路径仍然可用（保留了兼容垫片），但推荐迁移到新路径。

### 共享层 (`local_agent/shared/`)

提供统一的 Provider 访问，避免重复实例化：

```python
from local_agent.shared import get_llm_provider, get_tool_registry, get_mcp_manager

# 获取共享的 LLM Provider
provider = get_llm_provider(model="qwen2.5:7b")

# 获取共享的 Tool Registry
registry = get_tool_registry()

# 获取共享的 MCP Manager
mcp_manager = get_mcp_manager()
```

**优势**：
- ✅ 单例管理，避免重复创建
- ✅ 统一访问接口
- ✅ 易于测试和重置
- ✅ 减少模块间耦合

### ToolNode 增强 MCP 支持

ToolNode 原生支持 MCP 工具，无需额外配置：

```python
# 自动加载 MCP 工具
pipeline.add(ToolNode(
    tool="weather_get_current",  # MCP 工具名
    params={"city": "北京"},
    auto_load_mcp=True  # 自动加载
))

# 显式指定 MCP 服务器
pipeline.add(ToolNode(
    tool="get_current",
    mcp_server="weather",  # 指定服务器名
    params={"city": "上海"}
))

# 混合使用本地和 MCP 工具
pipeline = Pipeline("mixed_demo")

# 本地工具
pipeline.add(ToolNode(
    tool="file_reader",
    params={"path": "data.txt"}
))

# MCP 工具
pipeline.add(ToolNode(
    tool="translator_translate",
    params={"text": "{content}", "target": "en"},
    auto_load_mcp=True
))
```

### 统一工具调用路径

```
改进后：
所有路径 → 共享 ToolRegistry (get_tool_registry())
         ↓
    本地工具 + MCP 工具（统一注册）
```

**优势**：
- ✅ 工具行为一致
- ✅ 避免重复加载
- ✅ 统一管理和监控

---

## 🌍 MCP 支持

LocalAgent 支持 [Model Context Protocol](https://modelcontextprotocol.io)，可以像 Claude Desktop 一样接入任意 MCP 服务。

### MCP 配置

#### 方式一：在 config.yaml 中内联配置（推荐）

直接在 `config.yaml` 的 `mcp.servers` 段中定义服务器，无需维护单独的 JSON 文件：

```yaml
mcp:
  enabled: true
  servers:
    # stdio 模式（本地命令）
    filesystem:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
      description: "本地文件系统访问"
    github:
      command: "npx"
      args: ["-y", "@modelcontextprotocol/server-github"]
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"  # 支持 ${ENV_VAR} 占位符
      description: "GitHub 仓库管理"
    # sse 模式（远程服务）
    weather:
      url: "https://your-mcp-server/sse"
      description: "天气信息服务"
      enabled: true  # 可单独禁用某个服务器
```

**每个服务器支持的字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `command` | string | 可执行文件路径（stdio 模式，与 `url` 二选一） |
| `args` | list | 命令行参数列表 |
| `env` | dict | 额外的环境变量，支持 `${VAR}` 占位符 |
| `url` | string | SSE 端点 URL（sse 模式，与 `command` 二选一） |
| `enabled` | bool | 是否启用，默认 `true` |
| `description` | string | 人可读描述 |

#### 方式二：使用独立 JSON 文件（向后兼容）

编辑 `config/mcp.json`（与 Claude Desktop 格式兼容）：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
      "enabled": true,
      "description": "File system access"
    },
    "weather": {
      "command": "node",
      "args": ["./mcp-servers/weather/index.js"],
      "enabled": true,
      "description": "Weather information provider"
    }
  }
}
```

在 `config.yaml` 中指定文件路径：

```yaml
mcp:
  config_path: "./config/mcp.json"
  enabled: true
```

#### 两种方式同时使用

当 `config.yaml mcp.servers` 与 `config_path` 文件同时存在时，会自动合并，**`mcp.servers` 中的定义优先级更高**（同名服务器以 `config.yaml` 为准）。

### MCPNode vs ToolNode

#### MCPNode（保留，用于特殊场景）
```python
MCPNode(
    server="weather",
    tool="get_current",
    params={"city": "北京"}
)
```

**适用场景**：
- 需要显式控制 MCP 服务器连接
- 需要特定 MCP 服务器的上下文
- 调试和测试 MCP 服务

#### ToolNode（推荐，统一接口）
```python
ToolNode(
    tool="weather_get_current",
    params={"city": "北京"},
    auto_load_mcp=True  # 自动处理 MCP
)
```

**适用场景**：
- 日常使用（推荐）
- 不关心工具来源（本地 or MCP）
- 简化配置

---

## 🔄 多 LLM 提供商

### 支持的提供商

- **Ollama**：本地运行，保护隐私
- **OpenAI**：云端服务，GPT 系列模型
- **OpenAI 兼容**：Azure OpenAI、本地 API 等

### 配置方法

#### 通过 config.yaml（推荐）

```yaml
provider: "ollama"  # 或 "openai"

ollama:
  base_url: "http://localhost:11434"
  default_model: "qwen2.5:7b"
  timeout: 120

openai:
  api_key: ""  # 通过环境变量
  base_url: ""  # 可选
  default_model: "gpt-3.5-turbo"
```

#### 通过环境变量

创建 `.env` 文件：

```bash
# 选择提供商
LLM_PROVIDER=openai

# OpenAI 配置
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_DEFAULT_MODEL=gpt-4

# Ollama 配置（如需）
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=qwen2.5:7b
```

#### 代码中指定

```python
from local_agent.core.agent import LocalAgent

# 使用 Ollama
agent = LocalAgent.create(
    model="qwen2.5:7b",
    provider="ollama"
)

# 使用 OpenAI
agent = LocalAgent.create(
    model="gpt-4",
    provider="openai"
)
```

### 快速切换

```bash
# 终端命令
/model openai:gpt-4
/model ollama:qwen2.5:7b

# Python API
agent = LocalAgent.create(model="gpt-4", provider="openai")
```

### OpenAI 兼容接口示例

#### Azure OpenAI

```yaml
openai:
  api_key: "your-azure-key"
  base_url: "https://your-resource.openai.azure.com/openai/deployments/your-deployment"
  default_model: "gpt-4"
```

#### 本地 API 服务器

```yaml
openai:
  api_key: "dummy"  # 某些本地服务器可能不需要
  base_url: "http://localhost:8000/v1"
  default_model: "vicuna-7b"
```

### 性能对比

| 特性 | Ollama | OpenAI |
|------|--------|--------|
| 成本 | 免费（硬件成本） | 按 Token 计费 |
| 延迟 | 低（本地） | 中等（网络） |
| 隐私 | 完全本地 | 云端处理 |
| 模型质量 | 取决于硬件 | GPT-4 级别 |
| 可用性 | 依赖本地服务 | 99.9% SLA |

---

## 🔌 扩展

### 添加自定义工具

```python
# tools/my_tools.py
from local_agent.core.engine.tools import tool
# 或等价写法（推荐新代码使用完整路径）:
# from local_agent.core import tool

@tool
def my_custom_tool(input: str) -> str:
    """自定义工具描述"""
    return f"处理结果: {input}"

TOOLS = [my_custom_tool]
```

LocalAgent 会自动扫描并加载！

### 添加自定义技能

```python
# skills/my_skill/skill.py
from local_agent.skills.base import BaseSkill, SkillConfig

class MySkill(BaseSkill):
    @classmethod
    def get_config(cls) -> SkillConfig:
        return SkillConfig(
            name="my_skill",
            description="我的自定义技能",
            required_tools=["file_reader"],
            system_prompt="你是一个专业的...",
        )

SKILL = MySkill()
```

---

## 📊 项目结构

```
LocalAgent/
├── local_agent/           # 核心代码
│   ├── core/              # Agent 核心（含运行时引擎）
│   │   ├── agent.py       # LocalAgent 主类
│   │   ├── config.py      # 配置管理
│   │   ├── graph.py       # ReAct 图工厂
│   │   ├── state.py       # AgentState 状态定义
│   │   ├── debug.py       # 调试输出工具
│   │   └── engine/        # 底层运行时引擎（原 engine/ 模块合并至此）
│   │       ├── messages.py    # 消息类型（BaseMessage, HumanMessage, AIMessage…）
│   │       ├── tools.py       # 工具基类（BaseTool）与 @tool 装饰器
│   │       ├── react.py       # ReAct 循环引擎
│   │       └── llm/           # LLM HTTP 客户端
│   │           ├── base.py        # 抽象基类
│   │           ├── ollama.py      # Ollama 客户端
│   │           ├── openai_compat.py # OpenAI 兼容客户端
│   │           └── anthropic.py   # Anthropic/Claude 客户端
│   ├── engine/            # ⚠️ 向后兼容垫片（已合并至 core/engine/，此目录仅转发 import）
│   ├── tools/             # 工具系统
│   ├── skills/            # 技能系统
│   ├── subagent/          # SubAgent 编排
│   ├── mcp/               # MCP 集成
│   ├── llm/               # LLM 提供者适配层
│   ├── memory/            # 记忆系统
│   ├── shared/            # 共享层（统一 Provider 管理）
│   ├── cli/               # CLI 接口
│   └── api/               # Web API
├── tutorials/             # 教程和示例
│   ├── demo/              # 演示示例
│   └── test/              # 测试脚本
├── config/                # 向后兼容目录
│   └── mcp.json           # MCP 配置（仍被加载，优先级低于 config.yaml）
├── docs/                  # 文档
├── frontend/              # Web 前端
├── config.yaml            # 主配置文件
├── main.py                # 主入口
└── README.md              # 本文件
```

---

## 📚 教程和示例

### 目录结构

```
tutorials/
├── demo/                 # 演示示例
│   ├── subagent_demo.py              # SubAgent 系统快速入门
│   ├── document_analyzer.yaml         # 配置文件示例
│   └── README.md                      # 详细使用说明
└── test/                 # 测试脚本
    └── test_multi_provider.py         # 多 LLM 提供商测试
```

### 演示示例（tutorials/demo/）

**SubAgent 快速入门**：
```bash
# SubAgent 系统演示
python tutorials/demo/subagent_demo.py

# 使用配置文件运行 SubAgent
python -c "
from local_agent.subagent import load_subagent_from_file
agent = load_subagent_from_file('tutorials/demo/document_analyzer.yaml')
result = agent.run(input_data={'file_path': 'doc.txt'})
print(result)
"
```

### 测试脚本（tutorials/test/）

**多提供商测试**：
```bash
# 测试 Ollama 和 OpenAI 提供商
python tutorials/test/test_multi_provider.py
```

---

## 🎯 最佳实践

### 开发阶段
- 使用 Ollama 本地模型，快速迭代
- 启用调试日志，查看详细调用信息

### 生产环境
- 根据需求选择 OpenAI 或私有部署
- 使用环境变量管理敏感信息
- 启用 API Key 密钥管理服务

### 混合使用
- 简单任务用本地模型
- 复杂任务用云端模型

### 工具调用
1. ✅ **优先使用 ToolNode**（而不是 MCPNode）
2. ✅ **启用 auto_load_mcp** 获得最佳体验
3. ✅ **使用 output_key** 明确数据流
4. ✅ **参数支持变量替换** `{variable_name}`
5. ✅ **错误处理** 捕获 ValueError 和 ConnectionError

### API Key 安全
- 使用环境变量，不要硬编码
- `.env` 文件加入 `.gitignore`
- 生产环境使用密钥管理服务

---

## 🐛 调试配置

LocalAgent 提供了灵活的调试选项，帮助你查看运行时的详细信息。

### 调试模式配置

编辑 `config.yaml`：

```yaml
debug:
  print_mode: false  # 启用打印调试模式
  print_model_input: true  # 打印模型输入
  print_model_output: true  # 打印模型输出
  print_tool_calls: true  # 打印工具调用和结果
```

### 配置说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `print_mode` | 启用打印调试模式，控制是否在终端输出调试信息 | `false` |
| `print_model_input` | 打印发送给 LLM 的输入内容（包括用户消息、系统提示等） | `true` |
| `print_model_output` | 打印 LLM 返回的原始输出内容 | `true` |
| `print_tool_calls` | 打印工具调用的名称、参数和返回结果 | `true` |

### 使用场景

#### 开发调试
```yaml
debug:
  print_mode: true
  print_model_input: true
  print_model_output: true
  print_tool_calls: true
```
**适合**：开发新工具、调试 Prompt、排查问题

#### 生产环境
```yaml
debug:
  print_mode: false
  print_model_input: false
  print_model_output: false
  print_tool_calls: false
```
**适合**：生产部署，减少日志输出

#### 工具调试
```yaml
debug:
  print_mode: true
  print_model_input: false
  print_model_output: false
  print_tool_calls: true
```
**适合**：仅关注工具调用是否正确

### 示例输出

启用调试后，终端会显示详细信息：

```
[DEBUG] Model Input:
  System: You are a helpful assistant
  User: 列出当前目录的文件

[DEBUG] Tool Call:
  Name: file_reader
  Params: {"path": "."}
  Result: [main.py, config.yaml, ...]

[DEBUG] Model Output:
  当前目录包含以下文件：main.py, config.yaml...
```

---

## 🛠️ 故障排查

### Ollama 连接失败

```bash
# 检查服务是否运行
ollama serve

# 测试连接
curl http://localhost:11434/api/tags
```

### OpenAI 认证失败

```bash
# 验证 API Key
export OPENAI_API_KEY=sk-...
python -c "from openai import OpenAI; print(OpenAI().models.list())"
```

### 配置文件找不到

```bash
# 方式 1: 环境变量
export MCP_CONFIG_PATH="./config/mcp.json"

# 方式 2: config.yaml（推荐，在 mcp.servers 直接内联定义）
mcp:
  servers:
    my_server:
      command: "npx"
      args: ["-y", "my-mcp-server"]
```

### 工具未找到

```python
try:
    result = pipeline.run()
except ValueError as e:
    print(f"工具未找到: {e}")
    # 检查：
    # 1. 工具名称是否正确
    # 2. 是否启用 auto_load_mcp
    # 3. mcp_servers.json 是否配置
```

### MCP 连接失败

```python
try:
    result = pipeline.run()
except ConnectionError as e:
    print(f"MCP 连接失败: {e}")
    # 检查：
    # 1. MCP 服务是否运行
    # 2. mcp_servers.json 配置是否正确
    # 3. 网络连接是否正常
```

---

## 🔮 未来改进方向

### 工具发现机制
- [ ] 自动扫描和注册 MCP 工具
- [ ] 工具元数据查询接口
- [ ] 工具使用统计和监控

### 更好的 MCP 集成
- [ ] 异步 MCP 工具调用
- [ ] MCP 工具热重载
- [ ] MCP 工具版本管理

### 统一配置
- [x] 统一 Config 管理（`config.yaml` 现已内联 MCP servers 和 Skills 配置）
- [ ] 环境变量支持增强
- [ ] 配置验证和提示

### 性能优化
- [ ] 迁移 LocalAgent 使用共享层
- [ ] 添加更多测试确保兼容性
- [ ] 性能监控和优化

---

## 🤝 贡献

欢迎贡献代码、文档或报告问题！

1. Fork 本仓库
2. 创建特性分支
3. 提交代码
4. 发起 Pull Request

---

## 📄 许可证

[MIT License](LICENSE)

---

## 🙏 致谢

- [Ollama](https://ollama.ai) - 本地 LLM 运行
- [OpenAI](https://openai.com) - GPT 系列模型
- [Anthropic](https://anthropic.com) - Claude 系列模型
- [Model Context Protocol](https://modelcontextprotocol.io) - MCP 协议标准

---

<div align="center">

Made with ❤️ by LocalAgent Team

</div>
