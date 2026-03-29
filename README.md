# 🤖 LocalAgent

> A local AI agent powered by **Ollama** + **LangGraph**, with a modular skill system.

---

## Features

- 🤖 **LangGraph ReAct agent** — structured tool-calling with full conversation memory
- 🦙 **Ollama backend** — 100% local, no API keys, works with any pulled model
- 🔌 **Modular skills** — drop a `skill.py` into `skills/` and it's auto-loaded
- 🌐 **Browser skill** — fetch and parse any web page
- 🔍 **Web search** — DuckDuckGo search, no key required
- 🐍 **Code execution** — run Python scripts as subprocesses with timeout protection
- 📁 **File ops** — read, write, list local files
- 🖥️ **Rich terminal UI** — beautiful interactive REPL with markdown rendering
- ⚙️ **YAML config** — model, skills, timeouts all configurable

---

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running

---

## Quick Start

```bash
# 1. Clone / copy this project
cd LocalAgent

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull a model in Ollama
ollama pull qwen2.5:7b        # recommended
# or: ollama pull llama3.2:3b
# or: ollama pull mistral:7b

# 4. Start Ollama
ollama serve

# 5. Run LocalAgent (interactive)
python main.py

# Single-shot mode
python main.py "What is 2 to the power of 32?"

# Use a different model
python main.py --model llama3.2:3b

# Custom config file
python main.py --config my_config.yaml
```

---

## Configuration

Edit `config/config.yaml`:

```yaml
ollama:
  model: "qwen2.5:7b"     # any model you have pulled
  base_url: "http://localhost:11434"
  temperature: 0.1

skills:
  - browser
  - code_exec
  - web_search
  - file_ops
  # - calculator     # enable by uncommenting
  # - my_skill       # your custom skill
```

**Environment variable overrides:**
```bash
OLLAMA_MODEL=llama3.2:3b python main.py
OLLAMA_BASE_URL=http://remote:11434 python main.py
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/skills` | List loaded skills and tools |
| `/clear` | Clear conversation history |
| `/model` | Show current model info |
| `/exit` | Quit |

---

## Adding Custom Skills

1. Create `skills/my_skill/skill.py`:

```python
from typing import List, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from core.skill_base import OpenClawSkill

class MyInput(BaseModel):
    query: str = Field(description="What to look up")

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "Does something useful with the query"
    args_schema: Type[BaseModel] = MyInput

    def _run(self, query: str) -> str:
        return f"Result for {query}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)

class MySkill(OpenClawSkill):
    name = "my_skill"
    description = "My custom skill"
    version = "1.0.0"

    def get_tools(self) -> List[BaseTool]:
        return [MyTool()]
```

2. Add `"my_skill"` to `skills:` in `config/config.yaml`

See `skills/README.md` for full documentation.

---

## Project Structure

```
LocalAgent/
├── main.py                  # Entry point
├── requirements.txt
├── config/
│   └── config.yaml          # Main config
├── core/
│   ├── agent.py             # LangGraph agent
│   ├── config_loader.py     # YAML config loader
│   ├── skill_base.py        # Base class + registry
│   └── skill_loader.py      # Dynamic skill loader
├── ui/
│   └── cli.py               # Rich terminal UI
└── skills/
    ├── README.md            # How to write skills
    ├── browser/             # Web fetch skill
    ├── code_exec/           # Python execution skill
    ├── web_search/          # DuckDuckGo skill
    ├── file_ops/            # File I/O skill
    └── calculator/          # Example custom skill
```

---

## Recommended Models

| Model | Size | Best For |
|-------|------|----------|
| `qwen2.5:7b` | 4.7GB | Balanced, good tool calling |
| `qwen2.5:14b` | 9GB | Better reasoning |
| `llama3.2:3b` | 2GB | Fast, lightweight |
| `mistral:7b` | 4.1GB | Good general use |
| `deepseek-r1:7b` | 4.7GB | Strong reasoning |

---

## License

MIT
