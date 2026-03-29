# OpenClaw Skills

Skills are self-contained plugins that add tools to the OpenClaw agent.

## Directory Structure

```
skills/
├── browser/          ← Built-in: fetch web pages
│   └── skill.py
├── code_exec/        ← Built-in: run Python scripts
│   └── skill.py
├── web_search/       ← Built-in: DuckDuckGo search
│   └── skill.py
├── file_ops/         ← Built-in: read/write files
│   └── skill.py
├── calculator/       ← Example custom skill
│   └── skill.py
└── your_skill/       ← Your custom skill!
    └── skill.py
```

## Creating a Custom Skill

1. Create a new folder: `skills/my_skill/`
2. Create `skills/my_skill/skill.py` with this template:

```python
from typing import List, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from core.tool_base import OpenClawSkill


class MyToolInput(BaseModel):
    param: str = Field(description="What this parameter does")


class MyTool(BaseTool):
    name: str = "my_tool_name"
    description: str = "What this tool does (used by the AI to decide when to call it)"
    args_schema: Type[BaseModel] = MyToolInput

    def _run(self, param: str) -> str:
        # Your tool logic here
        return f"Result: {param}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class MySkill(OpenClawSkill):
    name = "my_skill"
    description = "Brief description of what this skill provides"
    version = "1.0.0"

    def get_tools(self) -> List[BaseTool]:
        return [MyTool()]
```

3. Add `"my_skill"` to the `skills` list in `config/config.yaml`
4. Restart OpenClaw

## Tips

- The `description` of each tool is crucial — the LLM uses it to decide when to call the tool
- Tools can have multiple parameters via the Pydantic schema
- Use `on_load()` for initialization (e.g., loading API keys, starting connections)
- Use `on_unload()` for cleanup
- Install any extra dependencies in your skill's `requirements.txt`

## Built-in Skills Summary

| Skill | Tools | Description |
|-------|-------|-------------|
| `browser` | `browser_fetch_page`, `browser_extract_links` | Fetch and parse web pages |
| `code_exec` | `code_run_python`, `code_eval_expression` | Execute Python code |
| `web_search` | `web_search` | DuckDuckGo search (no API key) |
| `file_ops` | `file_read`, `file_write`, `file_list_dir` | Local file operations |
| `calculator` | `calculator` | Safe math expressions |
