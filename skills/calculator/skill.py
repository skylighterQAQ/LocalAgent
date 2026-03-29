"""
Example custom skill: Calculator
Place this file at skills/calculator/skill.py to activate it.
Then add "calculator" to the skills list in config/config.yaml.
"""
import math
from typing import List, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.skill_base import OpenClawSkill


class CalcInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate, e.g. 'sqrt(144) + pi'")


class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = (
        "A safe mathematical calculator. Supports: +, -, *, /, **, sqrt, log, sin, cos, tan, pi, e. "
        "Example: 'sqrt(2) * pi' or '2**10 + 1'"
    )
    args_schema: Type[BaseModel] = CalcInput

    # Safe math namespace
    _SAFE_GLOBALS = {
        "sqrt": math.sqrt,
        "log": math.log,
        "log2": math.log2,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "ceil": math.ceil,
        "floor": math.floor,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
        "inf": math.inf,
        "__builtins__": {},
    }

    def _run(self, expression: str) -> str:
        try:
            result = eval(expression, self._SAFE_GLOBALS)
            return f"{expression} = {result}"
        except Exception as ex:
            return f"Error evaluating '{expression}': {ex}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class CalculatorSkill(OpenClawSkill):
    name = "calculator"
    description = "Safe mathematical calculator with common math functions"
    version = "1.0.0"

    def get_tools(self) -> List[BaseTool]:
        return [CalculatorTool()]
