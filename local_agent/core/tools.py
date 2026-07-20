"""
local_agent.engine.tools
=========================
自主实现的工具基类与 @tool 装饰器，替换 langchain_core.tools。

设计目标：
  1. @tool 装饰器签名与 langchain @tool 完全兼容 —— 现有工具文件无需修改逻辑，
     只需将 import 行换为本模块。
  2. BaseTool 提供统一的 run() / arun() 接口，以及生成 OpenAI function-calling
     JSON Schema 的 get_schema() 方法（供 LLM 客户端注入工具定义）。
  3. LocalAgentTool 是 BaseTool 的扩展，增加 category / requires_confirmation 元数据。
  4. tool_metadata 装饰器可为已有 @tool 函数注入元数据（兼容原有写法）。

JSON Schema 生成规则（get_schema）：
  - 从函数类型注解自动推导参数类型（str/int/float/bool/list/dict → JSON 类型）
  - 未注解参数默认类型为 string
  - 含默认值的参数标记为 non-required
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    get_type_hints,
)

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 类型注解 → JSON Schema 类型映射
# ---------------------------------------------------------------------------

_PY_TO_JSON_TYPE: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    List: "array",
    Dict: "object",
}


def _py_type_to_json(py_type: Any) -> str:
    """将 Python 类型注解转为 JSON Schema 类型字符串。"""
    # 处理 Optional[X] / Union[X, None]
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        import typing
        # Optional[X] == Union[X, None] → 取第一个非 None 参数
        args = getattr(py_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _py_type_to_json(non_none[0])
    return _PY_TO_JSON_TYPE.get(py_type, "string")


def _build_schema_from_func(func: Callable) -> Dict[str, Any]:
    """从函数签名自动构建 JSON Schema（parameters 部分）。"""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = {}

    # 尝试解析 docstring 中的参数说明（Google/Numpy style 不支持，仅支持 :param x: 格式）
    doc = inspect.getdoc(func) or ""

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        # skip **kwargs
        if param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue

        py_type = hints.get(param_name, str)
        json_type = _py_type_to_json(py_type)

        prop: Dict[str, Any] = {"type": json_type}

        # 从 docstring 提取参数描述（简单匹配 `:param name: desc` 或 `name: desc`）
        import re
        m = re.search(
            rf"(?::param {param_name}:|{param_name}\s*\(.*?\)\s*:|{param_name}\s*:)\s*([^\n]+)",
            doc,
        )
        if m:
            prop["description"] = m.group(1).strip()

        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------

class BaseTool:
    """
    所有 LocalAgent 工具的基类。

    子类必须实现 _run()（同步），可选实现 _arun()（异步）。
    若未实现 _arun()，默认在线程池中运行 _run()。

    Attributes:
        name        : 工具名称（全局唯一，必填）
        description : 工具功能描述（传给 LLM 的 function description）
        args_schema : 可选 Pydantic BaseModel，用于入参校验
        metadata    : 额外元数据字典（供注册表使用）
    """

    name: str = "base_tool"
    description: str = ""
    args_schema: Optional[Type[BaseModel]] = None
    metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 执行入口
    # ------------------------------------------------------------------

    def run(self, **kwargs: Any) -> str:
        """同步执行工具。入参来自 LLM tool_call.args。"""
        try:
            if self.args_schema is not None:
                validated = self.args_schema(**kwargs)
                kwargs = validated.model_dump()
            return self._run(**kwargs)
        except Exception as exc:
            logger.error("Tool '%s' run failed: %s", self.name, exc, exc_info=True)
            return f"[Tool Error] {exc}"

    async def arun(self, **kwargs: Any) -> str:
        """异步执行工具。默认回退到线程池中的同步实现。"""
        try:
            if self.args_schema is not None:
                validated = self.args_schema(**kwargs)
                kwargs = validated.model_dump()
            return await self._arun(**kwargs)
        except Exception as exc:
            logger.error("Tool '%s' arun failed: %s", self.name, exc, exc_info=True)
            return f"[Tool Error] {exc}"

    @abstractmethod
    def _run(self, **kwargs: Any) -> str:  # type: ignore[return]
        """子类实现：同步执行逻辑。"""
        ...

    async def _arun(self, **kwargs: Any) -> str:
        """子类可选覆盖：异步执行逻辑。默认委托给 _run()。"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._run(**kwargs))

    # ------------------------------------------------------------------
    # Schema 生成（供 LLM 客户端构建 tools 参数）
    # ------------------------------------------------------------------

    def get_schema(self) -> Dict[str, Any]:
        """
        返回 OpenAI function-calling 格式的工具定义：

        {
          "type": "function",
          "function": {
            "name": "...",
            "description": "...",
            "parameters": { ... JSON Schema ... }
          }
        }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._get_parameters_schema(),
            },
        }

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """从 args_schema 或 _run 签名推导参数 JSON Schema。"""
        if self.args_schema is not None:
            try:
                return self.args_schema.model_json_schema()
            except Exception:
                pass
        # 回退：从 _run 方法签名构建
        return _build_schema_from_func(self._run)

    # ------------------------------------------------------------------
    # 兼容旧代码：allow attribute access via [] and .get()
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# FunctionTool – @tool 装饰器生成的工具类
# ---------------------------------------------------------------------------

class FunctionTool(BaseTool):
    """
    由 @tool 装饰器动态生成的工具，将普通函数包装为 BaseTool 实例。
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self._func = func
        self.name = name or func.__name__
        self.description = description or (inspect.getdoc(func) or "").strip().split("\n")[0]
        self.args_schema = None
        self.metadata: Dict[str, Any] = {}
        self._schema_cache: Optional[Dict[str, Any]] = None

    def _run(self, **kwargs: Any) -> str:
        result = self._func(**kwargs)
        if not isinstance(result, str):
            try:
                result = json.dumps(result, ensure_ascii=False, default=str)
            except Exception:
                result = str(result)
        return result

    async def _arun(self, **kwargs: Any) -> str:
        if asyncio.iscoroutinefunction(self._func):
            result = await self._func(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self._func(**kwargs))
        if not isinstance(result, str):
            try:
                result = json.dumps(result, ensure_ascii=False, default=str)
            except Exception:
                result = str(result)
        return result

    def _get_parameters_schema(self) -> Dict[str, Any]:
        if self._schema_cache is None:
            self._schema_cache = _build_schema_from_func(self._func)
        return self._schema_cache

    def __repr__(self) -> str:
        return f"FunctionTool(name={self.name!r})"


# ---------------------------------------------------------------------------
# @tool 装饰器
# ---------------------------------------------------------------------------

def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """
    将普通函数包装为 BaseTool 实例的装饰器。

    与 langchain @tool 用法完全兼容::

        @tool
        def my_tool(path: str) -> str:
            "Read a file."
            ...

        @tool(name="custom_name", description="Custom description")
        def my_tool(path: str) -> str:
            ...
    """
    def _make_tool(f: Callable) -> FunctionTool:
        return FunctionTool(f, name=name, description=description)

    if func is not None:
        # @tool（不带括号）
        return _make_tool(func)
    # @tool(name=...) 或 @tool()
    return _make_tool


# ---------------------------------------------------------------------------
# LocalAgentTool – 带元数据的工具基类（兼容 tools/base.py 原有接口）
# ---------------------------------------------------------------------------

class LocalAgentTool(BaseTool):
    """
    LocalAgent 类实现工具的推荐基类。

    在 BaseTool 基础上增加：
      - category            : 工具分类（用于 UI 分组）
      - requires_confirmation: 执行前是否需要用户确认
      - enabled             : 是否启用
    """

    # 子类通过类变量声明这些字段
    category: str = "general"
    requires_confirmation: bool = False
    enabled: bool = True

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        # 自动从类变量初始化 metadata（供 ToolRegistry 使用）
        # 注意：子类可能通过 class body 赋值覆盖
        pass

    @property
    def _metadata_dict(self) -> Dict[str, Any]:
        return {
            "category": self.__class__.__dict__.get("category", self.category),
            "requires_confirmation": self.__class__.__dict__.get(
                "requires_confirmation", self.requires_confirmation
            ),
        }

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """返回 JSON 可序列化的元数据字典（供 ToolRegistry 使用）。"""
        return {
            "name": cls.__dict__.get("name", cls.__name__),
            "description": cls.__dict__.get("description", cls.__doc__ or ""),
            "category": cls.__dict__.get("category", "general"),
            "requires_confirmation": cls.__dict__.get("requires_confirmation", False),
        }


# ---------------------------------------------------------------------------
# tool_metadata 装饰器（兼容 tools/base.py 原有接口）
# ---------------------------------------------------------------------------

def tool_metadata(
    category: str = "general",
    requires_confirmation: bool = False,
) -> Callable:
    """
    为 @tool 函数注入 LocalAgent 元数据的装饰器。

    可叠加使用（必须写在 @tool 外层）::

        @tool_metadata(category="filesystem")
        @tool
        def read_file(path: str) -> str:
            "Read a file."
            ...
    """

    def decorator(func_or_tool: Any) -> Any:
        # 若未经 @tool 装饰，先包装
        if not isinstance(func_or_tool, BaseTool):
            wrapped: BaseTool = tool(func_or_tool)
        else:
            wrapped = func_or_tool

        # 写入元数据
        if wrapped.metadata is None:
            wrapped.metadata = {}
        wrapped.metadata["category"] = category
        wrapped.metadata["requires_confirmation"] = requires_confirmation
        return wrapped

    return decorator
