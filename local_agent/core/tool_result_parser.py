"""
local_agent.core.tool_result_parser
=====================================
ToolResultParser – 工具执行结果的结构化解析器。

核心功能：
  1. 将工具返回的原始字符串解析为结构化数据
  2. 生成简洁的摘要文本，供下一步 LLM 使用（而非原始长字符串）
  3. 在终端以结构化方式展示工具结果（而非原始 JSON dump）

支持的工具结果格式：
  - JSON 格式 → 直接解析为 dict/list
  - 搜索结果 → 提取 title/url/snippet 列表
  - 代码执行结果 → 提取 stdout/stderr/return_code
  - 文件/目录列表 → 解析为文件路径列表
  - 纯文本 → 提取关键句生成摘要

使用示例::

    parser = ToolResultParser()
    result = parser.parse("search_web", raw_tool_output)
    # result.summary  → 给 LLM 的简洁文本
    # result.structured_data → 结构化内容
    print(result.format_for_display())
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── 数据模型 ─────────────────────────────────────────────────────────────────

class ParsedToolResult(BaseModel):
    """工具执行结果解析后的结构化表示。"""

    tool_name: str = Field(description="工具名称")
    status: str = Field(
        default="success",
        description="执行状态: 'success' | 'error' | 'partial'",
    )
    summary: str = Field(
        default="",
        description="给下一步 LLM 使用的简洁摘要文本",
    )
    structured_data: Optional[Any] = Field(
        default=None,
        description="解析出的结构化内容（dict/list/None）",
    )
    raw: str = Field(default="", description="原始结果字符串")
    truncated: bool = Field(default=False, description="结果是否被截断")
    item_count: Optional[int] = Field(
        default=None,
        description="条目数（搜索结果数、文件数等）",
    )

    def format_for_display(self, max_lines: int = 30) -> str:
        """
        生成适合终端展示的格式化字符串。

        Args:
            max_lines: 最多展示的行数

        Returns:
            格式化字符串
        """
        lines = []
        status_icon = "✓" if self.status == "success" else ("⚠" if self.status == "partial" else "✗")
        
        header = f"[{status_icon} {self.tool_name}]"
        if self.item_count is not None:
            header += f" {self.item_count} 条结果"
        lines.append(header)
        
        if self.status == "error":
            lines.append(f"错误: {self.summary}")
            return "\n".join(lines)

        # 结构化展示
        if self.structured_data is not None:
            lines.extend(self._format_structured(self.structured_data, max_lines - 2))
        elif self.summary:
            # 纯文本摘要
            summary_lines = self.summary.splitlines()
            lines.extend(summary_lines[: max_lines - 2])
            if len(summary_lines) > max_lines - 2:
                lines.append(f"  ... (共 {len(summary_lines)} 行)")

        if self.truncated:
            lines.append(f"  [注: 结果已截断，原始长度 {len(self.raw)} 字符]")

        return "\n".join(lines)

    def _format_structured(self, data: Any, max_lines: int) -> List[str]:
        """递归格式化结构化数据。"""
        lines = []
        if isinstance(data, list):
            for i, item in enumerate(data[:max_lines]):
                if isinstance(item, dict):
                    # 搜索结果/文件条目等
                    title = item.get("title") or item.get("name") or item.get("path") or ""
                    url = item.get("url") or item.get("link") or ""
                    snippet = item.get("snippet") or item.get("description") or item.get("content") or ""
                    if title and url:
                        lines.append(f"  {i+1}. {title}")
                        lines.append(f"     {url}")
                        if snippet:
                            lines.append(f"     {snippet[:120]}")
                    elif title:
                        lines.append(f"  {i+1}. {title}")
                        if snippet:
                            lines.append(f"     {snippet[:120]}")
                    else:
                        lines.append(f"  {i+1}. {str(item)[:150]}")
                else:
                    lines.append(f"  {i+1}. {str(item)[:150]}")
            if len(data) > max_lines:
                lines.append(f"  ... (共 {len(data)} 条，仅显示前 {max_lines} 条)")
        elif isinstance(data, dict):
            for key, val in list(data.items())[:max_lines]:
                if isinstance(val, (dict, list)):
                    lines.append(f"  {key}: {json.dumps(val, ensure_ascii=False)[:200]}")
                else:
                    lines.append(f"  {key}: {str(val)[:200]}")
        else:
            lines.append(f"  {str(data)[:400]}")
        return lines

    def to_llm_context(self) -> str:
        """
        生成注入下一步 LLM 的上下文字符串。

        格式：
          [工具结果: tool_name | 状态: success]
          摘要内容...
        """
        lines = [f"[工具结果: {self.tool_name} | 状态: {self.status}]"]
        if self.item_count is not None:
            lines.append(f"共 {self.item_count} 条结果")
        lines.append("")
        lines.append(self.summary)
        if self.truncated:
            lines.append(f"\n[注: 原始结果已截断，完整内容约 {len(self.raw)} 字符]")
        return "\n".join(lines)


# ─── ToolResultParser 类 ─────────────────────────────────────────────────────

class ToolResultParser:
    """
    工具执行结果的结构化解析器。

    针对不同类型的工具输出，选择合适的解析策略：
      - search_* 工具 → 搜索结果格式
      - code_execute_* 工具 → 代码执行结果格式
      - fs_* 工具 → 文件系统结果格式
      - 其他 → 通用文本解析
    """

    # 工具名前缀到解析策略的映射
    _STRATEGY_MAP = {
        "search_": "_parse_search_result",
        "fetch_url": "_parse_text_content",
        "browser_get_text": "_parse_text_content",
        "browser_navigate": "_parse_text_content",
        "browser_extract_links": "_parse_links_result",
        "fs_list_dir": "_parse_file_list",
        "fs_read_file": "_parse_file_content",
        "fs_write_file": "_parse_write_result",
        "fs_search_files": "_parse_file_list",
        "fs_grep": "_parse_grep_result",
        "code_execute_": "_parse_code_result",
        "code_run_": "_parse_code_result",
        "shell_run": "_parse_code_result",
        "project_tree": "_parse_file_list",
        "git_": "_parse_git_result",
        # invoke_skill 返回子 skill 的完整结果，不应截断
        "invoke_skill": "_parse_skill_result",
    }

    def parse(self, tool_name: str, raw_result: str) -> ParsedToolResult:
        """
        解析工具执行结果。

        Args:
            tool_name:  工具名称
            raw_result: 工具返回的原始字符串

        Returns:
            ParsedToolResult 实例
        """
        # 错误结果直接处理
        if raw_result.startswith("[Tool Error]"):
            return ParsedToolResult(
                tool_name=tool_name,
                status="error",
                summary=raw_result,
                raw=raw_result,
            )

        # 选择解析策略
        strategy_method = self._select_strategy(tool_name)

        try:
            parser_fn = getattr(self, strategy_method)
            return parser_fn(tool_name, raw_result)
        except Exception as exc:
            logger.warning(
                "ToolResultParser: strategy '%s' failed for '%s': %s",
                strategy_method, tool_name, exc,
            )
            # Fallback：通用文本解析
            return self._parse_generic(tool_name, raw_result)

    # ── 策略选择 ─────────────────────────────────────────────────────────────

    def _select_strategy(self, tool_name: str) -> str:
        """根据工具名选择解析策略方法名。"""
        for prefix, method in self._STRATEGY_MAP.items():
            if tool_name.startswith(prefix) or tool_name == prefix.rstrip("_"):
                return method
        return "_parse_generic"

    # ── 解析策略实现 ─────────────────────────────────────────────────────────

    def _parse_search_result(self, tool_name: str, raw: str) -> ParsedToolResult:
        """解析搜索类工具结果（search_web、search_news 等）。"""
        # 尝试 JSON 解析
        structured = _try_parse_json(raw)
        if structured and isinstance(structured, list):
            items = structured
        elif structured and isinstance(structured, dict):
            items = structured.get("results") or structured.get("items") or [structured]
        else:
            # 文本格式：尝试按 "Title:/URL:/Snippet:" 块提取结构化条目
            items = _extract_search_items(raw)
            # 如果未能提取到任何结构化条目，fallback 到按行提取
            if not items:
                items = _extract_text_items(raw)

        # 生成摘要
        summary_lines = []
        for i, item in enumerate(items[:10]):
            if isinstance(item, dict):
                title = item.get("title", "")
                url = item.get("url") or item.get("link") or ""
                snippet = item.get("snippet") or item.get("description") or ""
                if title or url:
                    entry = f"{i+1}. {title}"
                    if url:
                        entry += f"\n   URL: {url}"
                    if snippet:
                        entry += f"\n   摘要: {snippet[:200]}"
                    summary_lines.append(entry)
            else:
                summary_lines.append(f"{i+1}. {str(item)[:200]}")

        summary = "\n".join(summary_lines) if summary_lines else raw[:500]

        return ParsedToolResult(
            tool_name=tool_name,
            status="success",
            summary=summary,
            structured_data=items,
            raw=raw,
            truncated=len(raw) > 4000,
            item_count=len(items),
        )

    def _parse_text_content(self, tool_name: str, raw: str) -> ParsedToolResult:
        """解析网页文本内容（fetch_url、browser_get_text 等）。"""
        # 全量保留内容，不截断
        return ParsedToolResult(
            tool_name=tool_name,
            status="success",
            summary=raw,
            structured_data=None,
            raw=raw,
            truncated=False,
        )

    def _parse_links_result(self, tool_name: str, raw: str) -> ParsedToolResult:
        """解析链接提取结果。"""
        structured = _try_parse_json(raw)
        if structured and isinstance(structured, list):
            links = structured
        else:
            # 从文本中提取 URL
            url_pattern = re.compile(r'https?://[^\s,"\'>\)\]\|]+')
            links = [u.rstrip('.,;)') for u in url_pattern.findall(raw)]

        summary_lines = [f"提取到 {len(links)} 个链接:"]
        for i, link in enumerate(links[:20]):
            if isinstance(link, dict):
                href = link.get("href") or link.get("url") or ""
                text = link.get("text") or link.get("title") or ""
                summary_lines.append(f"  {i+1}. {text}: {href}")
            else:
                summary_lines.append(f"  {i+1}. {link}")
        if len(links) > 20:
            summary_lines.append(f"  ... (共 {len(links)} 个)")

        return ParsedToolResult(
            tool_name=tool_name,
            status="success",
            summary="\n".join(summary_lines),
            structured_data=links,
            raw=raw,
            item_count=len(links),
        )

    def _parse_file_list(self, tool_name: str, raw: str) -> ParsedToolResult:
        """解析文件列表类工具结果（fs_list_dir、fs_search_files、project_tree 等）。"""
        # 尝试 JSON
        structured = _try_parse_json(raw)
        if structured:
            if isinstance(structured, list):
                items = structured
            else:
                items = None
        else:
            items = None

        # 构建摘要
        if items and isinstance(items, list):
            count = len(items)
            summary = f"共 {count} 个条目:\n"
            summary += "\n".join(f"  {str(item)[:100]}" for item in items[:30])
            if count > 30:
                summary += f"\n  ... (共 {count} 个)"
        else:
            # 直接使用文本
            summary = raw[:2000]

        return ParsedToolResult(
            tool_name=tool_name,
            status="success",
            summary=summary,
            structured_data=items,
            raw=raw,
            truncated=len(raw) > 4000,
            item_count=len(items) if items else None,
        )

    def _parse_file_content(self, tool_name: str, raw: str) -> ParsedToolResult:
        """解析文件内容读取结果。"""
        truncated = len(raw) > 8000
        summary = raw[:4000]
        if truncated:
            summary += f"\n...[文件共 {len(raw)} 字符，已截取前 4000 字]"

        return ParsedToolResult(
            tool_name=tool_name,
            status="success",
            summary=summary,
            structured_data=None,
            raw=raw,
            truncated=truncated,
        )

    def _parse_write_result(self, tool_name: str, raw: str) -> ParsedToolResult:
        """解析文件写入结果。"""
        return ParsedToolResult(
            tool_name=tool_name,
            status="success",
            summary=raw[:500],
            structured_data=None,
            raw=raw,
        )

    def _parse_grep_result(self, tool_name: str, raw: str) -> ParsedToolResult:
        """解析 grep 搜索结果。"""
        lines = raw.splitlines()
        count = len([l for l in lines if l.strip()])
        summary = f"匹配到 {count} 行:\n" + "\n".join(lines[:50])
        if count > 50:
            summary += f"\n... (共 {count} 行)"

        return ParsedToolResult(
            tool_name=tool_name,
            status="success",
            summary=summary,
            structured_data=lines,
            raw=raw,
            truncated=len(lines) > 50,
            item_count=count,
        )

    def _parse_code_result(self, tool_name: str, raw: str) -> ParsedToolResult:
        """解析代码执行/Shell 命令结果。"""
        structured = _try_parse_json(raw)
        
        if structured and isinstance(structured, dict):
            stdout = structured.get("stdout") or structured.get("output") or ""
            stderr = structured.get("stderr") or ""
            return_code = structured.get("return_code") or structured.get("returncode") or 0
            
            status = "success" if return_code == 0 else "error"
            parts = []
            if stdout:
                parts.append(f"输出:\n{stdout[:2000]}")
                if len(stdout) > 2000:
                    parts.append(f"...[输出共 {len(stdout)} 字符]")
            if stderr:
                parts.append(f"错误输出:\n{stderr[:500]}")
            parts.append(f"返回码: {return_code}")
            
            summary = "\n".join(parts)
        else:
            # 纯文本输出
            status = "error" if raw.lower().startswith("error") else "success"
            summary = raw[:3000]
            if len(raw) > 3000:
                summary += f"\n...[共 {len(raw)} 字符]"

        return ParsedToolResult(
            tool_name=tool_name,
            status=status,
            summary=summary,
            structured_data=structured,
            raw=raw,
            truncated=len(raw) > 4000,
        )

    def _parse_git_result(self, tool_name: str, raw: str) -> ParsedToolResult:
        """解析 Git 命令结果。"""
        status = "error" if "error" in raw.lower() or "fatal" in raw.lower() else "success"
        summary = raw[:2000]
        if len(raw) > 2000:
            summary += f"\n...[共 {len(raw)} 字符]"

        return ParsedToolResult(
            tool_name=tool_name,
            status=status,
            summary=summary,
            raw=raw,
            truncated=len(raw) > 2000,
        )

    def _parse_skill_result(self, tool_name: str, raw: str) -> ParsedToolResult:
        """解析 invoke_skill 工具的返回结果，完整保留内容不截断。"""
        return ParsedToolResult(
            tool_name=tool_name,
            status="success",
            summary=raw,
            structured_data=None,
            raw=raw,
            truncated=False,
        )

    def _parse_generic(self, tool_name: str, raw: str) -> ParsedToolResult:
        """通用文本解析（兜底策略）。"""
        # 先尝试 JSON
        structured = _try_parse_json(raw)
        if structured:
            if isinstance(structured, list):
                count = len(structured)
                summary_lines = [f"结果列表 ({count} 条):"]
                for item in structured[:10]:
                    summary_lines.append(f"  - {str(item)[:150]}")
                if count > 10:
                    summary_lines.append(f"  ... (共 {count} 条)")
                summary = "\n".join(summary_lines)
            elif isinstance(structured, dict):
                summary = json.dumps(structured, ensure_ascii=False, indent=2)[:2000]
            else:
                summary = str(structured)[:2000]
        else:
            summary = raw[:2000]
            if len(raw) > 2000:
                summary += f"\n...[共 {len(raw)} 字符]"

        return ParsedToolResult(
            tool_name=tool_name,
            status="success",
            summary=summary,
            structured_data=structured,
            raw=raw,
            truncated=len(raw) > 4000,
        )


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def _try_parse_json(text: str) -> Optional[Any]:
    """尝试将字符串解析为 JSON，失败返回 None。"""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 尝试从文本中提取 JSON
    for pattern in [r'\[.*\]', r'\{.*\}']:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return None


def _extract_search_items(text: str, max_items: int = 20) -> List[Dict[str, str]]:
    """
    从 search_web / search_news 等工具输出的文本中提取结构化搜索条目。

    工具输出的典型格式（每条结果之间用 "---" 分隔）：
        Search results for '...' (via Bing):

        Title: 标题 A
        URL: https://...
        Snippet: 摘要 A

        ---
        Title: 标题 B
        URL: https://...
        Snippet: 摘要 B

    同时兼容 news 格式（Source / Date / Summary 字段）。

    Returns:
        list of dicts，每条包含 title / url / snippet（均可能为空字符串）。
    """
    items: List[Dict[str, str]] = []

    # 按 "---" 切块（去掉首行的 "Search results for ..." 说明行）
    blocks = re.split(r'\n\s*---\s*\n', text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # 解析 key: value 行（key 大小写均可）
        entry: Dict[str, str] = {}
        current_key: Optional[str] = None
        current_value_lines: List[str] = []

        for line in block.splitlines():
            # 尝试匹配 "Key: value" 形式（Key 为已知字段名）
            m = re.match(
                r'^(Title|URL|Snippet|Source|Date|Summary)\s*:\s*(.*)',
                line,
                re.IGNORECASE,
            )
            if m:
                # 保存上一个 key 的累积内容
                if current_key:
                    entry[current_key] = " ".join(current_value_lines).strip()
                current_key = m.group(1).lower()
                # 兼容 source/date → url/snippet 映射
                if current_key == "source":
                    current_key = "source"
                elif current_key == "date":
                    current_key = "date"
                elif current_key == "summary":
                    current_key = "snippet"
                current_value_lines = [m.group(2).strip()]
            elif current_key:
                # 多行值（缩进续行）
                current_value_lines.append(line.strip())

        # 保存最后一个 key
        if current_key and current_value_lines:
            entry[current_key] = " ".join(current_value_lines).strip()

        # 过滤：只收录含有 title 或 url 的块（排除纯说明行）
        if entry.get("title") or entry.get("url"):
            items.append({
                "title": entry.get("title", ""),
                "url": entry.get("url", ""),
                "snippet": entry.get("snippet", ""),
                # 保留 news 专有字段（如有）
                **{k: v for k, v in entry.items() if k in ("source", "date")},
            })

        if len(items) >= max_items:
            break

    return items


def _extract_text_items(text: str, max_items: int = 20) -> List[str]:
    """从文本中提取列表条目（按行分割）。"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # 过滤掉过短的行
    items = [l for l in lines if len(l) > 10]
    return items[:max_items]
