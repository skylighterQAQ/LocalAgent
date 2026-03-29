"""
Example custom tool: Search
Place this file at tools/search/tool.py to activate it.
Then add "search" to the tools list in config/config.yaml.
"""
from typing import Any, List, Type
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from core.tool_base import LocalAgentTool


class SearchInput(BaseModel):
    query: str = Field(description="The search query")
    num_results: int = Field(default=5, description="Number of results to return (1-10)")


class _DuckDuckGoSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web using DuckDuckGo. "
        "Returns a list of search results with titles, URLs, and snippets. "
        "Use this to find current information, articles, and resources online."
    )
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; OpenClaw/1.0)"
            }
            # DuckDuckGo HTML search endpoint
            params = {"q": query, "kl": "us-en"}
            response = requests.get(
                "https://html.duckduckgo.com/html/",
                params=params,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []

            for result in soup.select(".result__body")[:num_results]:
                title_tag = result.select_one(".result__title")
                link_tag = result.select_one(".result__url")
                snippet_tag = result.select_one(".result__snippet")

                title = title_tag.get_text(strip=True) if title_tag else "No title"
                link = link_tag.get_text(strip=True) if link_tag else ""
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

                if not link.startswith("http"):
                    link = "https://" + link

                results.append(f"**{title}**\nURL: {link}\n{snippet}")

            if not results:
                return f"No results found for: {query}"

            return "\n\n".join(results)

        except Exception as e:
            return f"Search error: {e}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class WebSearchTool(LocalAgentTool):
    name = "web_search"
    description = "Search the web using DuckDuckGo (no API key required)"
    version = "1.0.0"

    def get_tools(self) -> List[BaseTool]:
        return [_DuckDuckGoSearchTool()]
