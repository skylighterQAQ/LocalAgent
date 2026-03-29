"""
Example custom tool: Browser
Place this file at tools/browser/tool.py to activate it.
Then add "browser" to the tools list in config/config.yaml.
"""
from typing import Any, List, Optional, Type
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from core.tool_base import LocalAgentTool


class FetchPageInput(BaseModel):
    url: str = Field(description="The URL to fetch")
    extract_text_only: bool = Field(default=True, description="If True, return plain text; if False, return HTML")
    max_chars: int = Field(default=4000, description="Maximum characters to return")


class ClickableLinkInput(BaseModel):
    url: str = Field(description="The URL to extract links from")


class _FetchPageTool(BaseTool):
    name: str = "browser_fetch_page"
    description: str = (
        "Fetch and read the content of a web page. "
        "Returns the text content or HTML of the page. "
        "Useful for reading documentation, articles, and web content."
    )
    args_schema: Type[BaseModel] = FetchPageInput

    def _run(self, url: str, extract_text_only: bool = True, max_chars: int = 4000) -> str:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; OpenClaw/1.0; +https://github.com/openclaw)"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            if extract_text_only:
                soup = BeautifulSoup(response.text, "html.parser")
                # Remove scripts, styles, nav elements
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                # Clean up blank lines
                lines = [l for l in text.splitlines() if l.strip()]
                text = "\n".join(lines)
                return text[:max_chars]
            else:
                return response.text[:max_chars]
        except requests.RequestException as e:
            return f"Error fetching {url}: {e}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class _ExtractLinksTool(BaseTool):
    name: str = "browser_extract_links"
    description: str = (
        "Extract all hyperlinks from a web page. "
        "Returns a list of URLs and their link text. "
        "Useful for discovering resources on a page."
    )
    args_schema: Type[BaseModel] = ClickableLinkInput

    def _run(self, url: str) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (compatible; OpenClaw/1.0)"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                text = a.get_text(strip=True) or "(no text)"
                # Make relative links absolute
                if href.startswith("/"):
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    href = f"{parsed.scheme}://{parsed.netloc}{href}"
                if href.startswith("http"):
                    links.append(f"[{text}] {href}")

            return "\n".join(links[:50]) if links else "No links found."
        except requests.RequestException as e:
            return f"Error: {e}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class BrowserTool(LocalAgentTool):
    name = "browser"
    description = "Web browsing - fetch pages and extract links"
    version = "1.0.0"

    def get_tools(self) -> List[BaseTool]:
        return [_FetchPageTool(), _ExtractLinksTool()]
