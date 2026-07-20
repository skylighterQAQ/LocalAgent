"""
Browser Tools - Playwright-based browser automation
"""
from pathlib import Path
import tempfile
from typing import Optional
from local_agent.core.tools import tool

# Global browser state
_browser_state = {
    "playwright": None,
    "browser": None,
    "page": None,
}


def _get_page():
    """Get or create a browser page (lazy initialization)"""
    import asyncio

    async def _init():
        from playwright.async_api import async_playwright
        if _browser_state["playwright"] is None:
            _browser_state["playwright"] = await async_playwright().start()
            _browser_state["browser"] = await _browser_state["playwright"].chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            _browser_state["page"] = await _browser_state["browser"].new_page()
            # Set realistic viewport and user agent
            await _browser_state["page"].set_viewport_size({"width": 1280, "height": 720})
            await _browser_state["page"].set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
        return _browser_state["page"]

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_init())
    finally:
        loop.close()


def _run_async(coro):
    """Run an async coroutine synchronously"""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@tool
def browser_navigate(url: str) -> str:
    """
    Navigate the browser to a URL.
    Args:
        url: URL to navigate to
    """
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            title = page.title()
            current_url = page.url
            browser.close()
        return f"Navigated to: {current_url}\nPage title: {title}"
    except ImportError:
        return "Error: playwright not installed. Run: pip install playwright && playwright install chromium"
    except Exception as e:
        return f"Navigation error: {e}"


@tool
def browser_get_text(url: str, selector: str = "body") -> str:
    """
    Get text content from a webpage.
    Args:
        url: URL to fetch
        selector: CSS selector to extract text from (default: body)
    """
    try:
        from playwright.sync_api import sync_playwright
        from bs4 import BeautifulSoup

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            # Wait for content to load
            page.wait_for_timeout(2000)

            if selector == "body":
                content = page.content()
                browser.close()
                soup = BeautifulSoup(content, "html.parser")
                for tag in soup(["script", "style", "nav", "footer"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
                lines = [l for l in text.splitlines() if l.strip()]
                return "\n".join(lines[:200])
            else:
                element = page.query_selector(selector)
                text = element.inner_text() if element else f"Element '{selector}' not found"
                browser.close()
                return text
    except ImportError:
        return "Error: playwright not installed. Run: pip install playwright && playwright install chromium"
    except Exception as e:
        return f"Error getting page text: {e}"


@tool
def browser_screenshot(url: str, output_path: Optional[str] = None) -> str:
    """
    Take a screenshot of a webpage.
    Args:
        url: URL to screenshot
        output_path: Path to save the screenshot. Defaults to the system
            temporary directory when omitted.
    """
    try:
        output_path = output_path or str(Path(tempfile.gettempdir()) / "screenshot.png")
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.set_viewport_size({"width": 1280, "height": 720})
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2000)
            page.screenshot(path=output_path, full_page=False)
            browser.close()
        return f"Screenshot saved to: {output_path}"
    except ImportError:
        return "Error: playwright not installed."
    except Exception as e:
        return f"Error taking screenshot: {e}"


@tool
def browser_click_and_get(url: str, selector: str) -> str:
    """
    Navigate to URL, click an element, and return the resulting page content.
    Args:
        url: Starting URL
        selector: CSS selector of element to click
    """
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(1000)

            element = page.query_selector(selector)
            if not element:
                browser.close()
                return f"Element '{selector}' not found on page"

            element.click()
            page.wait_for_timeout(2000)
            new_url = page.url
            title = page.title()
            browser.close()
        return f"Clicked '{selector}'\nNew URL: {new_url}\nPage title: {title}"
    except Exception as e:
        return f"Error: {e}"


@tool
def browser_fill_form(url: str, form_data: str) -> str:
    """
    Fill a form on a webpage.
    Args:
        url: URL with the form
        form_data: JSON string with selector:value pairs, e.g. '{"#username": "user", "#password": "pass"}'
    """
    import json
    try:
        from playwright.sync_api import sync_playwright
        data = json.loads(form_data)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)

            filled = []
            for selector, value in data.items():
                try:
                    page.fill(selector, str(value))
                    filled.append(f"  Filled {selector}: {value}")
                except Exception as e:
                    filled.append(f"  Failed {selector}: {e}")

            browser.close()
        return f"Form filling results:\n" + "\n".join(filled)
    except json.JSONDecodeError:
        return "Error: form_data must be a valid JSON string"
    except Exception as e:
        return f"Error filling form: {e}"


@tool
def browser_extract_links(url: str) -> str:
    """Extract all links from a webpage."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)

            links = page.eval_on_selector_all(
                "a[href]",
                "elements => elements.map(e => ({href: e.href, text: e.innerText.trim()}))"
            )
            browser.close()

        if not links:
            return f"No links found on {url}"

        lines = [f"Links on {url} ({len(links)} found):"]
        for link in links[:30]:
            text = link.get("text", "")[:50]
            href = link.get("href", "")
            lines.append(f"  [{text}] {href}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error extracting links: {e}"


@tool
def browser_search_and_get(query: str, engine: str = "google") -> str:
    """
    Search using a browser and return first page results.
    Args:
        query: What to search for
        engine: Search engine to use ('google' or 'bing')
    """
    if engine == "google":
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
    else:
        url = f"https://www.bing.com/search?q={query.replace(' ', '+')}"

    return browser_get_text.run(url=url, selector="body")


browser_navigate.metadata = browser_navigate.metadata or {}
browser_navigate.metadata["category"] = "browser"
browser_get_text.metadata = browser_get_text.metadata or {}
browser_get_text.metadata["category"] = "browser"
browser_screenshot.metadata = browser_screenshot.metadata or {}
browser_screenshot.metadata["category"] = "browser"
browser_click_and_get.metadata = browser_click_and_get.metadata or {}
browser_click_and_get.metadata["category"] = "browser"
browser_fill_form.metadata = browser_fill_form.metadata or {}
browser_fill_form.metadata["category"] = "browser"
browser_extract_links.metadata = browser_extract_links.metadata or {}
browser_extract_links.metadata["category"] = "browser"
browser_search_and_get.metadata = browser_search_and_get.metadata or {}
browser_search_and_get.metadata["category"] = "browser"

TOOLS = [
    browser_navigate,
    browser_get_text,
    browser_screenshot,
    browser_click_and_get,
    browser_fill_form,
    browser_extract_links,
    browser_search_and_get,
]
