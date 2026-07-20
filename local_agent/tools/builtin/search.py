"""
Search Tools - Web search, Wikipedia, arXiv, URL fetching
"""
from typing import Optional
from local_agent.core.tools import tool


def _resolve_baidu_redirect_with_page(page: "Any", href: str) -> str:
    """
    在已有的 Playwright page 上打开新标签页，访问百度跳转链接，捕获重定向后的真实 URL。

    这是解析 baidu.com/link?url=... 最可靠的方式：在完整浏览器环境（带 Cookie/Session）
    中直接跟随跳转，而不是用裸 HTTP 请求（百度会拒绝无 Cookie 的请求）。

    Args:
        page: 当前百度搜索结果的 Playwright Page 对象（共享浏览器上下文/Cookie）
        href:  baidu.com/link?url=... 格式的跳转 URL

    Returns:
        重定向后的真实 URL；若解析失败则返回空字符串。
    """
    if "baidu.com/link" not in href:
        return href
    try:
        # 在同一个 browser context 中打开新标签页，可以复用百度的 Session Cookie
        new_page = page.context.new_page()
        try:
            # 访问跳转链接，等待页面跳转完成（networkidle 或 load）
            new_page.goto(href, wait_until="load", timeout=10000)
            # 等待跳转稳定后获取最终 URL
            new_page.wait_for_timeout(500)
            final_url = new_page.url
        finally:
            new_page.close()

        # 过滤：只接受非百度域名的真实 URL
        if final_url and final_url.startswith("http") and "baidu.com" not in final_url:
            return final_url
    except Exception:
        pass
    return ""


def _search_via_browser(query: str, max_results: int = 5) -> Optional[str]:
    """
    Try to fetch real search results by visiting search engines with Playwright.
    Attempts Bing → Google → Baidu in order; returns formatted string on success,
    None if all engines fail or Playwright is unavailable.
    Always applies time filters to prioritize the latest/most recent results.

    Key fix: waits for JS-rendered result elements to appear (not just domcontentloaded),
    and tries multiple fallback CSS selectors to handle Bing/Google DOM changes.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None

    import urllib.parse
    encoded = urllib.parse.quote_plus(query)

    # Each entry:
    #   engine_name, url, wait_selector (for JS-render wait), result_selector,
    #   title_selector, link_selector, snippet_selectors
    # Time filter params:
    #   Bing: &filters=ex1%3A"ez5" (past year); returns real destination URLs, no redirect wrappers
    #   Google: tbs=qdr:y (past year)
    #   Baidu: gpc=stf (time sort); kept as last resort because it returns baidu.com/link?url= wrappers
    engines: list[tuple[str, str, str, str, str, str, list[str]]] = [
        (
            "Bing",
            # filters=ex1:"ez5" = past year on Bing; mkt=zh-CN ensures correct regional results
            f"https://www.bing.com/search?q={encoded}&filters=ex1%3A%22ez5%22&mkt=zh-CN",
            # wait_selector: wait until actual search result items appear
            "li.b_algo",
            # result_selector: only actual organic results, not knowledge cards or nav items
            "li.b_algo",
            "h2",
            "a",
            # Avoid bare "p" which matches any paragraph including unrelated content
            ["div.b_caption p", "p.b_lineclamp2", ".b_snippet", ".b_algoheader + div p"],
        ),
        (
            "Google",
            f"https://www.google.com/search?q={encoded}&tbs=qdr:y",
            "div.g, div[data-sokoban-container], .tF2Cxc",
            "div.g",
            "h3",
            "a",
            ["div.VwiC3b", "span.aCOpRe", "div.IsZvec", ".s3v9rd"],
        ),
        (
            "Baidu",
            # NOTE: Baidu returns baidu.com/link?url=... redirect wrappers instead of real URLs.
            # These wrappers cannot be fetched by simple HTTP (JS redirect), so Baidu is kept as
            # last resort only.
            f"https://www.baidu.com/s?wd={encoded}&rn=10&gpc=stf%3D1%2C1%2CSF_Timsort",
            "div.result, .c-container",
            "div.result",
            "h3",
            "a",
            ["div.c-abstract", "span.content-right_8Zs40", ".c-color-text"],
        ),
    ]

    for engine_name, url, wait_sel, result_sel, title_sel, link_sel, snippet_sels in engines:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        # Reduce headless fingerprinting
                        "--disable-blink-features=AutomationControlled",
                        "--disable-infobars",
                    ],
                )
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    locale="zh-CN",
                    viewport={"width": 1280, "height": 800},
                    # Mimic real browser capabilities
                    java_script_enabled=True,
                    accept_downloads=False,
                )
                page = context.new_page()
                # Mask webdriver property to reduce bot detection
                page.add_init_script(
                    "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
                )
                page.goto(url, wait_until="domcontentloaded", timeout=25000)

                # Wait for JS-rendered search result elements to appear (up to 8s)
                # This is critical: search engines render results via JS after DOM load.
                try:
                    page.wait_for_selector(wait_sel, timeout=8000, state="attached")
                except Exception:
                    # If wait_selector times out, still try to scrape what's there
                    page.wait_for_timeout(3000)

                # If CAPTCHA is shown, skip this engine
                body_text = page.inner_text("body")
                if any(kw in body_text for kw in [
                    "solve the challenge", "请解决", "CAPTCHA", "I'm not a robot"
                ]):
                    browser.close()
                    continue

                results: list[str] = []
                items = page.query_selector_all(result_sel)

                # If primary selector yields nothing, try narrower fallback selectors
                # IMPORTANT: avoid ".b_results li" — it matches ALL li including ads,
                # knowledge cards, related-search rows, and pagination, causing irrelevant results.
                if not items and engine_name == "Bing":
                    for fallback_sel in ["li[class*='algo']", "ol#b_results > li.b_algo"]:
                        items = page.query_selector_all(fallback_sel)
                        if items:
                            break
                    # Filter out non-result li items (ads, related searches, pagination, etc.)
                    # by ensuring each item has an <h2> or <h3> title element
                    if items:
                        items = [it for it in items if it.query_selector("h2, h3")]
                elif not items and engine_name == "Google":
                    for fallback_sel in [".tF2Cxc", "div[data-sokoban-container]", "div.rc"]:
                        items = page.query_selector_all(fallback_sel)
                        if items:
                            break

                for item in items[:max_results]:
                    # Title — use text_content() instead of inner_text() because
                    # Bing/Google may visually hide text via CSS (display:none) while
                    # text_content() still returns the underlying DOM text.
                    title_el = item.query_selector(title_sel)
                    title = (title_el.text_content() or "").strip() if title_el else "N/A"

                    # URL — for Bing, the h2>a has the best title link. Decode Bing tracking URLs.
                    link_el = item.query_selector(f"{title_sel} a") or item.query_selector(link_sel)
                    href = link_el.get_attribute("href") if link_el else "N/A"
                    if href:
                        if href.startswith("/"):
                            # Relative URL (Baidu sometimes does this); skip
                            href = "N/A"
                        elif "bing.com/ck/" in href:
                            # Bing tracking URL — decode the embedded real URL from ?u= param
                            import re as _re, base64 as _b64
                            u_match = _re.search(r'[?&]u=([a-zA-Z0-9_-]+)', href)
                            if u_match:
                                b64 = u_match.group(1)
                                if b64.startswith("a1"):
                                    b64 = b64[2:]
                                padded = b64 + "=" * (4 - len(b64) % 4)
                                try:
                                    real = _b64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")
                                    if real.startswith("http"):
                                        href = real
                                except Exception:
                                    pass
                        elif "baidu.com/link" in href:
                            # Baidu redirect wrapper — use in-page resolution
                            href = _resolve_baidu_redirect_with_page(page, href) or "N/A"

                    # Snippet — try multiple selectors in order
                    snippet = ""
                    for sel in snippet_sels:
                        snip_el = item.query_selector(sel)
                        if snip_el:
                            snippet = (snip_el.text_content() or "").strip()
                            if snippet:
                                break

                    if title and title != "N/A":
                        results.append(
                            f"Title: {title}\n"
                            f"URL: {href}\n"
                            f"Snippet: {snippet or 'N/A'}\n"
                        )

                browser.close()

                if results:
                    return (
                        f"Search results for '{query}' (via {engine_name}):\n\n"
                        + "\n---\n".join(results)
                    )
                # No results extracted from this engine — try the next one
        except Exception:
            # This engine failed; try the next one
            continue

    return None


def _is_relevant_result(query: str, result_text: str) -> bool:
    """
    Check if browser search results are actually relevant to the query.

    Problem: Bing China (国内版, mkt=zh-CN) performs "entity recognition" on location
    names like "北京". When the query starts with a known geo-entity, Bing returns a
    knowledge-panel style sitelinks list (e.g. baike.baidu.com/北京市, beijing.gov.cn,
    visitbeijing.com.cn, etc.) instead of searching for the full query string.
    These irrelevant results pass all other checks (non-empty, no baidu redirect URLs),
    causing the DuckDuckGo fallback to be skipped entirely.

    Heuristic: extract 2-character Chinese bigrams from the query (after removing
    known stopword characters and geo-entities), then check that at least one bigram
    appears in the results. If none appear, the results are deemed irrelevant.
    """
    import re as _re

    # Single-character stopwords (particles, prepositions, etc.)
    _STOP_CHARS = set("的了和是在有我他她们也都与或及等中上下一不就要会能可来去个这那")

    # Known geo-entity substrings to strip from query before bigram extraction
    _GEO_ENTITIES = [
        "北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "南京",
        "重庆", "西安", "青岛", "天津", "苏州", "长沙", "中国",
    ]

    # Generic modifiers that alone don't indicate topic relevance
    _GENERIC_TOKENS = {"推荐", "攻略", "大全", "指南", "哪里", "哪家", "什么", "怎么"}

    # Step 1: strip known geo-entities from the query
    stripped_query = query
    for geo in _GEO_ENTITIES:
        stripped_query = stripped_query.replace(geo, "")

    # Step 2: extract all individual Chinese characters (excluding stopword chars)
    chars = [c for c in stripped_query if '\u4e00' <= c <= '\u9fff' and c not in _STOP_CHARS]

    # Step 3: build 2-character bigrams from remaining chars
    bigrams = {"".join(chars[i:i+2]) for i in range(len(chars) - 1)}

    # Step 4: remove generic modifiers
    bigrams -= _GENERIC_TOKENS

    # Step 5: also include ASCII keywords (≥3 chars) from the query
    ascii_tokens = set(_re.findall(r'[a-zA-Z]{3,}', query))
    key_tokens = bigrams | ascii_tokens

    # If no meaningful key tokens remain, skip the relevance check
    if not key_tokens:
        return True

    result_lower = result_text.lower()
    for token in key_tokens:
        if token.lower() in result_lower:
            return True

    return False


def _search_via_baidu(query: str, max_results: int = 5) -> str:
    """
    使用百度搜索引擎获取搜索结果（通过 Playwright）。

    策略：
    - 请求 max_results * 3 条候选结果，以保证过滤后仍有 max_results 条有效结果
    - 对每个 baidu.com/link 跳转 URL，在同一浏览器上下文中打开新标签解析真实 URL
    - 时间限定：百度 gpc 参数限定最近一年
    - 若最终有效结果数不足 max_results，抛出 RuntimeError 触发降级 DuckDuckGo

    成功时返回格式化的搜索结果字符串。
    遇到以下情况时抛出异常（调用方捕获后降级 DuckDuckGo）：
    - Playwright 未安装
    - 页面无法访问
    - CAPTCHA / 反爬机制触发
    - 解析失败（有效结果不足 1 条）

    Args:
        query: 搜索关键词
        max_results: 最终期望的最大结果数（默认 5）

    Returns:
        格式化的搜索结果字符串

    Raises:
        ImportError: Playwright 未安装
        RuntimeError: 无法访问百度或解析失败
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "Playwright is not installed. Install with: pip install playwright && playwright install chromium"
        )

    import urllib.parse
    encoded = urllib.parse.quote_plus(query)
    # 请求 max_results*3 条候选，确保过滤百度跳转链接后仍有足够结果
    fetch_count = max_results * 3
    # gpc 参数：stf=1,1,SF_Timsort 限定时间排序（近期优先）
    # 额外加 tbl=3 强制时间过滤（近一年）
    url = (
        f"https://www.baidu.com/s?wd={encoded}"
        f"&rn={fetch_count}"
        f"&gpc=stf%3D1%2C1%2CSF_Timsort"
        f"&tbl=3"
    )

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
            ],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
            viewport={"width": 1280, "height": 800},
            java_script_enabled=True,
            accept_downloads=False,
        )
        page = context.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        try:
            # 先访问百度首页建立 Session Cookie，避免直接跳搜索页被重定向到 CAPTCHA
            page.goto("https://www.baidu.com", wait_until="domcontentloaded", timeout=15000)
            page.wait_for_timeout(1500)
        except Exception:
            # 首页访问失败不影响后续，继续尝试搜索页
            pass

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=25000)
        except Exception as e:
            browser.close()
            raise RuntimeError(f"无法访问百度搜索页面: {e}")

        # 等待结果渲染
        try:
            page.wait_for_selector("div.result, .c-container", timeout=8000, state="attached")
        except Exception:
            page.wait_for_timeout(2000)

        # 检测 CAPTCHA / 反爬
        body_text = page.inner_text("body")
        if any(kw in body_text for kw in [
            "请输入验证码", "百度安全验证", "verify", "CAPTCHA",
            "人机验证", "安全检验", "请完成验证"
        ]):
            browser.close()
            raise RuntimeError("百度触发反爬验证（CAPTCHA），无法获取结果")

        results: list[str] = []
        items = page.query_selector_all("div.result")

        # fallback selector
        if not items:
            items = page.query_selector_all(".c-container")

        for item in items:
            if len(results) >= max_results:
                break

            # 标题
            title_el = item.query_selector("h3") or item.query_selector("h3.t")
            title = (title_el.text_content() or "").strip() if title_el else ""
            if not title:
                continue

            # URL：优先取 h3 > a
            link_el = item.query_selector("h3 a") or item.query_selector("a")
            href = link_el.get_attribute("href") if link_el else ""
            if not href:
                continue
            if href.startswith("/"):
                continue

            # 解析百度跳转链接为真实 URL
            if "baidu.com/link" in href:
                real_url = _resolve_baidu_redirect_with_page(page, href)
                if not real_url:
                    # 无法解析真实 URL，跳过该条目
                    continue
                href = real_url

            # 最终过滤：只接受以 http 开头且不含 baidu.com 的 URL
            if not (href.startswith("http") and "baidu.com" not in href):
                continue

            # 摘要
            snippet = ""
            for sel in ["div.c-abstract", "span.content-right_8Zs40", ".c-color-text", "div.c-span9"]:
                snip_el = item.query_selector(sel)
                if snip_el:
                    snippet = (snip_el.text_content() or "").strip()
                    if snippet:
                        break

            results.append(
                f"Title: {title}\n"
                f"URL: {href}\n"
                f"Snippet: {snippet or 'N/A'}\n"
            )

        browser.close()

        if not results:
            raise RuntimeError("百度搜索返回了页面，但未能解析到任何结果条目")

        return (
            f"Search results for '{query}' (via Baidu):\n\n"
            + "\n---\n".join(results)
        )


def _search_via_sogou_fallback(query: str, max_results: int = 5) -> "Optional[str]":
    """
    降级搜索：通过 Playwright 使用搜狗（国内可直接访问）获取搜索结果。

    仅在百度搜索失败后调用，作为不依赖外网的备用方案。
    搜狗在中国大陆可正常访问，通过追踪式跳转链接解析真实 URL。

    Returns:
        格式化的搜索结果字符串；若失败返回 None。
    """
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        _logger.warning("_search_via_sogou_fallback: Playwright not installed")
        return None

    import urllib.parse
    encoded = urllib.parse.quote_plus(query)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-infobars",
                ],
            )
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                locale="zh-CN",
                viewport={"width": 1280, "height": 800},
                java_script_enabled=True,
                accept_downloads=False,
            )
            page = context.new_page()
            page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

            # 先访问搜狗首页建立 Cookie
            try:
                page.goto("https://www.sogou.com", wait_until="domcontentloaded", timeout=15000)
                page.wait_for_timeout(800)
            except Exception:
                pass

            url = f"https://www.sogou.com/web?query={encoded}&num={max_results * 3}"
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=25000)
            except Exception as e:
                browser.close()
                _logger.warning("_search_via_sogou_fallback: 无法访问搜狗: %s", e)
                return None

            # 等待结果渲染
            try:
                page.wait_for_selector(".vrwrap", timeout=8000, state="attached")
            except Exception:
                page.wait_for_timeout(2000)

            # 检测是否被封禁 / 跳首页
            if "www.sogou.com/web" not in page.url and "sogou.com" not in page.url:
                browser.close()
                _logger.warning("_search_via_sogou_fallback: 搜狗重定向到非搜索页")
                return None

            items = page.query_selector_all(".vrwrap")
            # fallback selector
            if not items:
                items = page.query_selector_all("div.results div.rb")

            results: list[str] = []
            for item in items:
                if len(results) >= max_results:
                    break

                # 标题
                h3_el = item.query_selector("h3")
                title = (h3_el.text_content() or "").strip() if h3_el else ""
                if not title:
                    continue

                # 链接
                link_el = item.query_selector("h3 a") or item.query_selector("a")
                href = link_el.get_attribute("href") if link_el else ""
                if not href:
                    continue

                # 搜狗使用 /link?url=... 收缨跳转链接，需在浏览器中追踪跳转
                if href.startswith("/link?") or href.startswith("https://www.sogou.com/link?"):
                    full_href = ("https://www.sogou.com" + href) if href.startswith("/") else href
                    new_page = context.new_page()
                    try:
                        new_page.goto(full_href, wait_until="load", timeout=8000)
                        new_page.wait_for_timeout(300)
                        real_url = new_page.url
                    except Exception:
                        real_url = ""
                    finally:
                        new_page.close()

                    if not real_url or not real_url.startswith("http") or "sogou.com" in real_url:
                        continue
                    href = real_url
                elif href.startswith("http") and "sogou.com" not in href:
                    pass  # 已是真实 URL
                else:
                    continue  # 其他格式（相对路径等）跳过

                # 摘要
                snippet = ""
                for sel in [".star-wiki", ".ft", ".space-txt", "p"]:
                    snip_el = item.query_selector(sel)
                    if snip_el:
                        snippet = (snip_el.text_content() or "").strip()
                        if snippet:
                            break

                results.append(
                    f"Title: {title}\n"
                    f"URL: {href}\n"
                    f"Snippet: {snippet or 'N/A'}\n"
                )

            browser.close()

            if not results:
                _logger.warning("_search_via_sogou_fallback: 搜狗未解析到任何结果")
                return None

            return (
                f"Search results for '{query}' (via 搜狗):\n\n"
                + "\n---\n".join(results)
            )
    except Exception as e:
        _logger.warning("_search_via_sogou_fallback: 搜狗搜索异常: %s", e)
        return None


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web and return the latest results.
    Prioritizes Baidu search engine (via Playwright, 国内可访问).
    If Baidu fails, falls back to 搜狗 Sogou (国内可直连，不依赖境外服务).
    Both engines are accessible from mainland China without external network access.
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 5)
    """
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    # ── 优先：百度搜索 ────────────────────────────────────────────────────────
    try:
        baidu_result = _search_via_baidu(query, max_results)
        _logger.info("search_web: Baidu search succeeded for query=%r", query[:60])
        return baidu_result
    except ImportError as e:
        _logger.warning("search_web: Baidu search unavailable (Playwright not installed): %s — degrading to 搜狗", e)
    except RuntimeError as e:
        _logger.warning("search_web: Baidu search failed (%s) — degrading to 搜狗", e)
    except Exception as e:
        _logger.warning("search_web: Baidu search error (%s) — degrading to 搜狗", e)

    # ── 降级：搜狗（国内可访问，不依赖外网）─────────────────────────
    # DuckDuckGo/Bing 均不适合作为内网降级方案。
    # 搜狗在中国大陆可正常访问，通过 Playwright 追踪跳转链接解析真实 URL。
    sogou_result = _search_via_sogou_fallback(query, max_results)
    if sogou_result is not None:
        return sogou_result

    return f"No results found for: {query}"


@tool
def search_news(query: str, max_results: int = 5) -> str:
    """Search for recent news articles using DuckDuckGo News."""
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append(
                    f"Title: {r.get('title', 'N/A')}\n"
                    f"Source: {r.get('source', 'N/A')}\n"
                    f"Date: {r.get('date', 'N/A')}\n"
                    f"URL: {r.get('url', 'N/A')}\n"
                    f"Summary: {r.get('body', 'N/A')}\n"
                )
        if not results:
            return f"No news found for: {query}"
        return f"News results for '{query}':\n\n" + "\n---\n".join(results)
    except Exception as e:
        return f"News search error: {e}"


@tool
def search_wikipedia(query: str, sentences: int = 5) -> str:
    """
    Search Wikipedia for information about a topic.
    Args:
        query: Topic to search for
        sentences: Number of sentences to return from the summary
    """
    try:
        import wikipedia
        try:
            page = wikipedia.page(query, auto_suggest=True)
            summary = wikipedia.summary(query, sentences=sentences, auto_suggest=True)
            return (
                f"Wikipedia: {page.title}\n"
                f"URL: {page.url}\n\n"
                f"Summary:\n{summary}\n\n"
                f"Categories: {', '.join(page.categories[:10])}"
            )
        except wikipedia.exceptions.DisambiguationError as e:
            options = e.options[:10]
            return f"Disambiguation: '{query}' may refer to:\n" + "\n".join(f"  - {o}" for o in options)
        except wikipedia.exceptions.PageError:
            # Try search
            results = wikipedia.search(query, results=5)
            return f"Page not found. Related articles:\n" + "\n".join(f"  - {r}" for r in results)
    except ImportError:
        return "Error: wikipedia not installed. Run: pip install wikipedia"
    except Exception as e:
        return f"Wikipedia error: {e}"


@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for academic papers.
    Args:
        query: Search query for academic papers
        max_results: Maximum number of papers to return
    """
    try:
        import arxiv
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            authors = ", ".join(a.name for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += f" et al."
            results.append(
                f"Title: {paper.title}\n"
                f"Authors: {authors}\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"ArXiv ID: {paper.entry_id}\n"
                f"Abstract: {paper.summary[:300]}...\n"
                f"PDF: {paper.pdf_url}"
            )
        if not results:
            return f"No papers found for: {query}"
        return f"arXiv results for '{query}':\n\n" + "\n---\n".join(results)
    except ImportError:
        return "Error: arxiv not installed. Run: pip install arxiv"
    except Exception as e:
        return f"arXiv search error: {e}"


@tool
def fetch_url(url: str, extract_text: bool = True, **kwargs) -> str:
    """
    Fetch content from a URL and return the text.
    Args:
        url: URL to fetch
        extract_text: If True, extracts clean text; if False, returns raw HTML
    """
    # Silently ignore unexpected keyword arguments (e.g., access_strategy passed by mistake)
    if kwargs:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "fetch_url: ignoring unexpected keyword arguments: %s", list(kwargs.keys())
        )
    # Special handling for Baidu redirect links (baidu.com/link?url=...)
    # These use JS redirects that httpx cannot follow; use Playwright if available.
    if "baidu.com/link" in url:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"],
                )
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    locale="zh-CN",
                )
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=20000)
                page.wait_for_timeout(2000)
                final_url = page.url
                if extract_text:
                    content = page.inner_text("body") or ""
                    lines = [l.strip() for l in content.splitlines() if l.strip()]
                    text = "\n".join(lines)
                    browser.close()
                    return f"Content from {final_url} (redirected from Baidu):\n\n{text[:8000]}"
                else:
                    html = page.content()
                    browser.close()
                    return f"HTML from {final_url} (redirected from Baidu):\n\n{html[:8000]}"
        except ImportError:
            return (
                f"[Tool Error] Cannot follow Baidu redirect URL '{url}': "
                "Playwright is not installed. Install with: pip install playwright && playwright install chromium. "
                "Please use a non-Baidu URL instead."
            )
        except Exception as e:
            return f"[Tool Error] Failed to follow Baidu redirect '{url}': {e}"

    try:
        import httpx
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
        }
        response = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
        response.raise_for_status()

        if extract_text:
            soup = BeautifulSoup(response.text, "html.parser")
            # Remove scripts and styles
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Clean up excessive newlines
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            text = "\n".join(lines)
            return f"Content from {url}:\n\n{text[:8000]}"
        else:
            return f"HTML from {url}:\n\n{response.text[:8000]}"
    except ImportError:
        return "Error: httpx or beautifulsoup4 not installed"
    except Exception as e:
        return f"Error fetching URL: {e}"


search_web.metadata = search_web.metadata or {}
search_web.metadata["category"] = "search"
search_news.metadata = search_news.metadata or {}
search_news.metadata["category"] = "search"
search_wikipedia.metadata = search_wikipedia.metadata or {}
search_wikipedia.metadata["category"] = "search"
search_arxiv.metadata = search_arxiv.metadata or {}
search_arxiv.metadata["category"] = "search"
fetch_url.metadata = fetch_url.metadata or {}
fetch_url.metadata["category"] = "search"

TOOLS = [search_web, search_news, search_wikipedia, search_arxiv, fetch_url]
