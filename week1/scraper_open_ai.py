"""
Web scraper for JavaScript-rendered sites using Playwright.
Install: pip install playwright && playwright install chromium
"""

from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright



async def fetch_website_contents(url: str, wait_seconds: float = 3.0, max_chars: int = 2000) -> str:
    """Fetch rendered content from a JS-heavy website."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Use 'load' instead of 'networkidle' - more reliable for modern SPAs
        await page.goto(url, wait_until="load", timeout=30000)
        await page.wait_for_timeout(int(wait_seconds * 1000))  # Let JS render
        
        title = await page.title()
        
        await page.evaluate("""
            for (const el of document.querySelectorAll('script, style, nav, footer, header, iframe, noscript')) {
                el.remove();
            }
        """)
        
        text = await page.inner_text("body")
        await browser.close()
        
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    content = f"{title}\n\n" + "\n".join(lines)
    return content[:max_chars]
    

def fetch_website_links(url: str) -> list[str]:
    """Fetch all links from a JS-rendered page."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        
        links = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
        browser.close()
        
    return [link for link in links if link and not link.startswith("javascript:")]


if __name__ == "__main__":
    url = "https://openai.com"
    print(fetch_website_contents(url))