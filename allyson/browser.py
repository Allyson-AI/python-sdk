"""
Browser module for Allyson.
"""

import asyncio
import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Optional, Union

from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright

from allyson.page import Page

logger = logging.getLogger(__name__)


class Browser:
    """
    Browser class for Allyson.
    Provides a simplified interface to Playwright browser automation.
    Supports both synchronous and asynchronous usage.
    """

    def __init__(
        self,
        browser_type: str = "chromium",
        headless: bool = True,
        slow_mo: int = 0,
        viewport: Dict[str, int] = None,
        user_agent: Optional[str] = None,
        locale: Optional[str] = None,
        timeout: int = 30000,
        proxy: Optional[Dict[str, str]] = None,
        executable_path: Optional[str] = None,
    ):
        """
        Initialize a new Browser instance.

        Args:
            browser_type: Type of browser to use ('chromium', 'firefox', or 'webkit')
            headless: Whether to run browser in headless mode
            slow_mo: Slow down operations by the specified amount of milliseconds
            viewport: Viewport dimensions, e.g., {'width': 1280, 'height': 720}
            user_agent: User agent string
            locale: Browser locale
            timeout: Default timeout for operations in milliseconds
            proxy: Proxy settings, e.g., {'server': 'http://myproxy.com:3128'}
            executable_path: Path to browser executable (e.g., Chrome binary)
        """
        self.browser_type = browser_type
        self.headless = headless
        self.slow_mo = slow_mo
        self.viewport = viewport or {"width": 1280, "height": 720}
        self.user_agent = user_agent
        self.locale = locale
        self.timeout = timeout
        self.proxy = proxy
        self.executable_path = executable_path

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._is_async = False

    def __enter__(self):
        """Context manager for synchronous usage."""
        self._launch_sync()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources for synchronous usage."""
        self.close()

    async def __aenter__(self):
        """Context manager for asynchronous usage."""
        await self._launch_async()
        self._is_async = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources for asynchronous usage."""
        await self.aclose()

    def _launch_sync(self):
        """Launch the browser in synchronous mode."""
        self._playwright = sync_playwright().start()
        
        # Get the browser type
        if self.browser_type == "chromium":
            browser_type = self._playwright.chromium
        elif self.browser_type == "firefox":
            browser_type = self._playwright.firefox
        elif self.browser_type == "webkit":
            browser_type = self._playwright.webkit
        else:
            raise ValueError(f"Unsupported browser type: {self.browser_type}")
        
        # Launch the browser
        launch_options = {
            "headless": self.headless,
            "slow_mo": self.slow_mo,
        }
        
        # Add proxy if specified
        if self.proxy:
            launch_options["proxy"] = self.proxy
            
        # Add executable path if specified
        if self.executable_path:
            launch_options["executable_path"] = self.executable_path
        
        self._browser = browser_type.launch(**launch_options)
        
        # Create a new page
        context_options = {}
        if self.viewport:
            context_options["viewport"] = self.viewport
        if self.user_agent:
            context_options["user_agent"] = self.user_agent
        if self.locale:
            context_options["locale"] = self.locale
            
        self._context = self._browser.new_context(**context_options)
        self._page = Page(self._context.new_page(), is_async=False)

    async def _launch_async(self):
        """Launch the browser in asynchronous mode."""
        self._playwright = await async_playwright().start()
        
        # Get the browser type
        if self.browser_type == "chromium":
            browser_type = self._playwright.chromium
        elif self.browser_type == "firefox":
            browser_type = self._playwright.firefox
        elif self.browser_type == "webkit":
            browser_type = self._playwright.webkit
        else:
            raise ValueError(f"Unsupported browser type: {self.browser_type}")
        
        # Launch the browser
        launch_options = {
            "headless": self.headless,
            "slow_mo": self.slow_mo,
        }
        
        # Add proxy if specified
        if self.proxy:
            launch_options["proxy"] = self.proxy
            
        # Add executable path if specified
        if self.executable_path:
            launch_options["executable_path"] = self.executable_path
        
        self._browser = await browser_type.launch(**launch_options)
        
        # Create a new page
        context_options = {}
        if self.viewport:
            context_options["viewport"] = self.viewport
        if self.user_agent:
            context_options["user_agent"] = self.user_agent
        if self.locale:
            context_options["locale"] = self.locale
            
        self._context = await self._browser.new_context(**context_options)
        self._page = Page(await self._context.new_page(), is_async=True)

    def goto(self, url: str, wait_until: str = "load", timeout: Optional[int] = None):
        """
        Navigate to the specified URL.

        Args:
            url: URL to navigate to
            wait_until: When to consider navigation succeeded ('load', 'domcontentloaded', 'networkidle')
            timeout: Maximum navigation time in milliseconds
        """
        if self._is_async:
            raise RuntimeError("Use 'await browser.goto()' in async mode")
        return self._page.goto(url, wait_until, timeout)

    async def agoto(self, url: str, wait_until: str = "load", timeout: Optional[int] = None):
        """
        Navigate to the specified URL (async version).

        Args:
            url: URL to navigate to
            wait_until: When to consider navigation succeeded ('load', 'domcontentloaded', 'networkidle')
            timeout: Maximum navigation time in milliseconds
        """
        if not self._is_async:
            raise RuntimeError("Use 'browser.goto()' in sync mode")
        return await self._page.agoto(url, wait_until, timeout)

    def new_page(self):
        """Create a new page in the browser context."""
        if self._is_async:
            raise RuntimeError("Use 'await browser.new_page()' in async mode")
        return Page(self._context.new_page())

    async def anew_page(self):
        """Create a new page in the browser context (async version)."""
        if not self._is_async:
            raise RuntimeError("Use 'browser.new_page()' in sync mode")
        page = await self._context.new_page()
        return Page(page, is_async=True)

    def close(self):
        """Close the browser and clean up resources."""
        if self._is_async:
            raise RuntimeError("Use 'await browser.aclose()' in async mode")
        
        if self._context:
            self._context.close()
            self._context = None
        
        if self._browser:
            self._browser.close()
            self._browser = None
        
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    async def aclose(self):
        """Close the browser and clean up resources (async version)."""
        if not self._is_async:
            raise RuntimeError("Use 'browser.close()' in sync mode")
        
        if self._context:
            await self._context.close()
            self._context = None
        
        if self._browser:
            await self._browser.close()
            self._browser = None
        
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    # Delegate methods to the page
    def __getattr__(self, name):
        """Delegate method calls to the page object."""
        if self._page is None:
            raise RuntimeError("Browser not initialized. Use with 'with' statement or call launch() first.")
        
        attr = getattr(self._page, name)
        return attr 