"""
Page module for Allyson.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Union

from playwright.async_api import Page as AsyncPage
from playwright.sync_api import Page as SyncPage

from allyson.element import Element

logger = logging.getLogger(__name__)


class Page:
    """
    Page class for Allyson.
    Provides a simplified interface to Playwright page automation.
    Supports both synchronous and asynchronous usage.
    """

    def __init__(self, page: Union[SyncPage, AsyncPage], is_async: bool = False):
        """
        Initialize a new Page instance.

        Args:
            page: Playwright page object
            is_async: Whether the page is async
        """
        self._page = page
        self._is_async = is_async

    def goto(self, url: str, wait_until: str = "load", timeout: Optional[int] = None):
        """
        Navigate to the specified URL.

        Args:
            url: URL to navigate to
            wait_until: When to consider navigation succeeded ('load', 'domcontentloaded', 'networkidle')
            timeout: Maximum navigation time in milliseconds
        """
        if self._is_async:
            raise RuntimeError("Use 'await page.agoto()' in async mode")
        
        options = {}
        if wait_until:
            options["wait_until"] = wait_until
        if timeout:
            options["timeout"] = timeout
            
        response = self._page.goto(url, **options)
        return response

    async def agoto(self, url: str, wait_until: str = "load", timeout: Optional[int] = None):
        """
        Navigate to the specified URL (async version).

        Args:
            url: URL to navigate to
            wait_until: When to consider navigation succeeded ('load', 'domcontentloaded', 'networkidle')
            timeout: Maximum navigation time in milliseconds
        """
        if not self._is_async:
            raise RuntimeError("Use 'page.goto()' in sync mode")
            
        options = {}
        if wait_until:
            options["wait_until"] = wait_until
        if timeout:
            options["timeout"] = timeout
            
        response = await self._page.goto(url, **options)
        return response

    def click(self, selector: str, timeout: Optional[int] = None, force: bool = False):
        """
        Click on an element.

        Args:
            selector: Element selector (CSS, XPath, or text content)
            timeout: Maximum time to wait for the element in milliseconds
            force: Whether to bypass actionability checks
        """
        if self._is_async:
            raise RuntimeError("Use 'await page.aclick()' in async mode")
            
        options = {"force": force}
        if timeout:
            options["timeout"] = timeout
            
        # Try to find the element by text if selector doesn't look like CSS or XPath
        if not selector.startswith(("//", ".", "#", "[")) and " " in selector:
            try:
                self._page.click(f"text={selector}", **options)
                return
            except Exception as e:
                logger.debug(f"Failed to click by text: {e}")
                
        # Try regular selector
        self._page.click(selector, **options)

    async def aclick(self, selector: str, timeout: Optional[int] = None, force: bool = False):
        """
        Click on an element (async version).

        Args:
            selector: Element selector (CSS, XPath, or text content)
            timeout: Maximum time to wait for the element in milliseconds
            force: Whether to bypass actionability checks
        """
        if not self._is_async:
            raise RuntimeError("Use 'page.click()' in sync mode")
            
        options = {"force": force}
        if timeout:
            options["timeout"] = timeout
            
        # Try to find the element by text if selector doesn't look like CSS or XPath
        if not selector.startswith(("//", ".", "#", "[")) and " " in selector:
            try:
                await self._page.click(f"text={selector}", **options)
                return
            except Exception as e:
                logger.debug(f"Failed to click by text: {e}")
                
        # Try regular selector
        await self._page.click(selector, **options)

    def fill(self, selector: str, value: str, timeout: Optional[int] = None):
        """
        Fill an input field.

        Args:
            selector: Element selector (CSS, XPath, or label text)
            value: Value to fill
            timeout: Maximum time to wait for the element in milliseconds
        """
        if self._is_async:
            raise RuntimeError("Use 'await page.afill()' in async mode")
            
        options = {}
        if timeout:
            options["timeout"] = timeout
            
        # Try to find the element by label text if selector doesn't look like CSS or XPath
        if not selector.startswith(("//", ".", "#", "[")) and " " in selector:
            try:
                self._page.fill(f"[placeholder='{selector}']", value, **options)
                return
            except Exception:
                try:
                    self._page.fill(f"text={selector} >> xpath=../input", value, **options)
                    return
                except Exception as e:
                    logger.debug(f"Failed to fill by label text: {e}")
                
        # Try regular selector
        self._page.fill(selector, value, **options)

    async def afill(self, selector: str, value: str, timeout: Optional[int] = None):
        """
        Fill an input field (async version).

        Args:
            selector: Element selector (CSS, XPath, or label text)
            value: Value to fill
            timeout: Maximum time to wait for the element in milliseconds
        """
        if not self._is_async:
            raise RuntimeError("Use 'page.fill()' in sync mode")
            
        options = {}
        if timeout:
            options["timeout"] = timeout
            
        # Try to find the element by label text if selector doesn't look like CSS or XPath
        if not selector.startswith(("//", ".", "#", "[")) and " " in selector:
            try:
                await self._page.fill(f"[placeholder='{selector}']", value, **options)
                return
            except Exception:
                try:
                    await self._page.fill(f"text={selector} >> xpath=../input", value, **options)
                    return
                except Exception as e:
                    logger.debug(f"Failed to fill by label text: {e}")
                
        # Try regular selector
        await self._page.fill(selector, value, **options)

    def wait_for_selector(self, selector: str, timeout: Optional[int] = None, state: str = "visible"):
        """
        Wait for an element to be present.

        Args:
            selector: Element selector
            timeout: Maximum time to wait in milliseconds
            state: State to wait for ('attached', 'detached', 'visible', 'hidden')
        """
        if self._is_async:
            raise RuntimeError("Use 'await page.await_for_selector()' in async mode")
            
        options = {"state": state}
        if timeout:
            options["timeout"] = timeout
            
        element = self._page.wait_for_selector(selector, **options)
        return Element(element) if element else None

    async def await_for_selector(self, selector: str, timeout: Optional[int] = None, state: str = "visible"):
        """
        Wait for an element to be present (async version).

        Args:
            selector: Element selector
            timeout: Maximum time to wait in milliseconds
            state: State to wait for ('attached', 'detached', 'visible', 'hidden')
        """
        if not self._is_async:
            raise RuntimeError("Use 'page.wait_for_selector()' in sync mode")
            
        options = {"state": state}
        if timeout:
            options["timeout"] = timeout
            
        element = await self._page.wait_for_selector(selector, **options)
        return Element(element, is_async=True) if element else None

    def screenshot(self, path: Optional[str] = None, full_page: bool = False):
        """
        Take a screenshot of the page.

        Args:
            path: Path to save the screenshot
            full_page: Whether to take a screenshot of the full page
        """
        if self._is_async:
            raise RuntimeError("Use 'await page.ascreenshot()' in async mode")
            
        options = {"full_page": full_page}
        if path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            options["path"] = path
            
        return self._page.screenshot(**options)

    async def ascreenshot(self, path: Optional[str] = None, full_page: bool = False):
        """
        Take a screenshot of the page (async version).

        Args:
            path: Path to save the screenshot
            full_page: Whether to take a screenshot of the full page
        """
        if not self._is_async:
            raise RuntimeError("Use 'page.screenshot()' in sync mode")
            
        options = {"full_page": full_page}
        if path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            options["path"] = path
            
        return await self._page.screenshot(**options)

    def evaluate(self, expression: str, arg: Any = None):
        """
        Evaluate JavaScript in the page context.

        Args:
            expression: JavaScript expression to evaluate
            arg: Argument to pass to the expression
        """
        if self._is_async:
            raise RuntimeError("Use 'await page.aevaluate()' in async mode")
            
        return self._page.evaluate(expression, arg)

    async def aevaluate(self, expression: str, arg: Any = None):
        """
        Evaluate JavaScript in the page context (async version).

        Args:
            expression: JavaScript expression to evaluate
            arg: Argument to pass to the expression
        """
        if not self._is_async:
            raise RuntimeError("Use 'page.evaluate()' in sync mode")
            
        return await self._page.evaluate(expression, arg)

    def get_text(self, selector: str, timeout: Optional[int] = None):
        """
        Get the text content of an element.

        Args:
            selector: Element selector
            timeout: Maximum time to wait for the element in milliseconds
        """
        if self._is_async:
            raise RuntimeError("Use 'await page.aget_text()' in async mode")
            
        options = {}
        if timeout:
            options["timeout"] = timeout
            
        return self._page.text_content(selector, **options)

    async def aget_text(self, selector: str, timeout: Optional[int] = None):
        """
        Get the text content of an element (async version).

        Args:
            selector: Element selector
            timeout: Maximum time to wait for the element in milliseconds
        """
        if not self._is_async:
            raise RuntimeError("Use 'page.get_text()' in sync mode")
            
        options = {}
        if timeout:
            options["timeout"] = timeout
            
        return await self._page.text_content(selector, **options)

    def wait_for_load_state(self, state: str = "load", timeout: Optional[int] = None):
        """
        Wait for the page to reach a specific load state.

        Args:
            state: Load state to wait for ('load', 'domcontentloaded', 'networkidle')
            timeout: Maximum time to wait in milliseconds
        """
        if self._is_async:
            raise RuntimeError("Use 'await page.await_for_load_state()' in async mode")
            
        options = {}
        if timeout:
            options["timeout"] = timeout
            
        self._page.wait_for_load_state(state, **options)

    async def await_for_load_state(self, state: str = "load", timeout: Optional[int] = None):
        """
        Wait for the page to reach a specific load state (async version).

        Args:
            state: Load state to wait for ('load', 'domcontentloaded', 'networkidle')
            timeout: Maximum time to wait in milliseconds
        """
        if not self._is_async:
            raise RuntimeError("Use 'page.wait_for_load_state()' in sync mode")
            
        options = {}
        if timeout:
            options["timeout"] = timeout
            
        await self._page.wait_for_load_state(state, **options)

    def get_url(self):
        """Get the current URL of the page."""
        return self._page.url

    def get_title(self):
        """Get the current title of the page."""
        return self._page.title() 