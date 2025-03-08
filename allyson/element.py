"""
Element module for Allyson.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from playwright.async_api import ElementHandle as AsyncElementHandle
from playwright.sync_api import ElementHandle as SyncElementHandle

logger = logging.getLogger(__name__)


class Element:
    """
    Element class for Allyson.
    Provides a simplified interface to Playwright element automation.
    Supports both synchronous and asynchronous usage.
    """

    def __init__(self, element: Union[SyncElementHandle, AsyncElementHandle], is_async: bool = False):
        """
        Initialize a new Element instance.

        Args:
            element: Playwright element handle
            is_async: Whether the element is async
        """
        self._element = element
        self._is_async = is_async

    def click(self, force: bool = False, timeout: Optional[int] = None):
        """
        Click on the element.

        Args:
            force: Whether to bypass actionability checks
            timeout: Maximum time to wait in milliseconds
        """
        if self._is_async:
            raise RuntimeError("Use 'await element.aclick()' in async mode")
            
        options = {"force": force}
        if timeout:
            options["timeout"] = timeout
            
        self._element.click(**options)

    async def aclick(self, force: bool = False, timeout: Optional[int] = None):
        """
        Click on the element (async version).

        Args:
            force: Whether to bypass actionability checks
            timeout: Maximum time to wait in milliseconds
        """
        if not self._is_async:
            raise RuntimeError("Use 'element.click()' in sync mode")
            
        options = {"force": force}
        if timeout:
            options["timeout"] = timeout
            
        await self._element.click(**options)

    def fill(self, value: str, timeout: Optional[int] = None):
        """
        Fill the element with text.

        Args:
            value: Value to fill
            timeout: Maximum time to wait in milliseconds
        """
        if self._is_async:
            raise RuntimeError("Use 'await element.afill()' in async mode")
            
        options = {}
        if timeout:
            options["timeout"] = timeout
            
        self._element.fill(value, **options)

    async def afill(self, value: str, timeout: Optional[int] = None):
        """
        Fill the element with text (async version).

        Args:
            value: Value to fill
            timeout: Maximum time to wait in milliseconds
        """
        if not self._is_async:
            raise RuntimeError("Use 'element.fill()' in sync mode")
            
        options = {}
        if timeout:
            options["timeout"] = timeout
            
        await self._element.fill(value, **options)

    def get_text(self):
        """Get the text content of the element."""
        if self._is_async:
            raise RuntimeError("Use 'await element.aget_text()' in async mode")
            
        return self._element.text_content()

    async def aget_text(self):
        """Get the text content of the element (async version)."""
        if not self._is_async:
            raise RuntimeError("Use 'element.get_text()' in sync mode")
            
        return await self._element.text_content()

    def get_attribute(self, name: str):
        """
        Get an attribute of the element.

        Args:
            name: Attribute name
        """
        if self._is_async:
            raise RuntimeError("Use 'await element.aget_attribute()' in async mode")
            
        return self._element.get_attribute(name)

    async def aget_attribute(self, name: str):
        """
        Get an attribute of the element (async version).

        Args:
            name: Attribute name
        """
        if not self._is_async:
            raise RuntimeError("Use 'element.get_attribute()' in sync mode")
            
        return await self._element.get_attribute(name)

    def is_visible(self):
        """Check if the element is visible."""
        if self._is_async:
            raise RuntimeError("Use 'await element.ais_visible()' in async mode")
            
        return self._element.is_visible()

    async def ais_visible(self):
        """Check if the element is visible (async version)."""
        if not self._is_async:
            raise RuntimeError("Use 'element.is_visible()' in sync mode")
            
        return await self._element.is_visible()

    def is_enabled(self):
        """Check if the element is enabled."""
        if self._is_async:
            raise RuntimeError("Use 'await element.ais_enabled()' in async mode")
            
        return self._element.is_enabled()

    async def ais_enabled(self):
        """Check if the element is enabled (async version)."""
        if not self._is_async:
            raise RuntimeError("Use 'element.is_enabled()' in sync mode")
            
        return await self._element.is_enabled()

    def screenshot(self, path: Optional[str] = None):
        """
        Take a screenshot of the element.

        Args:
            path: Path to save the screenshot
        """
        if self._is_async:
            raise RuntimeError("Use 'await element.ascreenshot()' in async mode")
            
        options = {}
        if path:
            options["path"] = path
            
        return self._element.screenshot(**options)

    async def ascreenshot(self, path: Optional[str] = None):
        """
        Take a screenshot of the element (async version).

        Args:
            path: Path to save the screenshot
        """
        if not self._is_async:
            raise RuntimeError("Use 'element.screenshot()' in sync mode")
            
        options = {}
        if path:
            options["path"] = path
            
        return await self._element.screenshot(**options)

    def evaluate(self, expression: str, arg: Any = None):
        """
        Evaluate JavaScript in the context of the element.

        Args:
            expression: JavaScript expression to evaluate
            arg: Argument to pass to the expression
        """
        if self._is_async:
            raise RuntimeError("Use 'await element.aevaluate()' in async mode")
            
        return self._element.evaluate(expression, arg)

    async def aevaluate(self, expression: str, arg: Any = None):
        """
        Evaluate JavaScript in the context of the element (async version).

        Args:
            expression: JavaScript expression to evaluate
            arg: Argument to pass to the expression
        """
        if not self._is_async:
            raise RuntimeError("Use 'element.evaluate()' in sync mode")
            
        return await self._element.evaluate(expression, arg)

    def type(self, text: str, delay: Optional[int] = None):
        """
        Type text into the element.

        Args:
            text: Text to type
            delay: Delay between key presses in milliseconds
        """
        if self._is_async:
            raise RuntimeError("Use 'await element.atype()' in async mode")
            
        options = {}
        if delay:
            options["delay"] = delay
            
        self._element.type(text, **options)

    async def atype(self, text: str, delay: Optional[int] = None):
        """
        Type text into the element (async version).

        Args:
            text: Text to type
            delay: Delay between key presses in milliseconds
        """
        if not self._is_async:
            raise RuntimeError("Use 'element.type()' in sync mode")
            
        options = {}
        if delay:
            options["delay"] = delay
            
        await self._element.type(text, **options)

    def select_option(self, value: Union[str, List[str]]):
        """
        Select an option from a <select> element.

        Args:
            value: Option value or list of option values to select
        """
        if self._is_async:
            raise RuntimeError("Use 'await element.aselect_option()' in async mode")
            
        return self._element.select_option(value)

    async def aselect_option(self, value: Union[str, List[str]]):
        """
        Select an option from a <select> element (async version).

        Args:
            value: Option value or list of option values to select
        """
        if not self._is_async:
            raise RuntimeError("Use 'element.select_option()' in sync mode")
            
        return await self._element.select_option(value)

    def check(self, force: bool = False):
        """
        Check the element (checkbox or radio button).

        Args:
            force: Whether to bypass actionability checks
        """
        if self._is_async:
            raise RuntimeError("Use 'await element.acheck()' in async mode")
            
        self._element.check(force=force)

    async def acheck(self, force: bool = False):
        """
        Check the element (checkbox or radio button) (async version).

        Args:
            force: Whether to bypass actionability checks
        """
        if not self._is_async:
            raise RuntimeError("Use 'element.check()' in sync mode")
            
        await self._element.check(force=force)

    def uncheck(self, force: bool = False):
        """
        Uncheck the element (checkbox).

        Args:
            force: Whether to bypass actionability checks
        """
        if self._is_async:
            raise RuntimeError("Use 'await element.auncheck()' in async mode")
            
        self._element.uncheck(force=force)

    async def auncheck(self, force: bool = False):
        """
        Uncheck the element (checkbox) (async version).

        Args:
            force: Whether to bypass actionability checks
        """
        if not self._is_async:
            raise RuntimeError("Use 'element.uncheck()' in sync mode")
            
        await self._element.uncheck(force=force) 