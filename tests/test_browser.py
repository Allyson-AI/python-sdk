"""
Tests for the Browser class.
"""

import os
import pytest
import tempfile
from pathlib import Path

from allyson import Browser


def test_browser_sync():
    """Test the synchronous Browser API."""
    with Browser(headless=True) as browser:
        # Test navigation
        browser.goto("https://example.com")
        assert "Example" in browser.get_title()
        
        # Test evaluation
        title = browser.evaluate("document.title")
        assert "Example" in title
        
        # Test screenshot
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot_path = os.path.join(temp_dir, "screenshot.png")
            browser.screenshot(screenshot_path)
            assert os.path.exists(screenshot_path)
            assert os.path.getsize(screenshot_path) > 0


@pytest.mark.asyncio
async def test_browser_async():
    """Test the asynchronous Browser API."""
    async with Browser(headless=True) as browser:
        # Test navigation
        await browser.agoto("https://example.com")
        title = await browser.aevaluate("document.title")
        assert "Example" in title
        
        # Test element interaction
        heading = await browser.await_for_selector("h1")
        heading_text = await heading.aget_text()
        assert "Example" in heading_text
        
        # Test screenshot
        with tempfile.TemporaryDirectory() as temp_dir:
            screenshot_path = os.path.join(temp_dir, "screenshot.png")
            await browser.ascreenshot(screenshot_path)
            assert os.path.exists(screenshot_path)
            assert os.path.getsize(screenshot_path) > 0


def test_browser_multiple_pages():
    """Test working with multiple pages."""
    with Browser(headless=True) as browser:
        # Navigate to first page
        browser.goto("https://example.com")
        
        # Create a new page
        page2 = browser.new_page()
        
        # Navigate the new page to a different URL
        page2.goto("https://playwright.dev")
        
        # Check that the pages have different content
        assert "Example" in browser.get_title()
        assert "Playwright" in page2.get_title()


@pytest.mark.asyncio
async def test_browser_multiple_pages_async():
    """Test working with multiple pages asynchronously."""
    async with Browser(headless=True) as browser:
        # Navigate to first page
        await browser.agoto("https://example.com")
        
        # Create a new page
        page2 = await browser.anew_page()
        
        # Navigate the new page to a different URL
        await page2.agoto("https://playwright.dev")
        
        # Check that the pages have different content
        title1 = await browser.aevaluate("document.title")
        title2 = await page2.aevaluate("document.title")
        
        assert "Example" in title1
        assert "Playwright" in title2 