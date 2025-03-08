"""
Simple example of using Allyson to automate a web browser.
"""

import asyncio
import os
from allyson import Browser


async def async_example():
    """Example using the async API."""
    print("Running async example...")
    
    # Create a browser instance with async context manager
    async with Browser(headless=False) as browser:
        # Navigate to a website
        await browser.agoto("https://example.com")
        
        # Get the page title
        title = await browser.aevaluate("document.title")
        print(f"Page title: {title}")
        
        # Take a screenshot
        screenshots_dir = os.path.join(os.path.dirname(__file__), "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        await browser.ascreenshot(os.path.join(screenshots_dir, "example_async.png"))
        
        # Navigate to another page
        await browser.agoto("https://playwright.dev")
        
        # Wait for a specific element
        await browser.await_for_selector(".navbar__title")
        
        # Get text content
        title_text = await browser.aget_text(".navbar__title")
        print(f"Playwright title: {title_text}")
        
        # Take another screenshot
        await browser.ascreenshot(os.path.join(screenshots_dir, "playwright_async.png"))


def sync_example():
    """Example using the sync API."""
    print("Running sync example...")
    
    # Create a browser instance with context manager
    with Browser(headless=False) as browser:
        # Navigate to a website
        browser.goto("https://example.com")
        
        # Get the page title
        title = browser.evaluate("document.title")
        print(f"Page title: {title}")
        
        # Take a screenshot
        screenshots_dir = os.path.join(os.path.dirname(__file__), "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        browser.screenshot(os.path.join(screenshots_dir, "example_sync.png"))
        
        # Navigate to another page
        browser.goto("https://playwright.dev")
        
        # Wait for a specific element
        browser.wait_for_selector(".navbar__title")
        
        # Get text content
        title_text = browser.get_text(".navbar__title")
        print(f"Playwright title: {title_text}")
        
        # Take another screenshot
        browser.screenshot(os.path.join(screenshots_dir, "playwright_sync.png"))


async def main():
    """Run both examples."""
    # Run the async example
    await async_example()
    
    # Run the sync example (in a separate thread since it's blocking)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        await asyncio.get_event_loop().run_in_executor(executor, sync_example)


if __name__ == "__main__":
    asyncio.run(main()) 