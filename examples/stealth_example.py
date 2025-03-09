"""
Example demonstrating the anti-detection features of Allyson.
This example shows how the built-in anti-detection features help avoid bot detection.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import allyson
sys.path.append(str(Path(__file__).parent.parent))

from allyson import Browser


async def test_anti_detection():
    """
    Test the anti-detection features by visiting a bot detection website.
    """
    print("Testing Allyson's anti-detection features...")
    
    # Create a browser with anti-detection features (always enabled)
    async with Browser(
        headless=False,  # Set to True for headless operation
    ) as browser:
        # Visit a website that checks for bot detection
        await browser.agoto("https://bot.sannysoft.com/")
        print("Navigated to bot detection test page")
        
        # Wait for the page to fully load and run all detection tests
        await asyncio.sleep(5)
        
        # Take a screenshot of the results
        screenshots_dir = Path(__file__).parent / "screenshots"
        screenshots_dir.mkdir(exist_ok=True)
        
        screenshot_path = screenshots_dir / "anti_detection_test_results.png"
        await browser.apage.screenshot(path=str(screenshot_path))
        print(f"Screenshot saved to: {screenshot_path}")
        
        # Wait for user to see the results
        print("Press Enter to exit...")
        await asyncio.to_thread(input)
    
    print("\nTest completed. The screenshot shows how well Allyson's anti-detection features work.")


if __name__ == "__main__":
    asyncio.run(test_anti_detection()) 