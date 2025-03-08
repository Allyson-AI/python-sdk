"""
Pytest configuration for Allyson tests.
"""

import os
import pytest
import asyncio
from playwright.sync_api import sync_playwright
from playwright.async_api import async_playwright


# We don't need to define our own event_loop fixture as pytest-asyncio provides one
# This was causing a warning


@pytest.fixture(scope="session", autouse=True)
def install_browsers():
    """Install browsers if they are not already installed."""
    import subprocess
    import sys
    
    try:
        # Check if browsers are installed
        with sync_playwright() as p:
            # Try to launch each browser type
            for browser_type in [p.chromium, p.firefox, p.webkit]:
                browser = browser_type.launch()
                browser.close()
        
        # If we get here, browsers are installed
        return
    except Exception:
        # Browsers are not installed, install them
        print("Installing browsers...")
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install"],
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            pytest.fail(f"Failed to install browsers: {result.stderr}")


@pytest.fixture
def screenshots_dir(tmp_path):
    """Create a temporary directory for screenshots."""
    screenshots_dir = tmp_path / "screenshots"
    screenshots_dir.mkdir()
    return screenshots_dir 