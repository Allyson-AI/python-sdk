"""
Stealth module for Allyson.
Provides utilities to make browser automation less detectable.
"""

import logging
import random
from typing import Any, Dict, List, Optional

# Import stealth functions conditionally to handle cases where the package is not installed
try:
    from playwright_stealth import stealth_sync, stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False
    
logger = logging.getLogger(__name__)

# Latest Chrome user agent for version 133
CHROME_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"

# Additional browser arguments to avoid detection
BROWSER_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--disable-dev-shm-usage",
    "--no-sandbox",
    "--ignore-certificate-errors",
    "--enable-javascript",
    "--disable-notifications",
    "--disable-extensions",
    "--disable-popup-blocking",
    "--disable-web-security",
    "--disable-features=IsolateOrigins,site-per-process",
]

# Default viewport size
DEFAULT_VIEWPORT = {"width": 1280, "height": 720}

def apply_stealth_sync(page: Any) -> None:
    """
    Apply stealth mode to a synchronous page to avoid detection.
    
    Args:
        page: A Playwright page object
    """
    if not STEALTH_AVAILABLE:
        logger.warning("playwright-stealth package is not installed. Stealth mode will not be applied.")
        return
    
    try:
        stealth_sync(page)
        
        # Additional JavaScript to mask automation
        page.add_init_script("""
        // Override the navigator.webdriver property
        Object.defineProperty(navigator, 'webdriver', {
            get: () => false,
        });
        
        // Override the navigator.plugins to appear more like a regular browser
        if (navigator.plugins.length === 0) {
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
        }
        
        // Override the chrome object if it exists
        if (window.chrome) {
            window.chrome.runtime = {};
        }
        
        // Prevent detection via permissions API
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' || 
            parameters.name === 'midi' || 
            parameters.name === 'camera' || 
            parameters.name === 'microphone' || 
            parameters.name === 'geolocation' || 
            parameters.name === 'clipboard-read' || 
            parameters.name === 'clipboard-write'
        ) 
            ? Promise.resolve({state: 'prompt', onchange: null}) 
            : originalQuery(parameters);
        """)
        
        logger.debug("Applied stealth mode to synchronous page")
    except Exception as e:
        logger.error(f"Failed to apply stealth mode: {e}")

async def apply_stealth_async(page: Any) -> None:
    """
    Apply stealth mode to an asynchronous page to avoid detection.
    
    Args:
        page: A Playwright page object
    """
    if not STEALTH_AVAILABLE:
        logger.warning("playwright-stealth package is not installed. Stealth mode will not be applied.")
        return
    
    try:
        await stealth_async(page)
        
        # Additional JavaScript to mask automation
        await page.add_init_script("""
        // Override the navigator.webdriver property
        Object.defineProperty(navigator, 'webdriver', {
            get: () => false,
        });
        
        // Override the navigator.plugins to appear more like a regular browser
        if (navigator.plugins.length === 0) {
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
        }
        
        // Override the chrome object if it exists
        if (window.chrome) {
            window.chrome.runtime = {};
        }
        
        // Prevent detection via permissions API
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' || 
            parameters.name === 'midi' || 
            parameters.name === 'camera' || 
            parameters.name === 'microphone' || 
            parameters.name === 'geolocation' || 
            parameters.name === 'clipboard-read' || 
            parameters.name === 'clipboard-write'
        ) 
            ? Promise.resolve({state: 'prompt', onchange: null}) 
            : originalQuery(parameters);
        """)
        
        logger.debug("Applied stealth mode to asynchronous page")
    except Exception as e:
        logger.error(f"Failed to apply stealth mode: {e}")

def get_anti_detection_args() -> Dict[str, Any]:
    """
    Get browser context arguments that help avoid detection.
    
    Returns:
        Dictionary of context arguments
    """
    return {
        "user_agent": CHROME_USER_AGENT,
        "viewport": DEFAULT_VIEWPORT,
        "has_touch": False,
        "is_mobile": False,
        "device_scale_factor": 1,
        "reduced_motion": "no-preference",
        "forced_colors": "none",
        "color_scheme": "light",
        "locale": "en-US",
        "timezone_id": "America/New_York",
        "ignore_https_errors": True,
    }

def get_browser_args() -> List[str]:
    """
    Get browser launch arguments that help avoid detection.
    
    Returns:
        List of browser arguments
    """
    return BROWSER_ARGS 