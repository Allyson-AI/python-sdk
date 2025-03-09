"""
Stealth module for Allyson.
Provides utilities to make browser automation less detectable.
"""

import logging
from typing import Any, Dict, Optional

# Import stealth functions conditionally to handle cases where the package is not installed
try:
    from playwright_stealth import stealth_sync, stealth_async
    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False
    
logger = logging.getLogger(__name__)

# Latest Chrome user agent for version 133
CHROME_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"

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
        "viewport": {"width": 1280, "height": 720},
        "has_touch": False,
        "is_mobile": False,
        "device_scale_factor": 1,
        "reduced_motion": "no-preference",
        "forced_colors": "none",
        "color_scheme": "light",
    } 