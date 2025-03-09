"""
Tools for the agent loop.

This module provides tools that can be used by the agent loop to interact with web pages.
"""

import asyncio
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ToolType(str, Enum):
    """Types of tools that can be used by the agent."""
    BROWSER = "browser"  # Browser interaction tools (click, type, etc.)
    SYSTEM = "system"  # System tools (file operations, etc.)
    CUSTOM = "custom"  # Custom tools provided by the user


class Tool(BaseModel):
    """Definition of a tool that can be used by the agent."""
    name: str
    description: str
    type: ToolType
    parameters_schema: Dict[str, Any]
    function: Callable
    
    class Config:
        arbitrary_types_allowed = True


async def _goto(browser, url: str) -> Dict[str, Any]:
    """Navigate to a URL."""
    await browser.agoto(url)
    return {"url": url}


async def _click(browser, element_id: int) -> Dict[str, Any]:
    """Click on an element by its ID."""
    # Get the interactive elements
    dom_extractor = browser._dom_extractor
    elements = await dom_extractor.extract_interactive_elements()
    
    # Check if the element ID is valid
    if element_id < 1 or element_id > len(elements):
        raise ValueError(f"Element ID {element_id} is out of range (1-{len(elements)})")
    
    # Get the element (1-indexed to 0-indexed)
    element = elements[element_id - 1]
    
    # Get the element's bounding box
    box = element["boundingBox"]
    
    # Click in the center of the element
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    
    # Use the page's mouse click method
    await browser._page._page.mouse.click(x, y)
    
    return {
        "element_id": element_id,
        "element_type": element.get("elementType", "unknown"),
        "text": element.get("textContent", "")
    }


async def _type(browser, element_id: int, text: str) -> Dict[str, Any]:
    """Type text into an element by its ID."""
    # Get the interactive elements
    dom_extractor = browser._dom_extractor
    elements = await dom_extractor.extract_interactive_elements()
    
    # Check if the element ID is valid
    if element_id < 1 or element_id > len(elements):
        raise ValueError(f"Element ID {element_id} is out of range (1-{len(elements)})")
    
    # Get the element (1-indexed to 0-indexed)
    element = elements[element_id - 1]
    
    # Get the element's bounding box
    box = element["boundingBox"]
    
    # Click in the center of the element
    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    
    # Use the page's mouse click method
    await browser._page._page.mouse.click(x, y)
    
    # Type the text using the page's keyboard
    await browser._page._page.keyboard.type(text)
    
    return {
        "element_id": element_id,
        "text": text,
        "element_type": element.get("elementType", "unknown")
    }


async def _scroll(browser, direction: str, distance: int = 300) -> Dict[str, Any]:
    """Scroll the page."""
    if direction == "up":
        await browser._page.aevaluate(f"window.scrollBy(0, -{distance})")
    elif direction == "down":
        await browser._page.aevaluate(f"window.scrollBy(0, {distance})")
    elif direction == "left":
        await browser._page.aevaluate(f"window.scrollBy(-{distance}, 0)")
    elif direction == "right":
        await browser._page.aevaluate(f"window.scrollBy({distance}, 0)")
    
    return {"direction": direction, "distance": distance}


async def _enter(browser) -> Dict[str, Any]:
    """Press the Enter key."""
    await browser._page._page.keyboard.press("Enter")
    return {"key": "Enter"}


def get_default_tools(browser) -> List[Tool]:
    """
    Get the default tools for browser interaction.
    
    Args:
        browser: Browser instance to use for the tools
        
    Returns:
        List of default tools
    """
    return [
        # Browser navigation tool
        Tool(
            name="goto",
            description="Navigate to a URL",
            type=ToolType.BROWSER,
            parameters_schema={
                "url": {"type": "string", "description": "URL to navigate to"}
            },
            function=lambda url: _goto(browser, url)
        ),
        
        # Click tool
        Tool(
            name="click",
            description="Click on an element by its ID number",
            type=ToolType.BROWSER,
            parameters_schema={
                "element_id": {"type": "integer", "description": "ID of the element to click"}
            },
            function=lambda element_id: _click(browser, element_id)
        ),
        
        # Type tool
        Tool(
            name="type",
            description="Type text into an element by its ID number",
            type=ToolType.BROWSER,
            parameters_schema={
                "element_id": {"type": "integer", "description": "ID of the element to type into"},
                "text": {"type": "string", "description": "Text to type"}
            },
            function=lambda element_id, text: _type(browser, element_id, text)
        ),
        
        # Enter key tool
        Tool(
            name="enter",
            description="Press the Enter key to submit a form or activate the default action",
            type=ToolType.BROWSER,
            parameters_schema={},
            function=lambda: _enter(browser)
        ),
        
        # Scroll tool
        Tool(
            name="scroll",
            description="Scroll the page",
            type=ToolType.BROWSER,
            parameters_schema={
                "direction": {"type": "string", "description": "Direction to scroll", "enum": ["up", "down", "left", "right"]},
                "distance": {"type": "integer", "description": "Distance to scroll in pixels", "default": 300}
            },
            function=lambda direction, distance=300: _scroll(browser, direction, distance)
        ),
        
        # Done tool
        Tool(
            name="done",
            description="Mark the task as complete",
            type=ToolType.SYSTEM,
            parameters_schema={
                "message": {"type": "string", "description": "Final message to the user"}
            },
            function=lambda message: {"done": True, "message": message}
        )
    ] 