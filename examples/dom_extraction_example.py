"""
Simple example of using Allyson's DOMExtractor to extract DOM elements.
"""

import asyncio
import json
import os
import sys
from typing import Dict, Any

from allyson import Browser, DOMExtractor


async def simple_dom_extraction():
    """
    Simple example of extracting DOM elements from a webpage.
    """
    print("Running simple DOM extraction example...")
    
    # Create a browser instance
    async with Browser(headless=False) as browser:
        # Navigate to a website
        print("Navigating to example.com...")
        await browser.agoto("https://example.com")
        
        # Create a DOM extractor
        dom_extractor = DOMExtractor(browser._page)
        
        # 1. Extract and print page metadata
        print("\n1. Page Metadata:")
        metadata = await dom_extractor.get_page_metadata()
        print(f"Title: {metadata['title']}")
        print(f"URL: {metadata['url']}")
        print(f"Domain: {metadata['domain']}")
        print(f"Viewport: {metadata['viewport']['width']}x{metadata['viewport']['height']}")
        
        # 2. Extract interactive elements
        print("\n2. Interactive Elements:")
        interactive_elements = await dom_extractor.extract_interactive_elements()
        print(f"Found {len(interactive_elements)} interactive elements")
        
        # Print details of each interactive element
        for i, element in enumerate(interactive_elements):
            print(f"\nElement {i+1}:")
            print(f"  Type: {element['elementType']}")
            print(f"  Tag: {element['tagName']}")
            if 'textContent' in element and element['textContent']:
                print(f"  Text: {element['textContent']}")
            if 'id' in element and element['id']:
                print(f"  ID: {element['id']}")
            if 'boundingBox' in element:
                box = element['boundingBox']
                print(f"  Position: ({box['x']}, {box['y']}), Size: {box['width']}x{box['height']}")
        
        # 3. Highlight interactive elements
        print("\n3. Highlighting interactive elements for 5 seconds...")
        if interactive_elements:
            elements_data = {
                "children": interactive_elements
            }
            await dom_extractor.highlight_elements(elements_data, duration_ms=5000)
            # Wait for the highlighting to be visible
            await asyncio.sleep(5)
        
        # 4. Extract full DOM tree (limited depth for readability)
        print("\n4. Extracting DOM tree (limited depth)...")
        dom_tree = await dom_extractor.extract_dom_elements(max_depth=2)
        
        # Print a summary of the DOM tree
        print(f"Root element: {dom_tree['tagName']}")
        print(f"Number of direct children: {len(dom_tree.get('children', []))}")
        
        # 5. Inject DOM observer
        print("\n5. Injecting DOM observer...")
        await dom_extractor.inject_dom_observer()
        
        # 6. Navigate to another page to demonstrate DOM changes
        print("\n6. Navigating to another page...")
        await browser.agoto("https://playwright.dev")
        
        # Wait a moment for the page to load
        await asyncio.sleep(2)
        
        # 7. Get DOM changes
        print("\n7. Getting DOM changes...")
        changes = await dom_extractor.get_dom_changes()
        print(f"Detected {len(changes)} DOM changes after navigation")
        
        # 8. Extract DOM for AI (formatted for AI consumption)
        print("\n8. Extracting DOM for AI...")
        dom_for_ai = await dom_extractor.extract_dom_for_ai(include_interactive_only=True)
        print(f"Extracted metadata and {len(dom_for_ai.get('interactiveElements', []))} interactive elements for AI")
        
        # Wait a moment before closing
        print("\nExample completed. Closing browser in 3 seconds...")
        await asyncio.sleep(3)


async def main():
    """Run the example."""
    await simple_dom_extraction()


if __name__ == "__main__":
    asyncio.run(main()) 