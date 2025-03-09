"""
Example of using Allyson's DOMExtractor to take annotated screenshots.

This example demonstrates how to take screenshots with annotated bounding boxes
for interactive elements, which is useful for AI analysis.
"""

import asyncio
import os
import json
from typing import Dict, Any

from allyson import Browser, DOMExtractor


async def annotated_screenshot_example():
    """
    Example of taking screenshots with annotated bounding boxes.
    """
    print("Running annotated screenshot example...")
    
    # Create a browser instance
    async with Browser(headless=False) as browser:
        # Navigate to a website with interactive elements
        print("Navigating to a website...")
        await browser.agoto("https://www.wikipedia.org")
        
        # Create a DOM extractor
        dom_extractor = DOMExtractor(browser._page)
        
        # Create screenshots directory
        screenshots_dir = os.path.join(os.path.dirname(__file__), "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        # 1. Take a screenshot with annotations for all interactive elements
        print("\n1. Taking screenshot with annotations for all interactive elements...")
        screenshot_path = os.path.join(screenshots_dir, "wikipedia.png")
        result = await dom_extractor.screenshot_with_annotations(
            path=screenshot_path,
            full_page=False,
            show_element_ids=True,
            box_color="red",
            text_color="white"
        )
        
        print(f"Clean screenshot saved to: {result['clean']}")
        print(f"Annotated screenshot saved to: {result['annotated']}")
        
        # 2. Extract specific elements (search box and button)
        print("\n2. Extracting specific elements...")
        interactive_elements = await dom_extractor.extract_interactive_elements()
        
        # Filter for search elements
        search_elements = [
            element for element in interactive_elements
            if (element.get('elementType', '').startswith('input') and 
                'search' in element.get('attributes', {}).get('placeholder', '').lower()) or
               (element.get('elementType') == 'button' and 
                element.get('attributes', {}).get('type') == 'submit')
        ]
        
        print(f"Found {len(search_elements)} search-related elements")
        
        # 3. Take a screenshot with annotations only for search elements
        print("\n3. Taking screenshot with annotations only for search elements...")
        search_screenshot_path = os.path.join(screenshots_dir, "wikipedia_search.png")
        search_result = await dom_extractor.screenshot_with_annotations(
            path=search_screenshot_path,
            elements=search_elements,
            full_page=False,
            show_element_ids=True,
            box_color="blue",
            text_color="white"
        )
        
        print(f"Clean screenshot saved to: {search_result['clean']}")
        print(f"Annotated screenshot saved to: {search_result['annotated']}")
        
        # 4. Create an element map for AI analysis
        print("\n4. Creating element map for AI analysis...")
        map_screenshot_path = os.path.join(screenshots_dir, "wikipedia_map.png")
        map_result = await dom_extractor.screenshot_with_element_map(
            path=map_screenshot_path,
            full_page=False
        )
        
        # Save the element map to a JSON file
        element_map_path = os.path.join(screenshots_dir, "wikipedia_element_map.json")
        with open(element_map_path, 'w') as f:
            json.dump(map_result["elementMap"], f, indent=2)
        
        print(f"Screenshot saved to: {map_result['screenshot']}")
        print(f"Element map saved to: {element_map_path}")
        print(f"Found {len(map_result['elementMap'])} interactive elements")
        
        # 5. Navigate to another page and take an annotated screenshot
        print("\n5. Navigating to another page...")
        await browser.agoto("https://github.com")
        
        # Take a screenshot with annotations
        github_screenshot_path = os.path.join(screenshots_dir, "github.png")
        github_result = await dom_extractor.screenshot_with_annotations(
            path=github_screenshot_path,
            full_page=False
        )
        
        print(f"Clean screenshot saved to: {github_result['clean']}")
        print(f"Annotated screenshot saved to: {github_result['annotated']}")
        
        # Wait a moment before closing
        print("\nExample completed. Closing browser in 3 seconds...")
        await asyncio.sleep(3)


async def main():
    """Run the example."""
    await annotated_screenshot_example()


if __name__ == "__main__":
    asyncio.run(main()) 