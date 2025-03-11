"""
Integration tests for the DOMExtractor class.

These tests use a real browser to test the DOMExtractor functionality.
"""

import pytest
import asyncio
import os
from pathlib import Path

from allyson import Browser, DOMExtractor


@pytest.mark.asyncio
async def test_dom_extractor_with_real_browser():
    """Test DOMExtractor with a real browser on a simple HTML page."""
    # Create a simple HTML file for testing
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DOMExtractor Test Page</title>
        <meta name="description" content="Test page for DOMExtractor">
    </head>
    <body>
        <h1 id="main-heading">Test Heading</h1>
        <p>This is a test paragraph.</p>
        <button id="test-button">Click Me</button>
        <a href="https://example.com">Example Link</a>
        <input type="text" placeholder="Enter text">
        <div class="container">
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
                <li>Item 3</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    # Create a temporary directory for the test file
    test_dir = Path(os.path.dirname(__file__)) / "temp"
    test_dir.mkdir(exist_ok=True)
    
    # Write the HTML file
    test_file = test_dir / "test_page.html"
    with open(test_file, "w") as f:
        f.write(html_content)
    
    # Get the file URL
    file_url = f"file://{test_file.absolute()}"
    
    try:
        # Create a browser instance
        async with Browser(headless=True) as browser:
            # Navigate to the test page
            await browser.agoto(file_url)
            
            # Create a DOM extractor
            dom_extractor = DOMExtractor(browser._page)
            
            # Test metadata extraction
            metadata = await dom_extractor.get_page_metadata()
            assert metadata["title"] == "DOMExtractor Test Page"
            assert any(meta.get("content") == "Test page for DOMExtractor" for meta in metadata["meta"])
            
            # Test interactive elements extraction
            interactive_elements = await dom_extractor.extract_interactive_elements()
            assert len(interactive_elements) == 3  # button, link, and input
            
            # Check if we found the button
            button = next((el for el in interactive_elements if el["elementType"] == "button"), None)
            assert button is not None
            assert button["textContent"] == "Click Me"
            assert button["attributes"]["id"] == "test-button"
            
            # Check if we found the link (a tag)
            link = next((el for el in interactive_elements if el["tagName"] == "a"), None)
            assert link is not None
            assert link["textContent"] == "Example Link"
            assert link["attributes"]["href"] == "https://example.com"
            
            # Check if we found the input
            input_el = next((el for el in interactive_elements if el["elementType"].startswith("input")), None)
            assert input_el is not None
            assert input_el["attributes"]["placeholder"] == "Enter text"
            
            # Test full DOM extraction
            dom_tree = await dom_extractor.extract_dom_elements()
            assert dom_tree["tagName"] == "body"
            
            # Find the heading
            heading = None
            for child in dom_tree["children"]:
                if child["tagName"] == "h1":
                    heading = child
                    break
            
            assert heading is not None
            assert heading["textContent"] == "Test Heading"
            assert heading["attributes"]["id"] == "main-heading"
            
            # Test DOM for AI extraction
            dom_for_ai = await dom_extractor.extract_dom_for_ai()
            assert "metadata" in dom_for_ai
            assert "domTree" in dom_for_ai
            assert dom_for_ai["metadata"]["title"] == "DOMExtractor Test Page"
            
    finally:
        # Clean up the test file
        if test_file.exists():
            test_file.unlink()
        
        # Try to remove the directory (will only succeed if empty)
        try:
            test_dir.rmdir()
        except:
            pass


@pytest.mark.asyncio
async def test_dom_observer():
    """Test the DOM observer functionality."""
    # Create a simple HTML file for testing
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DOM Observer Test</title>
        <script>
            // Function to add a new element after a delay
            function addElement() {
                setTimeout(() => {
                    const newDiv = document.createElement('div');
                    newDiv.id = 'new-element';
                    newDiv.textContent = 'New Element';
                    document.body.appendChild(newDiv);
                }, 500);
            }
        </script>
    </head>
    <body onload="addElement()">
        <h1>DOM Observer Test</h1>
    </body>
    </html>
    """
    
    # Create a temporary directory for the test file
    test_dir = Path(os.path.dirname(__file__)) / "temp"
    test_dir.mkdir(exist_ok=True)
    
    # Write the HTML file
    test_file = test_dir / "observer_test.html"
    with open(test_file, "w") as f:
        f.write(html_content)
    
    # Get the file URL
    file_url = f"file://{test_file.absolute()}"
    
    try:
        # Create a browser instance
        async with Browser(headless=True) as browser:
            # Navigate to the test page
            await browser.agoto(file_url)
            
            # Create a DOM extractor
            dom_extractor = DOMExtractor(browser._page)
            
            # Inject the DOM observer
            await dom_extractor.inject_dom_observer()
            
            # Wait for the new element to be added
            await asyncio.sleep(1)
            
            # Get the DOM changes
            changes = await dom_extractor.get_dom_changes()
            
            # Check that we detected the new element
            assert len(changes) > 0
            
            # Find the childList change that added the new element
            added_element = False
            for change in changes:
                if change["type"] == "childList" and change["addedNodes"]:
                    for node in change["addedNodes"]:
                        if node.get("tagName") == "div":
                            added_element = True
                            break
            
            assert added_element, "Failed to detect the added div element"
            
    finally:
        # Clean up the test file
        if test_file.exists():
            test_file.unlink()
        
        # Try to remove the directory (will only succeed if empty)
        try:
            test_dir.rmdir()
        except:
            pass 