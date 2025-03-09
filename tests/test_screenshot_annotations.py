"""
Tests for the screenshot annotation functionality in the DOMExtractor class.
"""

import pytest
import asyncio
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import io
from PIL import Image, ImageDraw

from allyson import Browser, DOMExtractor


@pytest.mark.asyncio
async def test_screenshot_with_annotations():
    """Test that screenshot_with_annotations creates both clean and annotated screenshots."""
    # Create a mock page
    mock_page = MagicMock()
    
    # Create a simple test image
    test_image = Image.new('RGB', (800, 600), color='white')
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    # Mock the ascreenshot method to return the test image bytes
    mock_page.ascreenshot = AsyncMock()
    mock_page.ascreenshot.side_effect = [
        None,  # First call saves to file
        img_bytes.getvalue()  # Second call returns bytes
    ]
    
    # Create a DOMExtractor with the mock page
    extractor = DOMExtractor(mock_page)
    
    # Create test elements
    elements = [
        {
            "elementType": "button",
            "tagName": "button",
            "textContent": "Click me",
            "boundingBox": {
                "left": 100,
                "top": 100,
                "right": 200,
                "bottom": 150,
                "width": 100,
                "height": 50
            },
            "isVisible": True
        }
    ]
    
    # Create a temporary directory for test files
    test_dir = Path(os.path.dirname(__file__)) / "temp"
    test_dir.mkdir(exist_ok=True)
    
    # Test file path
    test_file = test_dir / "test_screenshot.png"
    
    # Mock PIL's Image.open to return our test image
    with patch('PIL.Image.open', return_value=test_image):
        # Mock PIL's ImageDraw.Draw
        mock_draw = MagicMock()
        with patch('PIL.ImageDraw.Draw', return_value=mock_draw):
            # Call screenshot_with_annotations
            result = await extractor.screenshot_with_annotations(
                path=str(test_file),
                elements=elements,
                full_page=False
            )
            
            # Check that ascreenshot was called twice
            assert mock_page.ascreenshot.call_count == 2
            
            # Check that the result contains the expected keys
            assert "clean" in result
            assert "annotated" in result
            
            # Check that the paths are correct
            assert result["clean"] == str(test_file)
            assert result["annotated"] == str(test_file).replace(".png", "_annotated.png")
    
    # Clean up
    if test_dir.exists():
        for file in test_dir.glob("*"):
            file.unlink()
        test_dir.rmdir()


@pytest.mark.asyncio
async def test_screenshot_with_element_map():
    """Test that screenshot_with_element_map creates a screenshot and element map."""
    # Create a mock page
    mock_page = MagicMock()
    
    # Mock the ascreenshot method
    mock_page.ascreenshot = AsyncMock()
    
    # Create a DOMExtractor with the mock page
    extractor = DOMExtractor(mock_page)
    
    # Create test elements
    elements = [
        {
            "elementType": "button",
            "tagName": "button",
            "textContent": "Click me",
            "attributes": {"id": "btn1", "class": "primary"},
            "boundingBox": {
                "left": 100,
                "top": 100,
                "right": 200,
                "bottom": 150,
                "width": 100,
                "height": 50
            },
            "isVisible": True
        },
        {
            "elementType": "link",
            "tagName": "a",
            "textContent": "Go to page",
            "attributes": {"href": "https://example.com"},
            "boundingBox": {
                "left": 300,
                "top": 100,
                "right": 400,
                "bottom": 150,
                "width": 100,
                "height": 50
            },
            "isVisible": True
        }
    ]
    
    # Create a temporary directory for test files
    test_dir = Path(os.path.dirname(__file__)) / "temp"
    test_dir.mkdir(exist_ok=True)
    
    # Test file path
    test_file = test_dir / "test_map.png"
    
    # Call screenshot_with_element_map
    result = await extractor.screenshot_with_element_map(
        path=str(test_file),
        elements=elements,
        full_page=False
    )
    
    # Check that ascreenshot was called
    mock_page.ascreenshot.assert_called_once()
    
    # Check that the result contains the expected keys
    assert "screenshot" in result
    assert "elementMap" in result
    
    # Check that the screenshot path is correct
    assert result["screenshot"] == str(test_file)
    
    # Check that the element map contains the expected elements
    assert len(result["elementMap"]) == 2
    assert result["elementMap"][0]["id"] == 1
    assert result["elementMap"][0]["elementType"] == "button"
    assert result["elementMap"][1]["id"] == 2
    assert result["elementMap"][1]["elementType"] == "link"
    
    # Clean up
    if test_dir.exists():
        for file in test_dir.glob("*"):
            file.unlink()
        test_dir.rmdir()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_integration_with_real_browser():
    """Integration test with a real browser."""
    # Skip this test if not running integration tests
    pytest.skip("Skipping integration test by default")
    
    # Create a simple HTML file for testing
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Screenshot Test Page</title>
    </head>
    <body>
        <h1 id="main-heading">Test Heading</h1>
        <button id="test-button" style="position: absolute; top: 100px; left: 100px; width: 100px; height: 50px;">Click Me</button>
        <a href="https://example.com" style="position: absolute; top: 200px; left: 100px;">Example Link</a>
    </body>
    </html>
    """
    
    # Create a temporary directory for the test file
    test_dir = Path(os.path.dirname(__file__)) / "temp"
    test_dir.mkdir(exist_ok=True)
    
    # Write the HTML file
    test_file = test_dir / "screenshot_test.html"
    with open(test_file, "w") as f:
        f.write(html_content)
    
    # Get the file URL
    file_url = f"file://{test_file.absolute()}"
    
    # Create a temporary directory for screenshots
    screenshots_dir = test_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    
    try:
        # Create a browser instance
        async with Browser(headless=True) as browser:
            # Navigate to the test page
            await browser.agoto(file_url)
            
            # Create a DOM extractor
            dom_extractor = DOMExtractor(browser._page)
            
            # Take a screenshot with annotations
            screenshot_path = str(screenshots_dir / "test_screenshot.png")
            result = await dom_extractor.screenshot_with_annotations(
                path=screenshot_path,
                full_page=False
            )
            
            # Check that the files exist
            assert os.path.exists(result["clean"])
            assert os.path.exists(result["annotated"])
            
            # Create an element map
            map_path = str(screenshots_dir / "test_map.png")
            map_result = await dom_extractor.screenshot_with_element_map(
                path=map_path,
                full_page=False
            )
            
            # Check that the file exists
            assert os.path.exists(map_result["screenshot"])
            
            # Check that the element map contains elements
            assert len(map_result["elementMap"]) > 0
            
    finally:
        # Clean up
        if test_file.exists():
            test_file.unlink()
        
        if screenshots_dir.exists():
            for file in screenshots_dir.glob("*"):
                file.unlink()
            screenshots_dir.rmdir()
        
        if test_dir.exists():
            try:
                test_dir.rmdir()
            except:
                pass 