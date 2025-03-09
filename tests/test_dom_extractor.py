"""
Tests for the DOMExtractor class.
"""

import pytest
import asyncio
from unittest.mock import MagicMock, patch
import inspect

from allyson import DOMExtractor


# Helper function to create a mock that works with async/await
def async_mock(return_value=None):
    async def _async_mock(*args, **kwargs):
        return return_value
    return MagicMock(side_effect=_async_mock)


@pytest.mark.asyncio
async def test_extract_dom_elements():
    """Test that extract_dom_elements calls page.evaluate with the correct arguments."""
    # Create a mock page with async evaluate method
    mock_page = MagicMock()
    mock_page.aevaluate = async_mock({"tagName": "body", "children": []})
    
    # Create a DOMExtractor with the mock page
    extractor = DOMExtractor(mock_page)
    
    # Call extract_dom_elements
    result = await extractor.extract_dom_elements()
    
    # Check that page.evaluate was called
    mock_page.aevaluate.assert_called_once()
    
    # Check that the result is what we expect
    assert result == {"tagName": "body", "children": []}


@pytest.mark.asyncio
async def test_extract_interactive_elements():
    """Test that extract_interactive_elements calls page.evaluate."""
    # Create a mock page with async evaluate method
    mock_page = MagicMock()
    mock_page.aevaluate = async_mock([
        {"elementType": "button", "tagName": "button", "textContent": "Click me"}
    ])
    
    # Create a DOMExtractor with the mock page
    extractor = DOMExtractor(mock_page)
    
    # Call extract_interactive_elements
    result = await extractor.extract_interactive_elements()
    
    # Check that page.evaluate was called
    mock_page.aevaluate.assert_called_once()
    
    # Check that the result is what we expect
    assert len(result) == 1
    assert result[0]["elementType"] == "button"
    assert result[0]["textContent"] == "Click me"


@pytest.mark.asyncio
async def test_highlight_elements():
    """Test that highlight_elements calls page.evaluate with the correct arguments."""
    # Create a mock page with async evaluate method
    mock_page = MagicMock()
    mock_page.aevaluate = async_mock(None)
    
    # Create a DOMExtractor with the mock page
    extractor = DOMExtractor(mock_page)
    
    # Call highlight_elements
    elements_data = {
        "tagName": "div",
        "boundingBox": {"top": 0, "left": 0, "width": 100, "height": 100},
        "isVisible": True,
        "elementType": "container"
    }
    await extractor.highlight_elements(elements_data)
    
    # Check that page.evaluate was called with the correct arguments
    mock_page.aevaluate.assert_called_once()
    args, kwargs = mock_page.aevaluate.call_args
    assert isinstance(args[0], str)  # Script is a string
    
    # Check that the arg parameter contains the expected data
    arg = args[1]
    assert "elementsData" in arg
    assert arg["elementsData"] == elements_data
    assert "options" in arg
    assert arg["options"]["durationMs"] == 2000  # Default duration is 2000ms


@pytest.mark.asyncio
async def test_get_page_metadata():
    """Test that get_page_metadata calls page.evaluate."""
    # Create a mock page with async evaluate method
    mock_page = MagicMock()
    mock_page.aevaluate = async_mock({
        "title": "Test Page",
        "url": "https://example.com",
        "domain": "example.com"
    })
    
    # Create a DOMExtractor with the mock page
    extractor = DOMExtractor(mock_page)
    
    # Call get_page_metadata
    result = await extractor.get_page_metadata()
    
    # Check that page.evaluate was called
    mock_page.aevaluate.assert_called_once()
    
    # Check that the result is what we expect
    assert result["title"] == "Test Page"
    assert result["url"] == "https://example.com"
    assert result["domain"] == "example.com"


@pytest.mark.asyncio
async def test_extract_dom_for_ai():
    """Test that extract_dom_for_ai calls the appropriate methods."""
    # Create a mock page with async evaluate method
    mock_page = MagicMock()
    
    # Create metadata and DOM tree results
    metadata_result = {
        "title": "Test Page",
        "url": "https://example.com"
    }
    dom_tree_result = {
        "tagName": "body",
        "children": []
    }
    
    # Create a DOMExtractor with the mock page
    extractor = DOMExtractor(mock_page)
    
    # Patch the methods that would be called
    with patch.object(extractor, 'get_page_metadata', return_value=metadata_result) as mock_get_metadata, \
         patch.object(extractor, 'extract_dom_elements', return_value=dom_tree_result) as mock_extract_dom:
        
        # Call extract_dom_for_ai
        result = await extractor.extract_dom_for_ai()
        
        # Check that the methods were called
        mock_get_metadata.assert_called_once()
        mock_extract_dom.assert_called_once()
        
        # Check that the result contains the expected keys
        assert "metadata" in result
        assert "domTree" in result
        assert result["metadata"] == metadata_result
        assert result["domTree"] == dom_tree_result


@pytest.mark.asyncio
async def test_inject_dom_observer():
    """Test that inject_dom_observer calls page.evaluate."""
    # Create a mock page with async evaluate method
    mock_page = MagicMock()
    mock_page.aevaluate = async_mock("DOM observer injected successfully")
    
    # Create a DOMExtractor with the mock page
    extractor = DOMExtractor(mock_page)
    
    # Call inject_dom_observer
    result = await extractor.inject_dom_observer()
    
    # Check that page.evaluate was called
    mock_page.aevaluate.assert_called_once()
    
    # Check that the result is what we expect
    assert result == "DOM observer injected successfully"


@pytest.mark.asyncio
async def test_get_dom_changes():
    """Test that get_dom_changes calls page.evaluate."""
    # Create a mock page with async evaluate method
    mock_page = MagicMock()
    mock_page.aevaluate = async_mock([
        {
            "type": "childList",
            "timestamp": "2023-01-01T00:00:00.000Z",
            "addedNodes": [{"nodeType": 1, "tagName": "div"}],
            "removedNodes": []
        }
    ])
    
    # Create a DOMExtractor with the mock page
    extractor = DOMExtractor(mock_page)
    
    # Call get_dom_changes
    result = await extractor.get_dom_changes()
    
    # Check that page.evaluate was called
    mock_page.aevaluate.assert_called_once()
    
    # Check that the result is what we expect
    assert len(result) == 1
    assert result[0]["type"] == "childList"
    assert len(result[0]["addedNodes"]) == 1
    assert result[0]["addedNodes"][0]["tagName"] == "div" 