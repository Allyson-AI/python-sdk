"""
DOM extraction and injection module for Allyson.

This module provides functionality to extract DOM elements with their properties,
bounding boxes, and other metadata, and to inject custom scripts into the page
for enhanced interaction.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union
from PIL import Image, ImageDraw, ImageFont
import io
import base64

logger = logging.getLogger(__name__)


class DOMExtractor:
    """
    Class for extracting DOM information and injecting scripts into web pages.
    
    This class works with the Browser and Page classes to provide enhanced
    DOM interaction capabilities, particularly for AI-driven automation.
    """
    
    def __init__(self, page):
        """
        Initialize a new DOMExtractor instance.
        
        Args:
            page: An Allyson Page object
        """
        self.page = page
    
    async def extract_dom_elements(self, selector: str = "body", include_text: bool = True, 
                                  include_attributes: bool = True, include_bounding_box: bool = True,
                                  include_styles: bool = False, max_depth: int = 10) -> Dict[str, Any]:
        """
        Extract DOM elements and their properties.
        
        Args:
            selector: CSS selector to start extraction from
            include_text: Whether to include text content
            include_attributes: Whether to include element attributes
            include_bounding_box: Whether to include bounding box information
            include_styles: Whether to include computed styles
            max_depth: Maximum depth of DOM tree to traverse
            
        Returns:
            Dictionary containing DOM structure with properties
        """
        script = """(arg) => {
            const selector = arg.selector;
            const includeText = arg.includeText;
            const includeAttributes = arg.includeAttributes;
            const includeBoundingBox = arg.includeBoundingBox;
            const includeStyles = arg.includeStyles;
            const maxDepth = arg.maxDepth;
            
            function extractElement(element, depth = 0) {
                if (depth > maxDepth) return null;
                
                const result = {
                    tagName: element.tagName.toLowerCase(),
                    nodeType: element.nodeType,
                    id: element.id || undefined,
                    className: element.className || undefined,
                    children: []
                };
                
                // Add element type classification
                result.elementType = getElementType(element);
                
                // Add text content if requested
                if (includeText && element.textContent) {
                    result.textContent = element.textContent.trim();
                }
                
                // Add attributes if requested
                if (includeAttributes) {
                    result.attributes = {};
                    for (const attr of element.attributes || []) {
                        result.attributes[attr.name] = attr.value;
                    }
                }
                
                // Add bounding box if requested
                if (includeBoundingBox) {
                    const rect = element.getBoundingClientRect();
                    result.boundingBox = {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                        top: rect.top,
                        right: rect.right,
                        bottom: rect.bottom,
                        left: rect.left
                    };
                    
                    // Add visibility information
                    const style = window.getComputedStyle(element);
                    result.isVisible = !(style.display === 'none' || 
                                        style.visibility === 'hidden' || 
                                        style.opacity === '0' ||
                                        (rect.width === 0 && rect.height === 0));
                }
                
                // Add computed styles if requested
                if (includeStyles) {
                    result.styles = {};
                    const style = window.getComputedStyle(element);
                    const relevantStyles = [
                        'color', 'backgroundColor', 'fontSize', 'fontWeight',
                        'display', 'visibility', 'position', 'zIndex',
                        'cursor', 'pointerEvents'
                    ];
                    
                    for (const prop of relevantStyles) {
                        result.styles[prop] = style[prop];
                    }
                }
                
                // Process children
                for (const child of element.children) {
                    const childData = extractElement(child, depth + 1);
                    if (childData) {
                        result.children.push(childData);
                    }
                }
                
                return result;
            }
            
            function getElementType(element) {
                const tagName = element.tagName.toLowerCase();
                const type = element.getAttribute('type');
                const role = element.getAttribute('role');
                
                // Interactive elements
                if (tagName === 'a') return 'link';
                if (tagName === 'button' || role === 'button') return 'button';
                if (tagName === 'input') {
                    if (type === 'text' || type === 'email' || type === 'password' || type === 'search') return 'textbox';
                    if (type === 'checkbox') return 'checkbox';
                    if (type === 'radio') return 'radio';
                    if (type === 'submit' || type === 'button') return 'button';
                    return `input-${type || 'unknown'}`;
                }
                if (tagName === 'select') return 'dropdown';
                if (tagName === 'textarea') return 'textarea';
                
                // Content elements
                if (tagName === 'img') return 'image';
                if (tagName === 'video') return 'video';
                if (tagName === 'audio') return 'audio';
                if (tagName === 'iframe') return 'iframe';
                if (tagName === 'canvas') return 'canvas';
                
                // Text and heading elements
                if (tagName === 'h1' || tagName === 'h2' || tagName === 'h3' || 
                    tagName === 'h4' || tagName === 'h5' || tagName === 'h6') return 'heading';
                if (tagName === 'p') return 'paragraph';
                if (tagName === 'span' || tagName === 'div') {
                    // Check if it's likely a text element
                    if (element.children.length === 0 && element.textContent.trim()) return 'text';
                }
                
                // Container elements
                if (tagName === 'form') return 'form';
                if (tagName === 'nav') return 'navigation';
                if (tagName === 'header') return 'header';
                if (tagName === 'footer') return 'footer';
                if (tagName === 'main') return 'main';
                if (tagName === 'section') return 'section';
                if (tagName === 'article') return 'article';
                if (tagName === 'aside') return 'aside';
                
                // List elements
                if (tagName === 'ul' || tagName === 'ol') return 'list';
                if (tagName === 'li') return 'list-item';
                
                // Table elements
                if (tagName === 'table') return 'table';
                if (tagName === 'tr') return 'table-row';
                if (tagName === 'td' || tagName === 'th') return 'table-cell';
                
                // Default
                return 'element';
            }
            
            const rootElement = document.querySelector(selector);
            if (!rootElement) return null;
            
            return extractElement(rootElement);
        }"""
        
        arg = {
            "selector": selector,
            "includeText": include_text,
            "includeAttributes": include_attributes,
            "includeBoundingBox": include_bounding_box,
            "includeStyles": include_styles,
            "maxDepth": max_depth
        }
        
        return await self.page.aevaluate(script, arg)
    
    async def highlight_elements(self, elements_data: Dict[str, Any], duration_ms: int = 2000, 
                               highlight_color: str = "rgba(255, 0, 0, 0.3)") -> None:
        """
        Highlight elements on the page based on their bounding boxes.
        
        Args:
            elements_data: DOM elements data from extract_dom_elements
            duration_ms: Duration of highlighting in milliseconds
            highlight_color: Color to use for highlighting
        """
        script = """(arg) => {
            const { elementsData, options } = arg;
            const { durationMs, highlightColor } = options;
            
            // Create a container for highlights
            const container = document.createElement('div');
            container.style.position = 'fixed';
            container.style.top = '0';
            container.style.left = '0';
            container.style.width = '100%';
            container.style.height = '100%';
            container.style.pointerEvents = 'none';
            container.style.zIndex = '10000';
            document.body.appendChild(container);
            
            function createHighlight(boundingBox, elementType) {
                const highlight = document.createElement('div');
                highlight.style.position = 'absolute';
                highlight.style.left = boundingBox.left + 'px';
                highlight.style.top = boundingBox.top + 'px';
                highlight.style.width = boundingBox.width + 'px';
                highlight.style.height = boundingBox.height + 'px';
                highlight.style.backgroundColor = highlightColor;
                highlight.style.border = '2px solid red';
                highlight.style.boxSizing = 'border-box';
                highlight.style.pointerEvents = 'none';
                
                // Add label
                const label = document.createElement('div');
                label.textContent = elementType;
                label.style.position = 'absolute';
                label.style.top = '-20px';
                label.style.left = '0';
                label.style.backgroundColor = 'black';
                label.style.color = 'white';
                label.style.padding = '2px 5px';
                label.style.fontSize = '12px';
                label.style.borderRadius = '3px';
                highlight.appendChild(label);
                
                return highlight;
            }
            
            function processElement(element) {
                if (element.boundingBox && element.isVisible) {
                    const highlight = createHighlight(element.boundingBox, element.elementType);
                    container.appendChild(highlight);
                }
                
                if (element.children) {
                    for (const child of element.children) {
                        processElement(child);
                    }
                }
            }
            
            processElement(elementsData);
            
            // Remove highlights after duration
            setTimeout(() => {
                document.body.removeChild(container);
            }, durationMs);
        }"""
        
        options = {
            "durationMs": duration_ms,
            "highlightColor": highlight_color
        }
        
        arg = {"elementsData": elements_data, "options": options}
        await self.page.aevaluate(script, arg)
    
    async def extract_interactive_elements(self) -> List[Dict[str, Any]]:
        """
        Extract only interactive elements like buttons, links, inputs, etc.
        
        Returns:
            List of interactive elements with their properties
        """
        script = """() => {
            const interactiveSelectors = [
                'a', 'button', 'input', 'select', 'textarea',
                '[role="button"]', '[role="link"]', '[role="checkbox"]',
                '[role="radio"]', '[role="menuitem"]', '[role="tab"]',
                '[role="combobox"]', '[role="slider"]', '[role="switch"]'
            ];
            
            const elements = document.querySelectorAll(interactiveSelectors.join(','));
            const result = [];
            
            elements.forEach(element => {
                const rect = element.getBoundingClientRect();
                const style = window.getComputedStyle(element);
                const isVisible = !(style.display === 'none' || 
                                   style.visibility === 'hidden' || 
                                   style.opacity === '0' ||
                                   (rect.width === 0 && rect.height === 0));
                
                if (isVisible) {
                    let elementType = element.tagName.toLowerCase();
                    if (element.hasAttribute('role')) {
                        elementType = element.getAttribute('role');
                    } else if (elementType === 'input') {
                        elementType = `input-${element.type || 'text'}`;
                    }
                    
                    const elementData = {
                        elementType,
                        tagName: element.tagName.toLowerCase(),
                        id: element.id || undefined,
                        className: element.className || undefined,
                        textContent: element.textContent.trim(),
                        boundingBox: {
                            x: rect.x,
                            y: rect.y,
                            width: rect.width,
                            height: rect.height,
                            top: rect.top,
                            right: rect.right,
                            bottom: rect.bottom,
                            left: rect.left
                        },
                        attributes: {}
                    };
                    
                    // Add relevant attributes
                    for (const attr of element.attributes) {
                        elementData.attributes[attr.name] = attr.value;
                    }
                    
                    result.push(elementData);
                }
            });
            
            return result;
        }"""
        
        return await self.page.aevaluate(script)
    
    async def inject_dom_observer(self) -> None:
        """
        Inject a DOM observer script that will track DOM changes.
        This is useful for monitoring dynamic content changes.
        """
        script = """() => {
            // Remove any existing observer
            if (window.__allysonObserver) {
                window.__allysonObserver.disconnect();
                delete window.__allysonObserver;
            }
            
            // Create a log for DOM changes
            window.__allysonDOMChanges = [];
            
            // Create a new observer
            window.__allysonObserver = new MutationObserver(mutations => {
                for (const mutation of mutations) {
                    const change = {
                        type: mutation.type,
                        timestamp: new Date().toISOString()
                    };
                    
                    if (mutation.type === 'childList') {
                        change.addedNodes = Array.from(mutation.addedNodes).map(node => ({
                            nodeType: node.nodeType,
                            tagName: node.tagName ? node.tagName.toLowerCase() : null,
                            id: node.id || undefined,
                            className: node.className || undefined
                        }));
                        
                        change.removedNodes = Array.from(mutation.removedNodes).map(node => ({
                            nodeType: node.nodeType,
                            tagName: node.tagName ? node.tagName.toLowerCase() : null,
                            id: node.id || undefined,
                            className: node.className || undefined
                        }));
                    } else if (mutation.type === 'attributes') {
                        change.attributeName = mutation.attributeName;
                        change.oldValue = mutation.oldValue;
                        change.newValue = mutation.target.getAttribute(mutation.attributeName);
                    } else if (mutation.type === 'characterData') {
                        change.oldValue = mutation.oldValue;
                        change.newValue = mutation.target.textContent;
                    }
                    
                    window.__allysonDOMChanges.push(change);
                    
                    // Keep the log at a reasonable size
                    if (window.__allysonDOMChanges.length > 1000) {
                        window.__allysonDOMChanges.shift();
                    }
                }
            });
            
            // Start observing
            window.__allysonObserver.observe(document.body, {
                childList: true,
                attributes: true,
                characterData: true,
                subtree: true,
                attributeOldValue: true,
                characterDataOldValue: true
            });
            
            return "DOM observer injected successfully";
        }"""
        
        return await self.page.aevaluate(script)
    
    async def get_dom_changes(self) -> List[Dict[str, Any]]:
        """
        Get the DOM changes that have been recorded by the injected observer.
        
        Returns:
            List of DOM changes
        """
        script = """() => {
            if (!window.__allysonDOMChanges) {
                return [];
            }
            
            const changes = [...window.__allysonDOMChanges];
            window.__allysonDOMChanges = [];
            return changes;
        }"""
        
        return await self.page.aevaluate(script)
    
    async def get_page_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the current page.
        
        Returns:
            Dictionary containing page metadata
        """
        script = """() => {
            return {
                title: document.title,
                url: window.location.href,
                domain: window.location.hostname,
                viewport: {
                    width: window.innerWidth,
                    height: window.innerHeight
                },
                meta: Array.from(document.querySelectorAll('meta')).map(meta => ({
                    name: meta.getAttribute('name'),
                    property: meta.getAttribute('property'),
                    content: meta.getAttribute('content')
                })).filter(meta => meta.name || meta.property),
                scripts: Array.from(document.querySelectorAll('script[src]')).map(script => script.src),
                stylesheets: Array.from(document.querySelectorAll('link[rel="stylesheet"]')).map(link => link.href)
            };
        }"""
        
        return await self.page.aevaluate(script)
    
    async def extract_dom_for_ai(self, include_metadata: bool = True, 
                               include_interactive_only: bool = False) -> Dict[str, Any]:
        """
        Extract DOM information specifically formatted for AI consumption.
        
        Args:
            include_metadata: Whether to include page metadata
            include_interactive_only: Whether to only include interactive elements
            
        Returns:
            Dictionary containing DOM information formatted for AI
        """
        result = {}
        
        if include_metadata:
            result["metadata"] = await self.get_page_metadata()
        
        if include_interactive_only:
            result["interactiveElements"] = await self.extract_interactive_elements()
        else:
            result["domTree"] = await self.extract_dom_elements(max_depth=5)
        
        return result
    
    async def screenshot_with_annotations(self, path: str, elements: List[Dict[str, Any]] = None, 
                                        full_page: bool = False, show_element_ids: bool = True,
                                        box_color: str = "red", text_color: str = "white",
                                        line_width: int = 2, font_size: int = 12) -> Dict[str, str]:
        """
        Take a screenshot with annotated bounding boxes for elements.
        
        Args:
            path: Path to save the screenshot
            elements: List of elements to annotate (if None, will extract interactive elements)
            full_page: Whether to take a screenshot of the full page
            show_element_ids: Whether to show element IDs in the annotations
            box_color: Color of the bounding box
            text_color: Color of the text
            line_width: Width of the bounding box line
            font_size: Size of the font for annotations
            
        Returns:
            Dictionary with paths to both annotated and clean screenshots
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Take a clean screenshot first
        clean_path = path
        await self.page.ascreenshot(clean_path, full_page=full_page)
        
        # If no elements provided, extract interactive elements
        if elements is None:
            elements = await self.extract_interactive_elements()
        
        # Generate annotated screenshot path
        base, ext = os.path.splitext(path)
        annotated_path = f"{base}_annotated{ext}"
        
        # Get the screenshot as bytes
        screenshot_bytes = await self.page.ascreenshot(None, full_page=full_page)
        
        # Open the image with PIL
        img = Image.open(io.BytesIO(screenshot_bytes))
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("Arial", font_size)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw bounding boxes and labels
        for i, element in enumerate(elements):
            if 'boundingBox' in element and element.get('isVisible', True):
                box = element['boundingBox']
                
                # Draw rectangle
                draw.rectangle(
                    [(box['left'], box['top']), (box['right'], box['bottom'])],
                    outline=box_color,
                    width=line_width
                )
                
                # Draw label with element ID
                if show_element_ids:
                    label = f"#{i+1}"
                    
                    # Get text size to create background
                    # PIL's text size methods changed in different versions
                    # For newer versions of PIL
                    if hasattr(font, "getbbox"):
                        bbox = font.getbbox(label)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                    # For older versions of PIL
                    elif hasattr(draw, "textsize"):
                        text_width, text_height = draw.textsize(label, font=font)
                    # Fallback
                    else:
                        text_width, text_height = 20, 15  # Reasonable defaults
                    
                    # Draw text background
                    draw.rectangle(
                        [(box['left'], box['top'] - text_height - 4), (box['left'] + text_width + 4, box['top'])],
                        fill=box_color
                    )
                    
                    # Draw text
                    draw.text(
                        (box['left'] + 2, box['top'] - text_height - 2),
                        label,
                        fill=text_color,
                        font=font
                    )
        
        # Save the annotated image
        img.save(annotated_path)
        
        return {
            "clean": clean_path,
            "annotated": annotated_path
        }
    
    async def screenshot_with_element_map(self, path: str, elements: List[Dict[str, Any]] = None,
                                        full_page: bool = False) -> Dict[str, Any]:
        """
        Take a screenshot and return a mapping of element IDs to their properties.
        
        Args:
            path: Path to save the screenshot
            elements: List of elements to include in the map (if None, will extract interactive elements)
            full_page: Whether to take a screenshot of the full page
            
        Returns:
            Dictionary with screenshot path and element map
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Take a screenshot
        await self.page.ascreenshot(path, full_page=full_page)
        
        # If no elements provided, extract interactive elements
        if elements is None:
            elements = await self.extract_interactive_elements()
        
        # Create element map
        element_map = []
        for i, element in enumerate(elements):
            if 'boundingBox' in element and element.get('isVisible', True):
                element_info = {
                    "id": i + 1,
                    "tagName": element.get('tagName', ''),
                    "elementType": element.get('elementType', ''),
                    "textContent": element.get('textContent', ''),
                    "boundingBox": element.get('boundingBox', {}),
                }
                
                # Add attributes if available
                if 'attributes' in element:
                    element_info['attributes'] = element['attributes']
                
                element_map.append(element_info)
        
        return {
            "screenshot": path,
            "elementMap": element_map
        } 