"""
Allyson: AI-powered web browser automation using Playwright.
"""

__version__ = "0.1.3"

from allyson.browser import Browser
from allyson.page import Page
from allyson.element import Element
from allyson.agent import Agent
from allyson.dom_extractor import DOMExtractor

__all__ = ["Browser", "Page", "Element", "Agent", "DOMExtractor"] 