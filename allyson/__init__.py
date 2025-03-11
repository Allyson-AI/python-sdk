"""
Allyson: AI-powered web browser automation using Playwright.
"""

__version__ = "0.1.8"

from allyson.browser import Browser
from allyson.page import Page
from allyson.element import Element
from allyson.agent import Agent
from allyson.dom_extractor import DOMExtractor
from allyson.agent_loop import AgentLoop
from allyson.tools import Tool, ToolType
from allyson.plan import Plan, PlanStep

__all__ = ["Browser", "Page", "Element", "Agent", "DOMExtractor", "AgentLoop", "Tool", "ToolType", "Plan", "PlanStep"] 