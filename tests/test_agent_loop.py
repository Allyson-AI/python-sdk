"""
Tests for the agent loop module.
"""

import asyncio
import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from allyson import Browser, Agent
from allyson.agent_loop import AgentLoop, Action, Observation, ActionStatus
from allyson.tools import Tool, ToolType


@pytest.fixture
def mock_browser():
    """Create a mock browser instance."""
    browser = MagicMock(spec=Browser)
    browser._page = AsyncMock()
    browser._page.url = "https://example.com"
    browser._page.atitle = AsyncMock(return_value="Example Domain")
    
    # Mock DOM extractor
    browser._dom_extractor = AsyncMock()
    browser._dom_extractor.extract_interactive_elements = AsyncMock(return_value=[
        {
            "elementType": "button",
            "textContent": "Click me",
            "boundingBox": {"x": 100, "y": 100, "width": 100, "height": 50}
        },
        {
            "elementType": "input",
            "textContent": "",
            "boundingBox": {"x": 100, "y": 200, "width": 200, "height": 30}
        }
    ])
    browser._dom_extractor.screenshot_with_annotations = AsyncMock(return_value={
        "clean": "screenshot.png",
        "annotated": "screenshot_annotated.png"
    })
    
    return browser


@pytest.fixture
def mock_agent():
    """Create a mock agent instance."""
    agent = MagicMock(spec=Agent)
    agent.achat = AsyncMock()
    return agent


@pytest.fixture
def agent_loop(mock_browser, mock_agent, tmp_path):
    """Create an agent loop instance with mock browser and agent."""
    screenshot_dir = str(tmp_path / "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    
    return AgentLoop(
        browser=mock_browser,
        agent=mock_agent,
        max_iterations=5,
        screenshot_dir=screenshot_dir,
        verbose=False
    )


def test_register_tool(agent_loop):
    """Test registering a custom tool."""
    # Create a custom tool
    tool = Tool(
        name="test_tool",
        description="Test tool",
        type=ToolType.CUSTOM,
        parameters_schema={
            "param1": {"type": "string", "description": "Parameter 1"}
        },
        function=lambda param1: {"result": param1}
    )
    
    # Register the tool
    agent_loop.register_tool(tool)
    
    # Check if the tool was registered
    assert "test_tool" in agent_loop.tools
    assert agent_loop.tools["test_tool"] == tool


def test_get_tools_schema(agent_loop):
    """Test getting the tools schema."""
    # Create a custom tool
    tool = Tool(
        name="test_tool",
        description="Test tool",
        type=ToolType.CUSTOM,
        parameters_schema={
            "param1": {"type": "string", "description": "Parameter 1"}
        },
        function=lambda param1: {"result": param1}
    )
    
    # Register the tool
    agent_loop.register_tool(tool)
    
    # Get the tools schema
    schema = agent_loop.get_tools_schema()
    
    # Check if the schema is correct
    assert len(schema) > 0
    
    # Find our custom tool in the schema
    test_tool_schema = None
    for tool_schema in schema:
        if tool_schema["function"]["name"] == "test_tool":
            test_tool_schema = tool_schema
            break
    
    assert test_tool_schema is not None
    assert test_tool_schema["function"]["description"] == "Test tool"
    assert "param1" in test_tool_schema["function"]["parameters"]["properties"]


@pytest.mark.asyncio
async def test_execute_action_unknown_tool(agent_loop):
    """Test executing an action with an unknown tool."""
    action = Action(tool="unknown_tool", parameters={})
    observation = await agent_loop._execute_action(action)
    
    assert observation.status == ActionStatus.ERROR
    assert "not found" in observation.error


@pytest.mark.asyncio
async def test_execute_action_success(agent_loop):
    """Test executing an action successfully."""
    # Create a test tool
    tool = Tool(
        name="test_tool",
        description="Test tool",
        type=ToolType.CUSTOM,
        parameters_schema={
            "param1": {"type": "string", "description": "Parameter 1"}
        },
        function=lambda param1: {"result": param1}
    )
    
    # Register the tool
    agent_loop.register_tool(tool)
    
    # Create an action
    action = Action(tool="test_tool", parameters={"param1": "test_value"})
    
    # Execute the action
    observation = await agent_loop._execute_action(action)
    
    # Check the observation
    assert observation.status == ActionStatus.SUCCESS
    assert observation.data == {"result": "test_value"}


@pytest.mark.asyncio
async def test_execute_action_error(agent_loop):
    """Test executing an action that raises an error."""
    # Create a test tool that raises an error
    def error_function(param1):
        raise ValueError("Test error")
    
    tool = Tool(
        name="error_tool",
        description="Tool that raises an error",
        type=ToolType.CUSTOM,
        parameters_schema={
            "param1": {"type": "string", "description": "Parameter 1"}
        },
        function=error_function
    )
    
    # Register the tool
    agent_loop.register_tool(tool)
    
    # Create an action
    action = Action(tool="error_tool", parameters={"param1": "test_value"})
    
    # Execute the action
    observation = await agent_loop._execute_action(action)
    
    # Check the observation
    assert observation.status == ActionStatus.ERROR
    assert "Test error" in observation.error


@pytest.mark.asyncio
async def test_execute_done_action(agent_loop):
    """Test executing the done action."""
    # Create a done action
    action = Action(tool="done", parameters={"message": "Task completed"})
    
    # Execute the action
    observation = await agent_loop._execute_action(action)
    
    # Check the observation
    assert observation.status == ActionStatus.SUCCESS
    assert observation.data == {"done": True, "message": "Task completed"}


@pytest.mark.asyncio
async def test_run_with_done_response(agent_loop, mock_agent):
    """Test running the agent loop with a done response."""
    # Mock the agent response
    mock_agent.achat.return_value = {
        "tool_calls": [
            {
                "function": {
                    "name": "done",
                    "arguments": json.dumps({"message": "Task completed"})
                }
            }
        ],
        "content": "I've completed the task."
    }
    
    # Run the agent loop
    memory = await agent_loop.run("Test task")
    
    # Check the memory
    assert len(memory) > 0
    assert memory[0]["role"] == "user"
    assert memory[0]["content"] == "Test task"
    
    # Check that the agent was called
    mock_agent.achat.assert_called_once()


@pytest.mark.asyncio
async def test_run_with_action_response(agent_loop, mock_agent):
    """Test running the agent loop with an action response."""
    # Mock the agent responses
    mock_agent.achat.side_effect = [
        # First response: goto action
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "goto",
                        "arguments": json.dumps({"url": "https://example.com"})
                    }
                }
            ],
            "content": "I'll navigate to example.com"
        },
        # Second response: done action
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "done",
                        "arguments": json.dumps({"message": "Task completed"})
                    }
                }
            ],
            "content": "I've completed the task."
        }
    ]
    
    # Run the agent loop
    memory = await agent_loop.run("Go to example.com")
    
    # Check the memory
    assert len(memory) > 0
    
    # Check that the agent was called twice
    assert mock_agent.achat.call_count == 2


@pytest.mark.asyncio
async def test_run_max_iterations(agent_loop, mock_agent):
    """Test running the agent loop with maximum iterations."""
    # Mock the agent response to always return an action
    mock_agent.achat.return_value = {
        "tool_calls": [
            {
                "function": {
                    "name": "goto",
                    "arguments": json.dumps({"url": "https://example.com"})
                }
            }
        ],
        "content": "I'll navigate to example.com"
    }
    
    # Set max iterations to 3
    agent_loop.max_iterations = 3
    
    # Run the agent loop
    memory = await agent_loop.run("Go to example.com")
    
    # Check that the agent was called 3 times
    assert mock_agent.achat.call_count == 3
    
    # Check that the last message indicates max iterations reached
    assert memory[-1]["role"] == "system"
    assert "maximum number of iterations" in memory[-1]["content"] 