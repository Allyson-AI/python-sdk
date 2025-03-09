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
    agent.chat_completion = MagicMock()
    return agent


@pytest.fixture
def agent_loop(mock_browser, mock_agent, tmp_path):
    """Create an agent loop instance with mock browser and agent."""
    screenshot_dir = str(tmp_path / "screenshots")
    plan_dir = str(tmp_path / "plans")
    
    for directory in [screenshot_dir, plan_dir]:
        os.makedirs(directory, exist_ok=True)
    
    return AgentLoop(
        browser=mock_browser,
        agent=mock_agent,
        max_steps=5,
        screenshot_dir=screenshot_dir,
        plan_dir=plan_dir,
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
async def test_create_plan(agent_loop, mock_agent):
    """Test creating a plan for a task."""
    # Mock the agent response
    mock_agent.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": """
# Plan for: Test task

## Steps:
- [ ] Step 1
- [ ] Step 2
  - [ ] Substep 2.1
  - [ ] Substep 2.2
- [ ] Step 3
"""
                }
            }
        ]
    }
    
    # Create a plan
    plan = await agent_loop._create_plan("Test task")
    
    # Check the plan
    assert "Plan for: Test task" in plan
    assert "- [ ] Step 1" in plan
    assert "- [ ] Step 2" in plan
    assert "  - [ ] Substep 2.1" in plan
    
    # Check that the agent was called
    mock_agent.chat_completion.assert_called_once()
    
    # Check that the plan was saved to a file
    assert agent_loop.state.plan == plan
    assert agent_loop.state.plan_path is not None
    assert os.path.exists(agent_loop.state.plan_path)


@pytest.mark.asyncio
async def test_update_plan(agent_loop, mock_agent):
    """Test updating a plan with a completed step."""
    # Set up a plan
    agent_loop.state.plan = """
# Plan for: Test task

## Steps:
- [ ] Step 1
- [ ] Step 2
  - [ ] Substep 2.1
  - [ ] Substep 2.2
- [ ] Step 3
"""
    
    # Create a temporary plan file
    plan_path = os.path.join(agent_loop.plan_dir, "test_plan.md")
    with open(plan_path, "w") as f:
        f.write(agent_loop.state.plan)
    
    agent_loop.state.plan_path = plan_path
    
    # Mock the agent response
    mock_agent.chat_completion.return_value = {
        "choices": [
            {
                "message": {
                    "content": """
# Plan for: Test task

## Steps:
- [x] Step 1
- [ ] Step 2
  - [ ] Substep 2.1
  - [ ] Substep 2.2
- [ ] Step 3
"""
                }
            }
        ]
    }
    
    # Update the plan
    updated_plan = await agent_loop._update_plan("Step 1")
    
    # Check the updated plan
    assert "- [x] Step 1" in updated_plan
    assert "- [ ] Step 2" in updated_plan
    
    # Check that the agent was called
    mock_agent.chat_completion.assert_called_once()
    
    # Check that the plan file was updated
    with open(plan_path, "r") as f:
        file_content = f.read()
    
    assert "- [x] Step 1" in file_content


@pytest.mark.asyncio
async def test_run_with_done_response(agent_loop, mock_agent):
    """Test running the agent loop with a done response."""
    # Mock the agent response for creating a plan
    mock_agent.chat_completion.side_effect = [
        # First response: create plan
        {
            "choices": [
                {
                    "message": {
                        "content": """
# Plan for: Test task

## Steps:
- [ ] Step 1
- [ ] Step 2
"""
                    }
                }
            ]
        },
        # Second response: done action
        {
            "choices": [
                {
                    "message": {
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
                }
            ]
        }
    ]
    
    # Run the agent loop
    memory = await agent_loop.run("Test task")
    
    # Check the memory
    assert len(memory) > 0
    assert memory[0]["role"] == "user"
    assert memory[0]["content"] == "Test task"
    
    # Check that the agent was called twice (once for plan, once for action)
    assert mock_agent.chat_completion.call_count == 2


@pytest.mark.asyncio
async def test_run_with_action_response(agent_loop, mock_agent):
    """Test running the agent loop with an action response."""
    # Mock the agent responses
    mock_agent.chat_completion.side_effect = [
        # First response: create plan
        {
            "choices": [
                {
                    "message": {
                        "content": """
# Plan for: Go to example.com

## Steps:
- [ ] Navigate to example.com
- [ ] Complete the task
"""
                    }
                }
            ]
        },
        # Second response: goto action
        {
            "choices": [
                {
                    "message": {
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
                }
            ]
        },
        # Third response: update plan
        {
            "choices": [
                {
                    "message": {
                        "content": """
# Plan for: Go to example.com

## Steps:
- [x] Navigate to example.com
- [ ] Complete the task
"""
                    }
                }
            ]
        },
        # Fourth response: done action
        {
            "choices": [
                {
                    "message": {
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
                }
            ]
        }
    ]
    
    # Run the agent loop
    memory = await agent_loop.run("Go to example.com")
    
    # Check the memory
    assert len(memory) > 0
    
    # Check that the agent was called four times (plan, goto, update plan, done)
    assert mock_agent.chat_completion.call_count == 4


@pytest.mark.asyncio
async def test_run_max_steps(agent_loop, mock_agent):
    """Test running the agent loop with maximum steps."""
    # Set max steps to 3
    agent_loop.max_steps = 3
    
    # Create a response for the plan creation
    plan_response = {
        "choices": [
            {
                "message": {
                    "content": """
# Plan for: Go to example.com

## Steps:
- [ ] Navigate to example.com
- [ ] Do something else
"""
                }
            }
        ]
    }
    
    # Create a response for the goto action
    goto_response = {
        "choices": [
            {
                "message": {
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
            }
        ]
    }
    
    # Create a response for the plan update
    plan_update_response = {
        "choices": [
            {
                "message": {
                    "content": """
# Plan for: Go to example.com

## Steps:
- [x] Navigate to example.com
- [ ] Do something else
"""
                }
            }
        ]
    }
    
    # Instead of using side_effect with a fixed list, we'll use a custom function
    # that returns different responses based on the call count
    call_count = 0
    
    def get_response(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        
        # First call: create plan
        if call_count == 1:
            return plan_response
        # Second call: goto action
        elif call_count == 2:
            return goto_response
        # Third call: update plan
        elif call_count == 3:
            return plan_update_response
        # Fourth call and beyond: goto action again
        else:
            return goto_response
    
    # Set the side effect to our custom function
    mock_agent.chat_completion.side_effect = get_response
    
    # Run the agent loop
    memory = await agent_loop.run("Go to example.com")
    
    # Check that the agent was called at least 3 times
    assert mock_agent.chat_completion.call_count >= 3
    
    # Check that the last message indicates max steps reached
    assert memory[-1]["role"] == "system"
    assert "maximum number of steps" in memory[-1]["content"] 