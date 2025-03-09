"""
Agent loop for executing tasks on web pages.

This module provides a loop that takes user instructions, sends them to an AI agent,
and executes the resulting actions on a web page.
"""

import asyncio
import json
import logging
import time
import os
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Type

import pydantic
from pydantic import BaseModel, Field, create_model

from allyson import Browser, DOMExtractor, Agent
from allyson.tools import Tool, ToolType, get_default_tools

logger = logging.getLogger(__name__)


class ActionStatus(str, Enum):
    """Status of an action execution."""
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"


class Action(BaseModel):
    """Base model for an action to be executed by the agent."""
    tool: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Observation from executing an action."""
    status: ActionStatus
    data: Any = None
    error: Optional[str] = None


class AgentState(BaseModel):
    """State of the agent during execution."""
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    current_url: Optional[str] = None
    page_title: Optional[str] = None
    last_observation: Optional[Observation] = None
    interactive_elements: Optional[List[Dict[str, Any]]] = None
    screenshot_path: Optional[str] = None
    pending_actions: Optional[List[Dict[str, Any]]] = None


class AgentMessage(BaseModel):
    """Message from the agent to the user."""
    message: str
    thinking: Optional[str] = None


class AgentResponse(BaseModel):
    """Response from the agent."""
    action: Optional[Action] = None
    message: Optional[AgentMessage] = None
    done: bool = False


class AgentLoop:
    """
    Agent loop for executing tasks on web pages.
    
    This class provides a loop that takes user instructions, sends them to an AI agent,
    and executes the resulting actions on a web page.
    """
    
    def __init__(
        self,
        browser: Browser,
        agent: Agent,
        tools: Optional[List[Tool]] = None,
        max_iterations: int = 10,
        screenshot_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the agent loop.
        
        Args:
            browser: Browser instance to use for the agent loop
            agent: Agent instance to use for the agent loop
            tools: Optional list of custom tools to add to the agent loop
            max_iterations: Maximum number of iterations to run the agent loop
            screenshot_dir: Directory to save screenshots to
            verbose: Whether to print verbose output
        """
        self.browser = browser
        self.agent = agent
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Create screenshot directory if it doesn't exist
        self.screenshot_dir = screenshot_dir
        if screenshot_dir and not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)
        
        # Initialize state
        self.state = AgentState()
        
        # Create DOM extractor if it doesn't exist
        if not hasattr(self.browser, '_dom_extractor'):
            self.browser._dom_extractor = DOMExtractor(self.browser._page)
        
        # Initialize tools
        self.tools = {}
        self._register_default_tools()
        
        # Register custom tools
        if tools:
            for tool in tools:
                self.register_tool(tool)
    
    def _register_default_tools(self):
        """Register default tools for the agent loop."""
        default_tools = get_default_tools(self.browser)
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: Tool):
        """
        Register a tool with the agent loop.
        
        Args:
            tool: Tool to register
        """
        if tool.name in self.tools:
            logger.warning(f"Tool {tool.name} already registered, overwriting")
        
        self.tools[tool.name] = tool
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Get the schema for all registered tools.
        
        Returns:
            List of tool schemas
        """
        tools_schema = []
        
        for tool_name, tool in self.tools.items():
            # Create a schema for the tool
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters_schema,
                        "required": list(tool.parameters_schema.keys())
                    }
                }
            }
            
            tools_schema.append(schema)
        
        return tools_schema
    
    async def _update_state(self):
        """Update the agent state with the current page state."""
        # Get the current URL and title
        self.state.current_url = await self.browser._page.aevaluate("window.location.href")
        self.state.page_title = await self.browser._page.aevaluate("document.title")
        
        # Extract interactive elements
        dom_extractor = self.browser._dom_extractor
        self.state.interactive_elements = await dom_extractor.extract_interactive_elements()
        
        # Take a screenshot
        if self.screenshot_dir:
            timestamp = int(time.time())
            screenshot_path = os.path.join(self.screenshot_dir, f"screenshot_{timestamp}.png")
            
            # Take a screenshot with annotations
            result = await dom_extractor.screenshot_with_annotations(
                path=screenshot_path,
                elements=self.state.interactive_elements,
                show_element_ids=True
            )
            
            self.state.screenshot_path = result["annotated"]
    
    async def _execute_action(self, action: Action) -> Observation:
        """
        Execute an action and return the observation.
        
        Args:
            action: Action to execute
            
        Returns:
            Observation from executing the action
        """
        # Check if the tool exists
        if action.tool not in self.tools:
            return Observation(
                status=ActionStatus.ERROR,
                error=f"Tool {action.tool} not found"
            )
        
        # Get the tool
        tool = self.tools[action.tool]
        
        try:
            # Execute the tool function with the parameters
            result = tool.function(**action.parameters)
            
            # If the result is a coroutine, await it
            if asyncio.iscoroutine(result):
                result = await result
            
            # Check if the task is done
            if action.tool == "done":
                return Observation(
                    status=ActionStatus.SUCCESS,
                    data=result
                )
            
            # Update the state
            await self._update_state()
            
            return Observation(
                status=ActionStatus.SUCCESS,
                data=result
            )
        except Exception as e:
            logger.exception(f"Error executing action {action.tool}: {e}")
            return Observation(
                status=ActionStatus.ERROR,
                error=str(e)
            )
    
    async def run(self, task: str) -> List[Dict[str, Any]]:
        """
        Run the agent loop with a task.
        
        Args:
            task: Task to run
            
        Returns:
            Memory of the agent loop
        """
        # Initialize the state
        self.state = AgentState()
        
        # Add the task to the memory
        self.state.memory.append({
            "role": "user",
            "content": task
        })
        
        # Update the state
        await self._update_state()
        
        # Run the loop
        iteration = 0
        done = False
        
        while iteration < self.max_iterations and not done:
            iteration += 1
            
            if self.verbose:
                logger.info(f"Iteration {iteration}/{self.max_iterations}")
            
            # Check if there are pending actions
            if hasattr(self.state, "pending_actions") and self.state.pending_actions:
                # Get the next action from pending actions
                action_data = self.state.pending_actions.pop(0)
                tool_name = action_data.get("tool")
                parameters = action_data.get("parameters", {})
                
                # Check if the action is "done"
                if tool_name == "done":
                    if self.verbose:
                        logger.info(f"Task is done from pending action: {parameters.get('message', 'No message')}")
                    
                    # Add the observation to the memory
                    self.state.memory.append({
                        "role": "system",
                        "content": json.dumps({
                            "observation": {
                                "status": ActionStatus.SUCCESS,
                                "data": {"done": True, "message": parameters.get("message", "Task completed")},
                                "error": None
                            }
                        })
                    })
                    
                    done = True
                    continue
                
                # Create the action
                action = Action(
                    tool=tool_name,
                    parameters=parameters
                )
                
                # Add the action to the memory
                self.state.memory.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "action": {
                            "tool": action.tool,
                            "parameters": action.parameters
                        },
                        "thinking": "Executing pending action"
                    })
                })
                
                # Execute the action
                observation = await self._execute_action(action)
                
                # Add the observation to the memory
                self.state.memory.append({
                    "role": "system",
                    "content": json.dumps({
                        "observation": {
                            "status": observation.status,
                            "data": observation.data,
                            "error": observation.error
                        }
                    })
                })
                
                # Update the last observation
                self.state.last_observation = observation
                
                # Continue to the next iteration
                continue
            
            # Get the agent response
            response = await self._get_agent_response()
            
            # Check if the agent is done
            if response.done:
                if self.verbose:
                    logger.info(f"Agent is done: {response.message.message if response.message else 'No message'}")
                
                # Add the message to the memory
                if response.message:
                    self.state.memory.append({
                        "role": "assistant",
                        "content": response.message.message
                    })
                
                done = True
                continue
            
            # Execute the action
            if response.action:
                if self.verbose:
                    logger.info(f"Executing action: {response.action.tool} with parameters {response.action.parameters}")
                
                # Add the action to the memory
                self.state.memory.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "action": {
                            "tool": response.action.tool,
                            "parameters": response.action.parameters
                        },
                        "thinking": response.message.thinking if response.message else None
                    })
                })
                
                # Execute the action
                observation = await self._execute_action(response.action)
                
                # Check if the task is done
                if response.action.tool == "done":
                    if self.verbose:
                        logger.info(f"Task is done: {observation.data.get('message', 'No message')}")
                    
                    # Add the observation to the memory
                    self.state.memory.append({
                        "role": "system",
                        "content": json.dumps({
                            "observation": {
                                "status": observation.status,
                                "data": observation.data,
                                "error": observation.error
                            }
                        })
                    })
                    
                    done = True
                    continue
                
                # Add the observation to the memory
                self.state.memory.append({
                    "role": "system",
                    "content": json.dumps({
                        "observation": {
                            "status": observation.status,
                            "data": observation.data,
                            "error": observation.error
                        }
                    })
                })
                
                # Update the last observation
                self.state.last_observation = observation
            
            # Add the message to the memory if there is one
            if response.message and not response.action:
                self.state.memory.append({
                    "role": "assistant",
                    "content": response.message.message
                })
        
        # Check if we reached the maximum number of iterations
        if iteration >= self.max_iterations and not done:
            logger.warning(f"Reached maximum number of iterations ({self.max_iterations})")
            
            # Add a message to the memory
            self.state.memory.append({
                "role": "system",
                "content": f"Reached maximum number of iterations ({self.max_iterations})"
            })
        
        return self.state.memory
    
    async def _get_agent_response(self) -> AgentResponse:
        """
        Get a response from the agent.
        
        Returns:
            Agent response
        """
        # Create the messages for the agent
        messages = [
            {"role": "system", "content": self._get_system_message()}
        ]
        
        # Add the memory to the messages
        for message in self.state.memory:
            messages.append(message)
        
        # Get the response from the agent
        response = self.agent.chat_completion(
            messages=messages,
            tools=self.get_tools_schema()
        )
        
        # Debug log the response
        if self.verbose:
            logger.info(f"Agent response: {json.dumps(response, indent=2)}")
        
        # Parse the response
        return self._parse_agent_response(response)
    
    def _get_system_message(self) -> str:
        """
        Get the system message for the agent.
        
        Returns:
            System message
        """
        # Create the system message
        system_message = """
You are an AI assistant that helps users automate web browsing tasks. You are given a task to complete, and you can use a set of tools to interact with the web page.

Your goal is to complete the task by using the available tools. You should think step by step and explain your reasoning.

The current state of the web page is provided to you, including:
- The current URL
- The page title
- Interactive elements on the page (with their IDs, types, and text content)

Each interactive element has an ID number that you can use to interact with it. The IDs start at 1.

Available tools:
- goto: Navigate to a URL
- click: Click on an element by its ID number
- type: Type text into an element by its ID number
- enter: Press the Enter key to submit a form or activate the default action
- scroll: Scroll the page in a direction
- done: Mark the task as complete with a final message

You can use these tools in two ways:

1. Single action: Respond with a single tool call and its parameters.
```json
{
  "action": {
    "tool": "click",
    "parameters": {
      "element_id": 3
    }
  },
  "thinking": "I need to click on the search button which is element ID 3."
}
```

2. Multiple actions: Chain multiple actions together to be executed in sequence.
```json
{
  "actions": [
    {
      "tool": "type",
      "parameters": {
        "element_id": 2,
        "text": "search query"
      }
    },
    {
      "tool": "enter",
      "parameters": {}
    }
  ],
  "thinking": "I'll type the search query and press Enter to submit it."
}
```

When you want to provide a message to the user without using a tool, respond with just your message.

When you have completed the task, use the "done" tool with a message explaining what you did.

Remember to:
1. Analyze the page state carefully
2. Choose the appropriate tool for each step
3. Provide clear explanations of your actions
4. Complete the task efficiently
5. Chain actions together when it makes sense to do so
"""
        
        # Add the current state
        state_info = f"""
Current state:
- URL: {self.state.current_url}
- Title: {self.state.page_title}
- Interactive elements:
"""
        
        # Add the interactive elements
        if self.state.interactive_elements:
            for i, element in enumerate(self.state.interactive_elements, 1):
                element_type = element.get("elementType", "unknown")
                text = element.get("textContent", "").strip()
                text = text[:50] + "..." if len(text) > 50 else text
                state_info += f"  {i}. {element_type}: {text}\n"
        else:
            state_info += "  No interactive elements found\n"
        
        # Add the screenshot path if available
        if self.state.screenshot_path:
            state_info += f"\nA screenshot of the page with annotated elements is available at: {self.state.screenshot_path}\n"
        
        # Add the action history
        if len(self.state.memory) > 1:  # Skip the initial user message
            action_history = "\nAction history:\n"
            
            for i, message in enumerate(self.state.memory[1:], 1):
                role = message.get("role", "")
                content = message.get("content", "")
                
                if role == "assistant" and content and content.startswith("{"):
                    try:
                        data = json.loads(content)
                        if "action" in data:
                            action = data["action"]
                            tool = action.get("tool", "unknown")
                            params = action.get("parameters", {})
                            action_history += f"  {i}. Used tool: {tool} with parameters: {params}\n"
                    except:
                        pass
                elif role == "system" and content and content.startswith("{"):
                    try:
                        data = json.loads(content)
                        if "observation" in data:
                            observation = data["observation"]
                            status = observation.get("status", "unknown")
                            data_result = observation.get("data", {})
                            error = observation.get("error")
                            
                            if status == "success":
                                action_history += f"  {i}. Result: Success - {data_result}\n"
                            else:
                                action_history += f"  {i}. Result: Error - {error}\n"
                    except:
                        pass
            
            state_info += action_history
        
        return system_message + state_info
    
    def _parse_agent_response(self, response: Dict[str, Any]) -> AgentResponse:
        """
        Parse the response from the agent.
        
        Args:
            response: Response from the agent
            
        Returns:
            Parsed agent response
        """
        # Get the message from the response
        message = response["choices"][0]["message"]
        content = message.get("content", "")
        
        # Check if the response has tool calls
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            # Get the first tool call
            tool_call = tool_calls[0]
            
            # Get the tool name and parameters
            tool_name = tool_call["function"]["name"]
            
            try:
                parameters = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                logger.error(f"Error parsing tool parameters: {tool_call['function']['arguments']}")
                parameters = {}
            
            # Check if the tool is "done"
            if tool_name == "done":
                return AgentResponse(
                    done=True,
                    message=AgentMessage(
                        message=parameters.get("message", "Task completed"),
                        thinking=response.get("thinking")
                    )
                )
            
            # Create the action
            action = Action(
                tool=tool_name,
                parameters=parameters
            )
            
            # Create the message
            message_obj = None
            if content:
                message_obj = AgentMessage(
                    message=content,
                    thinking=response.get("thinking")
                )
            
            return AgentResponse(
                action=action,
                message=message_obj
            )
        
        # Check if the content is a JSON string with an action or actions
        if content and content.strip().startswith("{"):
            try:
                content_json = json.loads(content)
                
                # Check for a single action
                if "action" in content_json:
                    action_data = content_json["action"]
                    tool_name = action_data.get("tool")
                    parameters = action_data.get("parameters", {})
                    
                    # Check if the tool is "done"
                    if tool_name == "done":
                        return AgentResponse(
                            done=True,
                            message=AgentMessage(
                                message=parameters.get("message", "Task completed"),
                                thinking=content_json.get("thinking")
                            )
                        )
                    
                    # Create the action
                    action = Action(
                        tool=tool_name,
                        parameters=parameters
                    )
                    
                    # Create the message
                    message_obj = None
                    if "thinking" in content_json:
                        message_obj = AgentMessage(
                            message="",
                            thinking=content_json.get("thinking")
                        )
                    
                    return AgentResponse(
                        action=action,
                        message=message_obj
                    )
                
                # Check for multiple actions
                if "actions" in content_json:
                    actions_data = content_json["actions"]
                    if actions_data and isinstance(actions_data, list) and len(actions_data) > 0:
                        # Get the first action
                        action_data = actions_data[0]
                        tool_name = action_data.get("tool")
                        parameters = action_data.get("parameters", {})
                        
                        # Check if the tool is "done"
                        if tool_name == "done":
                            return AgentResponse(
                                done=True,
                                message=AgentMessage(
                                    message=parameters.get("message", "Task completed"),
                                    thinking=content_json.get("thinking")
                                )
                            )
                        
                        # Create the action
                        action = Action(
                            tool=tool_name,
                            parameters=parameters
                        )
                        
                        # Store the remaining actions in the state for later processing
                        self.state.pending_actions = actions_data[1:]
                        
                        # Create the message
                        message_obj = None
                        if "thinking" in content_json:
                            message_obj = AgentMessage(
                                message="",
                                thinking=content_json.get("thinking")
                            )
                        
                        return AgentResponse(
                            action=action,
                            message=message_obj
                        )
            except json.JSONDecodeError:
                # Not a valid JSON, treat as regular message
                pass
        
        # If there's no tool call, just return the message
        if content:
            return AgentResponse(
                message=AgentMessage(
                    message=content,
                    thinking=response.get("thinking")
                )
            )
        
        # If there's no content, return an empty response
        return AgentResponse() 