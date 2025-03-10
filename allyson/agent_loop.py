"""
Agent loop for executing tasks on web pages.

This module provides a loop that takes user instructions, sends them to an AI agent,
and executes the resulting actions on a web page.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Type
import base64

import pydantic
from pydantic import BaseModel, Field, create_model

from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_fixed

from allyson.agent import Agent
from allyson.tools import Tool, ToolType, get_default_tools
from allyson.browser import Browser
from allyson import DOMExtractor
from allyson.plan import Plan, PlanStep
from allyson.prompts import (
    get_agent_system_prompt,
    get_planner_system_prompt,
    get_plan_updater_system_prompt,
    build_context_sections,
)

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
    plan: Optional[str] = None
    plan_path: Optional[str] = None
    structured_plan: Optional[Plan] = None
    # Token tracking
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    token_usage_history: List[Dict[str, Any]] = Field(default_factory=list)


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
        max_steps: int = 10,
        screenshot_dir: Optional[str] = None,
        plan_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the agent loop.

        Args:
            browser: Browser instance to use for the agent loop
            agent: Agent instance to use for the agent loop
            tools: Optional list of custom tools to add to the agent loop
            max_steps: Maximum number of steps to run the agent loop
            screenshot_dir: Directory to save screenshots to
            plan_dir: Directory to save plan files to
            verbose: Whether to print verbose output
        """
        self.browser = browser
        self.agent = agent
        self.max_steps = max_steps
        self.verbose = verbose

        # Create screenshot directory if it doesn't exist
        self.screenshot_dir = screenshot_dir
        if screenshot_dir and not os.path.exists(screenshot_dir):
            os.makedirs(screenshot_dir)

        # Create plan directory if it doesn't exist
        self.plan_dir = plan_dir
        if plan_dir and not os.path.exists(plan_dir):
            os.makedirs(plan_dir)

        # Initialize state
        self.state = AgentState()

        # Create DOM extractor if it doesn't exist
        if not hasattr(self.browser, "_dom_extractor"):
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
                        "required": list(tool.parameters_schema.keys()),
                    },
                },
            }

            tools_schema.append(schema)

        return tools_schema

    async def _update_state(self):
        """Update the agent state with the current page state."""
        try:
            # Get the current URL and title
            self.state.current_url = await self.browser._page.aevaluate(
                "window.location.href"
            )
            self.state.page_title = await self.browser._page.aevaluate("document.title")

            # Extract interactive elements
            dom_extractor = self.browser._dom_extractor
            self.state.interactive_elements = (
                await dom_extractor.extract_interactive_elements()
            )

            # Take a screenshot
            if self.screenshot_dir:
                timestamp = int(time.time())
                screenshot_path = os.path.join(
                    self.screenshot_dir, f"screenshot_{timestamp}.png"
                )

                # Take a screenshot with annotations
                result = await dom_extractor.screenshot_with_annotations(
                    path=screenshot_path,
                    elements=self.state.interactive_elements,
                    show_element_ids=True,
                )

                self.state.screenshot_path = result["annotated"]
        except Exception as e:
            # Handle navigation-related errors gracefully
            logger.warning(f"Error updating state, likely due to navigation: {e}")
            # Wait a moment for the page to stabilize after navigation
            await asyncio.sleep(0.5)
            # Try again with a simplified approach
            try:
                self.state.current_url = await self.browser._page.aurl()
                self.state.page_title = await self.browser._page.atitle()
                
                # Extract interactive elements
                dom_extractor = self.browser._dom_extractor
                self.state.interactive_elements = (
                    await dom_extractor.extract_interactive_elements()
                )
                
                # Take a screenshot
                if self.screenshot_dir:
                    timestamp = int(time.time())
                    screenshot_path = os.path.join(
                        self.screenshot_dir, f"screenshot_{timestamp}.png"
                    )
                    
                    # Take a screenshot with annotations
                    result = await dom_extractor.screenshot_with_annotations(
                        path=screenshot_path,
                        elements=self.state.interactive_elements,
                        show_element_ids=True,
                    )
                    
                    self.state.screenshot_path = result["annotated"]
            except Exception as e2:
                logger.error(f"Failed to update state even after retry: {e2}")
                # Set minimal state information
                self.state.current_url = "unknown (navigation in progress)"
                self.state.page_title = "unknown (navigation in progress)"

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
                status=ActionStatus.ERROR, error=f"Tool {action.tool} not found"
            )

        # Get the tool
        tool = self.tools[action.tool]

        try:
            # Add more detailed logging for scroll actions
            if action.tool == "scroll" and self.verbose:
                logger.info(f"Executing scroll action with parameters: {action.parameters}")
            
            # Execute the tool function with the parameters
            result = tool.function(**action.parameters)

            # If the result is a coroutine, await it
            if asyncio.iscoroutine(result):
                result = await result
                
            # Add more detailed logging for scroll results
            if action.tool == "scroll" and self.verbose:
                logger.info(f"Scroll result: {result}")

            # Check if the task is done
            if action.tool == "done":
                return Observation(status=ActionStatus.SUCCESS, data=result)

            # Update the state
            await self._update_state()

            return Observation(status=ActionStatus.SUCCESS, data=result)
        except Exception as e:
            logger.exception(f"Error executing action {action.tool}: {e}")
            return Observation(status=ActionStatus.ERROR, error=str(e))

    async def _create_plan(self, task: str) -> str:
        """
        Create a plan for completing the task.

        Args:
            task: Task to create a plan for

        Returns:
            Markdown string of the plan
        """
        if self.verbose:
            logger.info("Creating plan for task")

        # Get current page information
        current_url = self.state.current_url or "unknown page"
        page_title = self.state.page_title or "unknown title"

        # Create a system message for the planner
        system_message = get_planner_system_prompt(current_url, page_title, self.max_steps)

        # Create the messages for the agent
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Create a strategic plan for accomplishing this task: {task}"},
        ]

        # Get the response from the agent
        response = self.agent.chat_completion(messages=messages)

        # Extract the plan from the response
        plan_markdown = response["choices"][0]["message"]["content"]
        
        # Parse the markdown into a structured Plan object
        plan = Plan.from_markdown(plan_markdown)
        
        # Convert back to markdown to ensure consistent formatting
        plan_markdown = plan.to_markdown()

        # Save the plan to a file if a plan directory is specified
        if self.plan_dir:
            timestamp = int(time.time())
            plan_filename = f"plan_{timestamp}.md"
            plan_path = os.path.join(self.plan_dir, plan_filename)

            with open(plan_path, "w") as f:
                f.write(plan_markdown)

            self.state.plan_path = plan_path

        # Store the plan in the state
        self.state.plan = plan_markdown
        
        # Also store the structured plan object in the state
        self.state.structured_plan = plan

        return plan_markdown

    async def _update_plan(self, completed_step: str) -> str:
        """
        Update the plan with a completed step.

        Args:
            completed_step: Description of the completed step

        Returns:
            Updated markdown string of the plan
        """
        if not self.state.plan:
            return ""

        if self.verbose:
            logger.info(f"Updating plan: marking step as completed: {completed_step}")
            
        # If we have a structured plan, use it to update the plan
        if hasattr(self.state, 'structured_plan') and self.state.structured_plan:
            plan = self.state.structured_plan
            
            # Find the step that matches the completed step description
            step = plan.find_step_by_description(completed_step)
            
            if step:
                # Mark the step as completed
                plan.mark_step_complete(step.id)
                
                # Update the current step to the next incomplete step
                next_step = plan.get_next_incomplete_step()
                if next_step:
                    plan.current_step_id = next_step.id
                
                # Convert the updated plan to markdown
                updated_plan = plan.to_markdown()
                
                # Save the updated plan to the file if a plan path exists
                if self.state.plan_path:
                    with open(self.state.plan_path, "w") as f:
                        f.write(updated_plan)
                
                # Update the plan in the state
                self.state.plan = updated_plan
                
                return updated_plan
        
        # Fallback to the old method if we don't have a structured plan or couldn't find the step
        # Create a system message for the plan updater
        system_message = get_plan_updater_system_prompt()

        # Create the messages for the agent
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Here is the current plan:\n\n{self.state.plan}\n\nMark the following action as completed: {completed_step}\n\nReturn only the updated plan in Markdown format.",
            },
        ]

        # Get the response from the agent
        response = self.agent.chat_completion(messages=messages)

        # Extract the updated plan from the response
        updated_plan = response["choices"][0]["message"]["content"]
        
        # Parse the updated plan into a structured Plan object
        plan = Plan.from_markdown(updated_plan)
        
        # Store the structured plan in the state
        self.state.structured_plan = plan

        # Save the updated plan to the file if a plan path exists
        if self.state.plan_path:
            with open(self.state.plan_path, "w") as f:
                f.write(updated_plan)

        # Update the plan in the state
        self.state.plan = updated_plan

        return updated_plan

    async def run(self, task: str) -> List[Dict[str, Any]]:
        """
        Run the agent loop for a given task.

        Args:
            task: Task to run the agent for

        Returns:
            Memory of the conversation (for backward compatibility)
            To get token usage, use get_token_usage() method
        """
        # Reset the state
        self.state = AgentState()
        
        if self.verbose:
            logger.info(f"Starting agent loop for task: {task}")

        # Update state to get current page information before creating the plan
        await self._update_state()

        # Create a plan for the task
        plan_markdown = await self._create_plan(task)

        # Add the task and plan to the memory
        self.state.memory.append({
            "role": "user",
            "content": f"Previous History: {self.state.memory} \n\nPlan: {plan_markdown} \n\nTask: {task}"
        })
        

        # Run the agent loop
        step_count = 0
        while step_count < self.max_steps:
            step_count += 1
            if self.verbose:
                logger.info(f"Step {step_count}/{self.max_steps}")

            # Get the next step from the plan if available
            current_step = None
            if hasattr(self.state, 'structured_plan') and self.state.structured_plan:
                plan = self.state.structured_plan
                current_step = plan.get_next_incomplete_step()
                
                if current_step:
                    # Update the current step ID
                    plan.current_step_id = current_step.id
                    
                    # Add a message about the current step
                    self.state.memory.append({
                        "role": "system",
                        "content": f"Current step: {current_step.description}"
                    })

            # Get the agent's response
            agent_response = await self._get_agent_response()

            # Check if the agent is done
            if agent_response.done:
                if self.verbose:
                    logger.info("Agent is done")
                break

            # Check if the agent wants to send a message
            if agent_response.message:
                if self.verbose:
                    logger.info(f"Agent message: {agent_response.message.message}")
                self.state.memory.append({
                    "role": "assistant",
                    "content": agent_response.message.message
                })
                continue

            # Check if the agent wants to execute an action
            if agent_response.action:
                action = agent_response.action
                if self.verbose:
                    logger.info(f"Executing action: {action.tool} with parameters {action.parameters}")

                # Execute the action
                observation = await self._execute_action(action)

                # Add the action and observation to the memory
                self.state.memory.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "action": {
                            "tool": action.tool,
                            "parameters": action.parameters
                        },
                        "thinking": agent_response.message.thinking if agent_response.message else None
                    })
                })
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

                # Update the plan if needed
                if self.state.plan:
                    # Create a description of the action for updating the plan
                    action_str = f"{action.tool}: {json.dumps(action.parameters)}"
                    
                    # If we have a current step, use its description instead
                    if current_step:
                        action_str = current_step.description
                    
                    # Update the plan
                    self.state.plan = await self._update_plan(action_str)
                    
                    # If the action was successful, mark the current step as completed
                    if observation.status == ActionStatus.SUCCESS and current_step:
                        if hasattr(self.state, 'structured_plan') and self.state.structured_plan:
                            plan = self.state.structured_plan
                            plan.mark_step_complete(current_step.id)
                            
                            # Get the next step
                            next_step = plan.get_next_incomplete_step()
                            if next_step:
                                plan.current_step_id = next_step.id
                                
                                # Add a message about the next step
                                self.state.memory.append({
                                    "role": "system",
                                    "content": f"Next step: {next_step.description}"
                                })

                # Check if the agent is done
                if action.tool == "done":
                    if self.verbose:
                        logger.info("Agent is done (done action)")
                    break
                
                if observation.status == ActionStatus.ERROR:
                    if self.verbose:
                        logger.error(f"Error executing action: {observation.error}")
                    self.state.memory.append({
                        "role": "system",
                        "content": f"Error executing action: {observation.error}"
                    })

        # Check if we reached the maximum number of steps
        if step_count >= self.max_steps:
            # Add a message to the memory
            self.state.memory.append({
                "role": "system",
                "content": f"Reached maximum number of steps ({self.max_steps})"
            })
            if self.verbose:
                logger.warning(f"Reached maximum number of steps ({self.max_steps})")
        else:
            if self.verbose:
                logger.info(f"Agent loop completed in {step_count} steps")

        # Return the memory for backward compatibility
        return self.state.memory
        
    def get_result(self) -> Dict[str, Any]:
        """
        Get the complete result including memory and token usage.
        
        Returns:
            Dictionary containing the memory and token usage
        """
        return {
            "memory": self.state.memory,
            "token_usage": self.get_token_usage()
        }

    async def _get_agent_response(self) -> AgentResponse:
        """
        Get a response from the agent.

        Returns:
            Agent response
        """
        # Create the messages for the agent
        messages = [{"role": "system", "content": self._get_system_message()}]

        # Process memory to include screenshot in user messages
        for msg in self.state.memory:
            if msg["role"] == "user" and self.state.screenshot_path and os.path.exists(self.state.screenshot_path):
                try:
                    # Read the screenshot file and encode it as base64
                    with open(self.state.screenshot_path, "rb") as image_file:
                        image_data = image_file.read()
                        base64_image = base64.b64encode(image_data).decode('utf-8')
                    
                    # Create a multimodal message with the original text and the image
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": msg["content"]},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    })
                except Exception as e:
                    logger.error(f"Error including screenshot in message: {str(e)}")
                    # Fall back to the original message
                    messages.append(msg)
            else:
                # For non-user messages or when there's no screenshot
                messages.append(msg)

        # Get the response from the agent
        response = self.agent.chat_completion(
            messages=messages, tools=self.get_tools_schema()
        )

        # Debug log the response
        if self.verbose:
            logger.info(f"Agent response: {json.dumps(response, indent=2)}")
            
        # Track token usage
        if "usage" in response:
            usage = response["usage"]
            # Update token counts
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            
            # Add to totals
            self.state.total_prompt_tokens += prompt_tokens
            self.state.total_completion_tokens += completion_tokens
            self.state.total_tokens += total_tokens
            
            # Add to history with timestamp
            self.state.token_usage_history.append({
                "timestamp": time.time(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model": response.get("model", self.agent.model),
                "step": len(self.state.memory) // 2  # Approximate step number
            })
            
            if self.verbose:
                logger.info(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
                logger.info(f"Cumulative tokens - Prompt: {self.state.total_prompt_tokens}, Completion: {self.state.total_completion_tokens}, Total: {self.state.total_tokens}")

        # Parse the response
        return self._parse_agent_response(response)

    def _get_system_message(self) -> str:
        """
        Get the system message for the agent.

        Returns:
            System message
        """
        # Get the tools schema
        tools_schema = self.get_tools_schema()
        
        # Get current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Get current page information
        current_url = self.state.current_url or "unknown page"
        page_title = self.state.page_title or "unknown title"
        
        # Get current step number
        current_step = self.state.memory.count({"role": "system", "content": lambda x: x.startswith("Current step:")}) + 1 if self.state.memory else 1

        # Create the system message
        system_message = get_agent_system_prompt(
            current_time=current_time,
            max_steps=self.max_steps,
            current_step=current_step,
            current_url=current_url,
            page_title=page_title,
            tools_schema=tools_schema
        )

        # Add context sections
        system_message += build_context_sections(self.state)

        return system_message

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
                logger.error(
                    f"Error parsing tool parameters: {tool_call['function']['arguments']}"
                )
                parameters = {}

            # Check if the tool is "done"
            if tool_name == "done":
                return AgentResponse(
                    done=True,
                    message=AgentMessage(
                        message=parameters.get("message", "Task completed"),
                        thinking=response.get("thinking"),
                    ),
                )

            # Create the action
            action = Action(tool=tool_name, parameters=parameters)

            # Create the message
            message_obj = None
            if content:
                message_obj = AgentMessage(
                    message=content, thinking=response.get("thinking")
                )

            return AgentResponse(action=action, message=message_obj)

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
                                thinking=content_json.get("thinking"),
                            ),
                        )

                    # Create the action
                    action = Action(tool=tool_name, parameters=parameters)

                    # Create the message
                    message_obj = None
                    if "thinking" in content_json:
                        message_obj = AgentMessage(
                            message="", thinking=content_json.get("thinking")
                        )

                    return AgentResponse(action=action, message=message_obj)

                # Check for multiple actions
                if "actions" in content_json:
                    actions_data = content_json["actions"]
                    if (
                        actions_data
                        and isinstance(actions_data, list)
                        and len(actions_data) > 0
                    ):
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
                                    thinking=content_json.get("thinking"),
                                ),
                            )

                        # Create the action
                        action = Action(tool=tool_name, parameters=parameters)

                        # Store the remaining actions in the state for later processing
                        self.state.pending_actions = actions_data[1:]

                        # Create the message
                        message_obj = None
                        if "thinking" in content_json:
                            message_obj = AgentMessage(
                                message="", thinking=content_json.get("thinking")
                            )

                        return AgentResponse(action=action, message=message_obj)
            except json.JSONDecodeError:
                # Not a valid JSON, treat as regular message
                pass

        # If there's no tool call, just return the message
        if content:
            return AgentResponse(
                message=AgentMessage(message=content, thinking=response.get("thinking"))
            )

        # If there's no content, return an empty response
        return AgentResponse()

    def get_token_usage(self) -> Dict[str, Any]:
        """
        Get token usage statistics.
        
        Returns:
            Dictionary with token usage statistics
        """
        return {
            "prompt_tokens": self.state.total_prompt_tokens,
            "completion_tokens": self.state.total_completion_tokens,
            "total_tokens": self.state.total_tokens,
            "history": self.state.token_usage_history,
            "estimated_cost": {
                # Approximate costs based on OpenAI's pricing (as of 2024)
                # These are estimates and may not be accurate
                "gpt-4o": {
                    "prompt": round(self.state.total_prompt_tokens * 0.00001, 6),  # $0.01 per 1K tokens
                    "completion": round(self.state.total_completion_tokens * 0.00003, 6),  # $0.03 per 1K tokens
                    "total": round(
                        (self.state.total_prompt_tokens * 0.00001) + 
                        (self.state.total_completion_tokens * 0.00003), 
                        6
                    )
                }
            }
        }
