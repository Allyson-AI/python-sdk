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

from allyson.agent import Agent
from allyson.tools import Tool, ToolType, get_default_tools
from allyson.browser import Browser
from allyson import DOMExtractor

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


class PlanStep(BaseModel):
    """A step in the plan."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])  # Unique identifier for the step
    description: str
    completed: bool = False
    parent_id: Optional[str] = None  # ID of parent step if this is a substep
    substeps: List["PlanStep"] = Field(default_factory=list)
    completion_time: Optional[datetime] = None  # When the step was completed


PlanStep.update_forward_refs()


class Plan(BaseModel):
    """A plan for completing a task."""

    task: str
    steps: List[PlanStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    current_step_id: Optional[str] = None  # Track the current step being worked on
    completed_steps: List[str] = Field(default_factory=list)  # List of completed step IDs
    
    def get_step_by_id(self, step_id: str) -> Optional[PlanStep]:
        """Find a step by its ID."""
        for step in self.steps:
            if step.id == step_id:
                return step
            if step.substeps:
                for substep in step.substeps:
                    if substep.id == step_id:
                        return substep
        return None
    
    def get_next_incomplete_step(self) -> Optional[PlanStep]:
        """Get the next incomplete step in the plan."""
        for step in self.steps:
            if not step.completed:
                return step
            if step.substeps:
                for substep in step.substeps:
                    if not substep.completed:
                        return substep
        return None
    
    def mark_step_complete(self, step_id: str) -> bool:
        """Mark a step as complete by its ID."""
        step = self.get_step_by_id(step_id)
        if step:
            step.completed = True
            step.completion_time = datetime.now()
            self.completed_steps.append(step_id)
            self.last_updated = datetime.now()
            return True
        return False
    
    def to_markdown(self) -> str:
        """Convert the plan to markdown format."""
        lines = [f"# Plan for: {self.task}\n", "## Steps:"]
        
        for step in self.steps:
            checkbox = "[x]" if step.completed else "[ ]"
            lines.append(f"- {checkbox} {step.description}")
            
            if step.substeps:
                for substep in step.substeps:
                    checkbox = "[x]" if substep.completed else "[ ]"
                    lines.append(f"  - {checkbox} {substep.description}")
        
        return "\n".join(lines)
    
    @classmethod
    def from_markdown(cls, markdown: str) -> "Plan":
        """
        Parse a markdown plan into a structured Plan object.
        
        Args:
            markdown: Markdown string containing the plan
            
        Returns:
            Plan object
        """
        lines = markdown.strip().split("\n")
        
        # Extract task from the title
        task = ""
        for line in lines:
            if line.startswith("# Plan for:"):
                task = line.replace("# Plan for:", "").strip()
                break
        
        plan = cls(task=task)
        
        # Parse steps and substeps
        current_step = None
        for line in lines:
            # Skip empty lines and headers
            if not line or line.startswith("#"):
                continue
                
            # Check if it's a step or substep
            if line.startswith("- "):
                # It's a main step
                step_text = line[2:].strip()
                completed = False
                
                # Check if it's completed
                if step_text.startswith("[x]"):
                    completed = True
                    step_text = step_text[3:].strip()
                elif step_text.startswith("[ ]"):
                    step_text = step_text[3:].strip()
                
                # Create the step
                current_step = PlanStep(
                    description=step_text,
                    completed=completed
                )
                
                # Add to the plan
                plan.steps.append(current_step)
                
                # If completed, add to completed steps
                if completed:
                    plan.completed_steps.append(current_step.id)
                
            elif line.lstrip().startswith("- ") and current_step and line.startswith("  "):
                # It's a substep (indented with spaces)
                substep_text = line.lstrip()[2:].strip()
                completed = False
                
                # Check if it's completed
                if substep_text.startswith("[x]"):
                    completed = True
                    substep_text = substep_text[3:].strip()
                elif substep_text.startswith("[ ]"):
                    substep_text = substep_text[3:].strip()
                
                # Create the substep
                substep = PlanStep(
                    description=substep_text,
                    completed=completed,
                    parent_id=current_step.id
                )
                
                # Add to the current step
                current_step.substeps.append(substep)
                
                # If completed, add to completed steps
                if completed:
                    plan.completed_steps.append(substep.id)
        
        # Set the current step to the first incomplete step
        next_step = plan.get_next_incomplete_step()
        if next_step:
            plan.current_step_id = next_step.id
            
        return plan
    
    def find_step_by_description(self, description: str) -> Optional[PlanStep]:
        """
        Find a step by its description (or partial match).
        
        Args:
            description: Description to match
            
        Returns:
            Matching PlanStep or None
        """
        # First try exact match
        for step in self.steps:
            if step.description.lower() == description.lower():
                return step
            
            for substep in step.substeps:
                if substep.description.lower() == description.lower():
                    return substep
        
        # Then try partial match
        best_match = None
        best_score = 0
        
        for step in self.steps:
            score = self._similarity_score(step.description.lower(), description.lower())
            if score > best_score:
                best_match = step
                best_score = score
            
            for substep in step.substeps:
                score = self._similarity_score(substep.description.lower(), description.lower())
                if score > best_score:
                    best_match = substep
                    best_score = score
        
        # Only return if we have a reasonable match
        if best_score > 0.5:
            return best_match
            
        return None
    
    def _similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate a simple similarity score between two strings.
        
        Args:
            text1: First string
            text2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Check if one is a substring of the other
        if text1 in text2 or text2 in text1:
            return 0.8
            
        # Count matching words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0
            
        common_words = words1.intersection(words2)
        return len(common_words) / max(len(words1), len(words2))


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
            # Execute the tool function with the parameters
            result = tool.function(**action.parameters)

            # If the result is a coroutine, await it
            if asyncio.iscoroutine(result):
                result = await result

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

        # Create a system message for the planner
        system_message = f"""
You are an expert strategic planner specializing in web automation and information retrieval tasks.

## YOUR OBJECTIVE
Create a comprehensive, structured plan that breaks down the given task into logical steps and substeps.

## PLAN REQUIREMENTS
Your plan must:
1. Be thorough yet concise - identify all necessary steps without excessive detail
2. Focus on strategic milestones rather than mechanical actions
3. Include appropriate substeps for complex operations
4. Anticipate potential challenges and decision points
5. Be formatted as a Markdown checklist with proper hierarchy
6. Consider the maximum step limit of {self.max_steps} while ensuring task completion

## PLAN STRUCTURE
```markdown
# Plan for: [Task Description]

## Steps:
- [ ] Major Step 1
  - [ ] Substep 1.1 (if needed)
  - [ ] Substep 1.2 (if needed)
- [ ] Major Step 2
  - [ ] Substep 2.1 (if needed)
...and so on
```

## PLANNING CONSIDERATIONS
- **Task Complexity Analysis**: Assess whether this is a simple, moderate, or complex task
  - Simple tasks (10 steps or less): Direct, straightforward operations (e.g., "Search for X")
  - Moderate tasks (10-20 steps): Multi-stage operations with clear progression
  - Complex tasks (30+ steps): Tasks requiring research, comparison, or multiple sources

- **Information Gathering**: For research tasks, include steps for:
  - Identifying authoritative sources
  - Cross-referencing information
  - Organizing findings by relevance or category
  - Synthesizing a comprehensive summary

- **Navigation Planning**: Include explicit steps for:
  - Initial navigation to appropriate starting points
  - Moving between different sites or sections as needed
  - Returning to previous pages when necessary

- **Contingency Planning**: Consider alternative paths when:
  - Information might not be available at the first source
  - User authentication might be required
  - Search results might need refinement

- **Verification Steps**: For critical information, include steps to:
  - Confirm data accuracy across multiple sources
  - Validate that all requested information has been gathered

## EXAMPLES OF EFFECTIVE PLANS

### Simple Task Example:
Task: Check the weather in New York
```markdown
# Plan for: Check the weather in New York

## Steps:
- [ ] Navigate to a weather service website
- [ ] Search for "New York weather"
- [ ] Extract current weather conditions and forecast
- [ ] Summarize the weather information
```

### Complex Task Example:
Task: Research and compare features of the top 3 electric vehicles
```markdown
# Plan for: Research and compare features of the top 3 electric vehicles

## Steps:
- [ ] Identify authoritative sources for EV information
  - [ ] Find automotive review websites
  - [ ] Locate manufacturer websites
- [ ] Determine the current top 3 electric vehicles by sales or ratings
- [ ] Research each vehicle individually
  - [ ] Gather specifications (range, charging time, price)
  - [ ] Collect performance data
  - [ ] Find safety ratings
  - [ ] Note unique features
- [ ] Create a comparative analysis
  - [ ] Organize data in a structured format
  - [ ] Highlight key differences
  - [ ] Note pros and cons of each vehicle
- [ ] Summarize findings with recommendations based on different priorities
  - [ ] Best value option
  - [ ] Best performance option
  - [ ] Best overall option
```

Now, analyze the following task and create an appropriate plan:
{task}
"""

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
        system_message = """
You are an expert plan tracker responsible for maintaining accurate progress records for complex tasks.

## YOUR OBJECTIVE
Update the provided Markdown plan by identifying and marking completed steps based on the action description.

## TASK REQUIREMENTS
1. Analyze the completed action description carefully
2. Find the step in the plan that best matches this action
3. Mark ONLY that step as completed by changing "[ ]" to "[x]"
4. Maintain the exact structure and formatting of the original plan
5. Return the complete updated plan in Markdown format

## MATCHING GUIDELINES
- **Exact Matches**: If the action description exactly matches a step, mark that step
- **Partial Matches**: If no exact match exists, use semantic understanding to find the closest step
- **Context Awareness**: Consider the logical progression of the plan when choosing between multiple potential matches
- **Hierarchical Awareness**: If a substep is completed but its parent step still has incomplete substeps, only mark the substep as completed
- **Completion Logic**: If all substeps of a parent step are completed, also mark the parent step as completed

## IMPORTANT CONSIDERATIONS
- Never add new steps or modify existing step descriptions
- Never mark multiple steps as completed unless they are directly related (parent/child)
- Preserve all formatting, indentation, and structure of the original plan
- If truly no matching step exists, return the plan unchanged with an explanation

## RESPONSE FORMAT
Return ONLY the updated Markdown plan with the appropriate step(s) marked as completed.
"""

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

        # Create a plan for the task
        plan_markdown = await self._create_plan(task)

        # Update state to get screenshot
        await self._update_state()

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

        # Create the system message
        system_message = f"""
You are an advanced AI navigator designed to accomplish web browsing tasks with precision and intelligence.

## TASK CONTEXT
- Current date and time: {current_time}
- Maximum steps available: {self.max_steps}
- You are currently on step: {self.state.memory.count({"role": "system", "content": lambda x: x.startswith("Current step:")}) + 1 if self.state.memory else 1}

## AVAILABLE TOOLS
{json.dumps(tools_schema, indent=2)}

## RESPONSE FORMAT
You must respond using function calling with the following format:

1. For executing actions:
```json
{{
  "action": {{
    "tool": "tool_name",
    "parameters": {{
      "param1": "value1",
      "param2": "value2"
    }}
  }},
  "thinking": "Your detailed reasoning process explaining why you're taking this action and how it contributes to the overall task"
}}
```

2. For direct responses to the user:
Simply provide your message as regular text without any special formatting.

3. For task completion:
```json
{{
  "action": {{
    "tool": "done",
    "parameters": {{
      "message": "Comprehensive summary of what you accomplished and all information gathered"
    }}
  }}
}}
```

## STRATEGIC GUIDELINES
1. **Task Analysis**
   - Break down complex tasks into logical steps
   - Maintain awareness of your progress through the task plan
   - Adapt your approach based on what you observe on each page

2. **Navigation Strategy**
   - Use precise element selection based on context and relevance
   - Handle unexpected situations (popups, login prompts, etc.) gracefully
   - If you encounter errors, try alternative approaches

3. **Information Gathering**
   - Extract relevant information completely and accurately
   - Organize information in a structured, readable format
   - Verify critical information when possible

4. **Memory Management**
   - Keep track of important information across multiple pages
   - Remember your progress on multi-step tasks
   - Count items when processing multiple similar elements (e.g., "3 of 10 items processed")

5. **Efficiency Considerations**
   - Chain related actions when appropriate (e.g., fill multiple form fields)
   - Minimize unnecessary page loads and navigation
   - Use scrolling to find elements before assuming they don't exist

## COMPLETION CRITERIA
- Only mark the task as complete when ALL requested information or actions are finished
- If you reach the maximum step limit, use the "done" tool with a summary of progress so far
- Include ALL gathered information in your final summary

Remember: You are the user's expert navigator. Think step-by-step, be thorough, and explain your reasoning clearly.
"""

        # Add information about the current state
        if self.state.current_url:
            system_message += f"\n\n## CURRENT CONTEXT\nURL: {self.state.current_url}"
        if self.state.page_title:
            system_message += f"\nPage title: {self.state.page_title}"

        # Add information about the plan if available
        if hasattr(self.state, 'structured_plan') and self.state.structured_plan:
            plan = self.state.structured_plan
            
            # Add the plan
            system_message += f"\n\n## TASK PLAN\n{plan.to_markdown()}"
            
            # Add information about the current step
            if plan.current_step_id:
                current_step = plan.get_step_by_id(plan.current_step_id)
                if current_step:
                    system_message += f"\n\n## CURRENT FOCUS\nActive Step: {current_step.description}"
                    
                    # Calculate progress percentage
                    total_steps = len(plan.steps)
                    completed_steps = len(plan.completed_steps)
                    progress_percentage = int((completed_steps / total_steps) * 100) if total_steps > 0 else 0
                    
                    system_message += f"\nProgress: {progress_percentage}% ({completed_steps}/{total_steps} steps completed)"
                    
                    # If there are substeps, add them
                    if current_step.substeps:
                        system_message += "\nRequired Substeps:"
                        for substep in current_step.substeps:
                            checkbox = "[x]" if substep.completed else "[ ]"
                            system_message += f"\n  - {checkbox} {substep.description}"
            
            # Add information about the next step
            next_step = plan.get_next_incomplete_step()
            if next_step:
                system_message += f"\n\nNext Step: {next_step.description}"
                
                # Add potential challenges for this step if applicable
                if "search" in next_step.description.lower():
                    system_message += "\nPotential challenges: Results may vary, be prepared to refine search terms"
                elif "login" in next_step.description.lower():
                    system_message += "\nPotential challenges: May encounter CAPTCHA or verification steps"
                elif "extract" in next_step.description.lower():
                    system_message += "\nPotential challenges: Content may be paginated or dynamically loaded"
        elif self.state.plan:
            system_message += f"\n\n## TASK PLAN\n{self.state.plan}"

        # Add information about interactive elements if available
        if self.state.interactive_elements:
            system_message += "\n\n## INTERACTIVE ELEMENTS\nAvailable elements on the current page:"
            for i, element in enumerate(self.state.interactive_elements):
                description = element.get('description', element.get('textContent', 'Unknown element'))
                element_type = element.get('type', element.get('elementType', 'unknown'))
                system_message += f"\n[{i+1}] {description} (type: {element_type})"
            
            system_message += "\n\nNote: Use the numeric index in [] to interact with elements"

        # Add information about the screenshot if available
        if self.state.screenshot_path:
            system_message += f"\n\n## VISUAL INFORMATION\nA screenshot of the current webpage is included with the user message."
            system_message += "\nThe screenshot shows numbered red boxes around interactive elements."
            system_message += "\nUse these numbers when referring to elements in your actions (e.g., click element #3)."

        # Add information about pending actions if available
        if self.state.pending_actions and len(self.state.pending_actions) > 0:
            system_message += "\n\n## PENDING ACTIONS\nActions queued for execution:"
            for i, action in enumerate(self.state.pending_actions):
                system_message += f"\n{i+1}. {json.dumps(action)}"

        # Add memory context if available
        if self.state.memory and len(self.state.memory) > 2:  # Skip the initial user message and plan
            # Extract the last few messages for context
            recent_messages = self.state.memory[-5:] if len(self.state.memory) > 5 else self.state.memory
            
            system_message += "\n\n## RECENT ACTIVITY"
            for msg in recent_messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                # Truncate long content
                if len(content) > 200:
                    content = content[:197] + "..."
                
                if role == "user":
                    system_message += f"\nUser: {content}"
                elif role == "assistant" and content and not content.startswith("{"):
                    system_message += f"\nYou: {content}"
                elif role == "system" and "step" in content.lower():
                    system_message += f"\nSystem: {content}"

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
                },
                "gpt-4": {
                    "prompt": round(self.state.total_prompt_tokens * 0.00003, 6),  # $0.03 per 1K tokens
                    "completion": round(self.state.total_completion_tokens * 0.00006, 6),  # $0.06 per 1K tokens
                    "total": round(
                        (self.state.total_prompt_tokens * 0.00003) + 
                        (self.state.total_completion_tokens * 0.00006), 
                        6
                    )
                },
                "gpt-3.5-turbo": {
                    "prompt": round(self.state.total_prompt_tokens * 0.000001, 6),  # $0.001 per 1K tokens
                    "completion": round(self.state.total_completion_tokens * 0.000002, 6),  # $0.002 per 1K tokens
                    "total": round(
                        (self.state.total_prompt_tokens * 0.000001) + 
                        (self.state.total_completion_tokens * 0.000002), 
                        6
                    )
                }
            }
        }
