"""
Prompts used by the Allyson agent.
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from allyson.plan import Plan


def get_agent_system_prompt(
    current_time: str,
    max_steps: int,
    current_step: int,
    current_url: str,
    page_title: str,
    tools_schema: List[Dict[str, Any]]
) -> str:
    """
    Get the main system prompt for the agent.
    
    Args:
        current_time: Current date and time
        max_steps: Maximum steps available
        current_step: Current step number
        current_url: Current page URL
        page_title: Current page title
        tools_schema: Schema of available tools
        
    Returns:
        System prompt for the agent
    """
    return f"""
You are an advanced AI navigator designed to accomplish web browsing tasks with precision and intelligence.

## TASK CONTEXT
- Current date and time: {current_time}
- Maximum steps available: {max_steps}
- You are currently on step: {current_step}
- Current page URL: {current_url}
- Current page title: {page_title}

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


def get_planner_system_prompt(current_url: str, page_title: str, max_steps: int) -> str:
    """
    Get the system prompt for the planner.
    
    Args:
        current_url: Current page URL
        page_title: Current page title
        max_steps: Maximum steps available
        
    Returns:
        System prompt for the planner
    """
    return f"""
You are an expert strategic planner specializing in web automation and information retrieval tasks.

## YOUR OBJECTIVE
Create a comprehensive, structured plan that breaks down the given task into logical steps and substeps.

## CURRENT CONTEXT
- You are already on a page with URL: {current_url}
- The current page title is: {page_title}
- Your plan should start from this current page, not from a blank state

## PLAN REQUIREMENTS
Your plan must:
1. Be thorough yet concise - identify all necessary steps without excessive detail
2. Focus on strategic milestones rather than mechanical actions
3. Include appropriate substeps for complex operations
4. Anticipate potential challenges and decision points
5. Be formatted as a Markdown checklist with proper hierarchy
6. Consider the maximum step limit of {max_steps} while ensuring task completion
7. DO NOT include steps to navigate to the current page as you are already there

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
  - Moving between different sites or sections as needed
  - Returning to previous pages when necessary

- **Contingency Planning**: Consider alternative paths when:
  - Information might not be available at the first source
  - User authentication might be required
  - Search results might need refinement
"""


def get_plan_updater_system_prompt() -> str:
    """
    Get the system prompt for the plan updater.
    
    Returns:
        System prompt for the plan updater
    """
    return """
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
"""


def build_context_sections(state) -> str:
    """
    Build the context sections for the system message.
    
    Args:
        state: Current agent state
        
    Returns:
        Context sections as a string
    """
    context_sections = ""
    
    # Add information about the current state
    if state.current_url:
        context_sections += f"\n\n## CURRENT CONTEXT\nURL: {state.current_url}"
    if state.page_title:
        context_sections += f"\nPage title: {state.page_title}"

    # Add information about the plan if available
    if hasattr(state, 'structured_plan') and state.structured_plan:
        plan = state.structured_plan
        
        # Add the plan
        context_sections += f"\n\n## TASK PLAN\n{plan.to_markdown()}"
        
        # Add information about the current step
        if plan.current_step_id:
            current_step = plan.get_step_by_id(plan.current_step_id)
            if current_step:
                context_sections += f"\n\n## CURRENT FOCUS\nActive Step: {current_step.description}"
                
                # Calculate progress percentage
                total_steps = len(plan.steps)
                completed_steps = len(plan.completed_steps)
                progress_percentage = int((completed_steps / total_steps) * 100) if total_steps > 0 else 0
                
                context_sections += f"\nProgress: {progress_percentage}% ({completed_steps}/{total_steps} steps completed)"
                
                # If there are substeps, add them
                if current_step.substeps:
                    context_sections += "\nRequired Substeps:"
                    for substep in current_step.substeps:
                        checkbox = "[x]" if substep.completed else "[ ]"
                        context_sections += f"\n  - {checkbox} {substep.description}"
        
        # Add information about the next step
        next_step = plan.get_next_incomplete_step()
        if next_step:
            context_sections += f"\n\nNext Step: {next_step.description}"
            
            # Add potential challenges for this step if applicable
            if "search" in next_step.description.lower():
                context_sections += "\nPotential challenges: Results may vary, be prepared to refine search terms"
            elif "login" in next_step.description.lower():
                context_sections += "\nPotential challenges: May encounter CAPTCHA or verification steps"
            elif "extract" in next_step.description.lower():
                context_sections += "\nPotential challenges: Content may be paginated or dynamically loaded"
    elif state.plan:
        context_sections += f"\n\n## TASK PLAN\n{state.plan}"

    # Add information about interactive elements if available
    if state.interactive_elements:
        context_sections += "\n\n## INTERACTIVE ELEMENTS\nAvailable elements on the current page:"
        for i, element in enumerate(state.interactive_elements):
            description = element.get('description', element.get('textContent', 'Unknown element'))
            element_type = element.get('type', element.get('elementType', 'unknown'))
            context_sections += f"\n[{i+1}] {description} (type: {element_type})"
        
        context_sections += "\n\nNote: Use the numeric index in [] to interact with elements"

    # Add information about the screenshot if available
    if state.screenshot_path:
        context_sections += f"\n\n## VISUAL INFORMATION\nA screenshot of the current webpage is included with the user message."
        context_sections += "\nThe screenshot shows numbered red boxes around interactive elements."
        context_sections += "\nUse these numbers when referring to elements in your actions (e.g., click element #3)."

    # Add information about pending actions if available
    if state.pending_actions and len(state.pending_actions) > 0:
        context_sections += "\n\n## PENDING ACTIONS\nActions queued for execution:"
        for i, action in enumerate(state.pending_actions):
            context_sections += f"\n{i+1}. {json.dumps(action)}"

    # Add memory context if available
    if state.memory and len(state.memory) > 2:  # Skip the initial user message and plan
        # Extract the last few messages for context
        recent_messages = state.memory[-5:] if len(state.memory) > 5 else state.memory
        
        context_sections += "\n\n## RECENT ACTIVITY"
        for msg in recent_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # Truncate long content
            if len(content) > 200:
                content = content[:197] + "..."
            
            if role == "user":
                context_sections += f"\nUser: {content}"
            elif role == "assistant" and content and not content.startswith("{"):
                context_sections += f"\nYou: {content}"
            elif role == "system" and "step" in content.lower():
                context_sections += f"\nSystem: {content}"
                
    return context_sections 