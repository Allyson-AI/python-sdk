# Allyson Python SDK

AI-powered web browser automation.

## Installation

```bash
pip install allyson
```

After installation, you'll need to install the Playwright browsers:

```bash
python -m playwright install
```

## Features

- Simple, intuitive API for browser automation
- AI-powered element selection and interaction
- Support for multiple browsers (Chromium, Firefox, WebKit)
- Asynchronous and synchronous interfaces
- Robust error handling and recovery
- DOM extraction and analysis for AI integration
- Screenshot annotation with element bounding boxes
- Agent loop for automating tasks with natural language

## Quick Start
```python
from allyson import Browser, Agent, AgentLoop, Tool, ToolType

async def automate_task():
    # Create a browser instance
    async with Browser(
        headless=False,
        # Optional: Use your own Chrome installation instead of the default Chromium
        executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    ) as browser:
        # Create an agent instance with your OpenAI API key
        agent = Agent(api_key="your-api-key")
        
        # Create a custom tool
        weather_tool = Tool(
            name="get_weather",
            description="Get the current weather for a location",
            type=ToolType.CUSTOM,
            parameters_schema={
                "location": {"type": "string", "description": "Location to get weather for"}
            },
            function=lambda location: {"temperature": 72, "condition": "Sunny"}
        )
        
        # Create an agent loop
        agent_loop = AgentLoop(
            browser=browser,
            agent=agent,
            tools=[weather_tool],  # Optional custom tools
            max_steps=15,
            screenshot_dir="screenshots",
            plan_dir="plans",      # Directory to save task plans
            verbose=True
        )
        
        # Run the agent loop with a natural language task
        task = "Go to Google, search for 'Python programming language', and find information about it"
        memory = await agent_loop.run(task)
        
        # The memory contains the full conversation and actions taken
        print("Task completed!")
        
        # Print the final plan with completed steps
        if agent_loop.state.plan_path:
            with open(agent_loop.state.plan_path, "r") as f:
                print(f.read())

# Run the async function
import asyncio
asyncio.run(automate_task())
```

### Using Your Own Chrome Installation

By default, Allyson uses the Playwright-managed Chromium browser. However, for better stability and compatibility, you can use your own Chrome installation:

```python
# Windows
browser = Browser(executable_path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")

# macOS
browser = Browser(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

# Linux
browser = Browser(executable_path="/usr/bin/google-chrome")
```

This is especially useful for automation tasks that require specific browser versions or configurations.

### Agent Loop Features

The agent loop provides several powerful features for automating web tasks:

1. **Natural Language Instructions**: Describe tasks in plain English, and the agent will figure out how to accomplish them.

2. **Task Planning**: The agent automatically creates a step-by-step plan for completing the task and tracks progress by marking steps as completed.

3. **Built-in Tools**:
   - `goto`: Navigate to a URL
   - `click`: Click on an element by its ID number
   - `type`: Type text into an element by its ID number
   - `enter`: Press the Enter key to submit forms
   - `scroll`: Scroll the page in any direction
   - `done`: Mark the task as complete

4. **Action Chaining**: The agent can chain multiple actions together for efficiency:

```python
# The agent can chain actions like typing and pressing Enter
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
  ]
}
```

5. **Custom Tools**: Add your own tools to extend the agent's capabilities.

6. **Memory and Context**: The agent maintains a memory of all actions and observations, providing context for decision-making.

7. **Error Handling**: The agent can recover from errors and try alternative approaches.

8. **Screenshot Annotations**: Automatically take screenshots with annotated elements for better visibility.

### Example Plan

The agent creates a Markdown plan like this for each task:

```markdown
# Plan for: Search for information about Python programming language

## Steps:
- [x] Navigate to a search engine
- [x] Search for "Python programming language"
- [ ] Review search results
  - [ ] Identify official Python website
  - [ ] Identify Wikipedia page
- [ ] Visit the most relevant page
- [ ] Extract key information
  - [ ] What is Python
  - [ ] Key features
  - [ ] Current version
- [ ] Summarize findings
```

As the agent completes steps, it automatically updates the plan by marking steps as completed with checkboxes.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

## Changelog
- **0.1.6** - Added support for custom Chrome browser path
- **0.1.5** - Added planner feature for creating and tracking task progress
- **0.1.4** - Enhanced agent loop with action chaining, Enter key tool, and improved error handling
- **0.1.3** - Added DOM extraction and screenshot annotation features
- **0.1.2** - Updated Description
- **0.1.1** - Test release for GitHub Actions automated publishing
- **0.1.0** - Initial release