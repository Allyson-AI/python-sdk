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
    async with Browser(headless=False) as browser:
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
            max_iterations=15,
            screenshot_dir="screenshots",
            verbose=True
        )
        
        # Run the agent loop with a natural language task
        task = "Go to Google, search for 'Python programming language', and find information about it"
        memory = await agent_loop.run(task)
        
        # The memory contains the full conversation and actions taken
        print("Task completed!")

# Run the async function
import asyncio
asyncio.run(automate_task())
```

### Agent Loop Features

The agent loop provides several powerful features for automating web tasks:

1. **Natural Language Instructions**: Describe tasks in plain English, and the agent will figure out how to accomplish them.

2. **Built-in Tools**:
   - `goto`: Navigate to a URL
   - `click`: Click on an element by its ID number
   - `type`: Type text into an element by its ID number
   - `enter`: Press the Enter key to submit forms
   - `scroll`: Scroll the page in any direction
   - `done`: Mark the task as complete

3. **Action Chaining**: The agent can chain multiple actions together for efficiency:

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

4. **Custom Tools**: Add your own tools to extend the agent's capabilities.

5. **Memory and Context**: The agent maintains a memory of all actions and observations, providing context for decision-making.

6. **Error Handling**: The agent can recover from errors and try alternative approaches.

7. **Screenshot Annotations**: Automatically take screenshots with annotated elements for better visibility.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 

## Changelog
- **0.1.4** - Enhanced agent loop with action chaining, Enter key tool, and improved error handling
- **0.1.3** - Added DOM extraction and screenshot annotation features
- **0.1.2** - Updated Description
- **0.1.1** - Test release for GitHub Actions automated publishing
- **0.1.0** - Initial release