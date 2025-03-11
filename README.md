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
- Advanced anti-detection features to bypass bot detection

## Quick Start
```python
from allyson import Browser, Agent, AgentLoop, Tool, ToolType

async def automate_task():
    # Create a browser instance
    async with Browser(
        headless=False,
        # Optional: Use your own Chrome installation instead of the default Chromium
        executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        # Optional: Customize viewport size (default is 1280x720)
        viewport={"width": 1366, "height": 768}
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

### Advanced Anti-Detection Features

Allyson includes comprehensive anti-detection features to bypass bot detection systems. These features are always enabled and require no configuration:

The anti-detection features include:

1. **Playwright Stealth**: Automatically applies various techniques to make the browser appear more like a regular user browser.

2. **Browser Launch Arguments**: Uses special browser arguments to disable automation flags:
   - Disables automation-controlled features
   - Disables infobars and notifications
   - Prevents automation detection
   - Handles security certificates automatically

3. **JavaScript Evasion**: Injects custom JavaScript to override browser properties commonly used for bot detection:
   - Sets `navigator.webdriver` to false
   - Modifies browser plugins and features
   - Handles permission queries naturally

4. **Customizable Viewport**: You can specify your preferred viewport size (default is 1280x720):
   ```python
   browser = Browser(viewport={"width": 1366, "height": 768})
   ```

5. **Realistic User Agent**: Uses a modern Chrome user agent string.

6. **Browser Fingerprint Protection**: Modifies browser properties that are commonly used for fingerprinting.

These features help bypass detection on websites with sophisticated anti-bot measures, making your automation more reliable and less likely to be blocked.

### Agent Loop Features

The agent loop provides several powerful features for automating web tasks:

1. **Natural Language Instructions**: Describe tasks in plain English, and Allyson will figure out how to accomplish them.

2. **Structured Task Planning**: Allyson automatically creates a step-by-step plan for completing the task and tracks progress by marking steps as completed. The enhanced planner:
   - Maintains a structured representation of the plan with steps and substeps
   - Automatically tracks the current step and progresses to the next step
   - Uses intelligent matching to find and update the right steps
   - Provides consistent formatting and organization of plans

3. **Built-in Tools**:
   - `goto`: Navigate to a URL
   - `click`: Click on an element by its ID number
   - `type`: Type text into an element by its ID number
   - `enter`: Press the Enter key to submit forms
   - `scroll`: Scroll the page in any direction
   - `done`: Mark the task as complete

4. **Action Chaining**: Allyson can chain multiple actions together for efficiency:

```python
# Allyson can chain actions like typing and pressing Enter
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

5. **Custom Tools**: Add your own tools to extend Allyson's capabilities.

6. **Memory and Context**: Allyson maintains a memory of all actions and observations, providing context for decision-making.

7. **Error Handling**: Allyson can recover from errors and try alternative approaches.

8. **Screenshot Annotations**: Automatically take screenshots with annotated elements for better visibility.

### Example Plan

Allyson creates a Markdown plan like this for each task:

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

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE). This means if you use this software as part of a service you offer to others, you must make the complete source code available to the users of that service under the same license.

1. AGPL-3.0 - Personal Use
2. Commercial License (Contact us for details)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed list of changes between versions.
