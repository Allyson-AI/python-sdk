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
from allyson import Browser

# Create a browser instance
browser = Browser()

# Navigate to a website
browser.goto("https://example.com")

# Interact with the page
browser.click("Sign in")
browser.fill("Email", "user@example.com")
browser.fill("Password", "password")
browser.click("Submit")

# Take a screenshot
browser.screenshot("login.png")

# Close the browser
browser.close()
```

## Advanced Usage

```python
from allyson import Browser

async def run_automation():
    # Use async API with context manager
    async with Browser(headless=False) as browser:
        await browser.goto("https://example.com")
        
        # Wait for specific element
        await browser.wait_for_selector(".content")
        
        # Execute JavaScript
        result = await browser.evaluate("document.title")
        print(f"Page title: {result}")
        
        # Multiple tabs/pages
        new_page = await browser.new_page()
        await new_page.goto("https://another-example.com")

# Run the async function
import asyncio
asyncio.run(run_automation())
```

## DOM Extraction and Screenshot Annotation

```python
from allyson import Browser, DOMExtractor

async def extract_and_annotate():
    async with Browser(headless=False) as browser:
        # Navigate to a website
        await browser.goto("https://example.com")
        
        # Create a DOM extractor
        dom_extractor = DOMExtractor(browser._page)
        
        # Extract interactive elements
        elements = await dom_extractor.extract_interactive_elements()
        print(f"Found {len(elements)} interactive elements")
        
        # Take a screenshot with annotations
        result = await dom_extractor.screenshot_with_annotations(
            path="screenshot.png",
            elements=elements,
            show_element_ids=True,
            box_color="red"
        )
        
        print(f"Clean screenshot: {result['clean']}")
        print(f"Annotated screenshot: {result['annotated']}")
        
        # Create an element map for AI analysis
        map_result = await dom_extractor.screenshot_with_element_map(
            path="element_map.png"
        )
        
        # The element map contains detailed information about each element
        for element in map_result["elementMap"]:
            print(f"Element #{element['id']}: {element['elementType']}")

# Run the async function
import asyncio
asyncio.run(extract_and_annotate())
```

## Agent Loop for Task Automation

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

## Automated Publishing

This package uses GitHub Actions for automated testing and publishing to PyPI. The workflow is configured to:

1. Run tests on every push to the main branch and on pull requests
2. Build the package on every push to the main branch
3. Publish to PyPI automatically when:
   - A new tag is pushed with the format `v*` (e.g., v0.1.0, v1.0.0)
   - A new GitHub Release is created

To publish a new version:

1. Update the version number in `setup.py`
2. Commit and push your changes to the main branch
3. Create and push a new tag:
   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```
4. The GitHub Action will automatically build and publish the package to PyPI

Note: You need to set up a PyPI API token as a GitHub secret named `PYPI_API_TOKEN` for the automated publishing to work.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 


## Changelog
- **0.1.4** - Enhanced agent loop with action chaining, Enter key tool, and improved error handling
- **0.1.3** - Added DOM extraction and screenshot annotation features
- **0.1.2** - Updated Description
- **0.1.1** - Test release for GitHub Actions automated publishing
- **0.1.0** - Initial release