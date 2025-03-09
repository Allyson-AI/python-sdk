"""
Example of using Allyson's AgentLoop to automate web browsing tasks.

This example demonstrates how to use the AgentLoop to automate web browsing tasks
using natural language instructions, including action chaining, the enter key tool,
and the planner feature for tracking progress.
"""

import asyncio
import os
import json
import logging
from typing import Dict, Any, List, Optional

from allyson import Browser, Agent
from allyson.agent_loop import AgentLoop
from allyson.tools import Tool, ToolType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


async def run_agent_loop():
    """
    Run the agent loop to automate a web browsing task.
    
    This example demonstrates:
    1. Creating a plan for the task
    2. Navigating to Google
    3. Searching for a topic using action chaining (type + enter)
    4. Finding information from the search results
    5. Tracking progress using the plan
    """
    print("Running agent loop example...")
    
    # Get the OpenAI API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it to your OpenAI API key."
        )
    
    # Create a browser instance
    async with Browser(headless=False) as browser:
        # Create an agent instance
        agent = Agent(api_key=api_key)
        
        # Create a custom weather tool
        weather_tool = Tool(
            name="get_weather",
            description="Get the current weather for a location",
            type=ToolType.CUSTOM,
            parameters_schema={
                "location": {"type": "string", "description": "Location to get weather for"}
            },
            function=lambda location: get_weather(location)
        )
        
        # Create directories for screenshots and plans
        screenshot_dir = "screenshots"
        plan_dir = "plans"
        
        for directory in [screenshot_dir, plan_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Create an agent loop
        agent_loop = AgentLoop(
            browser=browser,
            agent=agent,
            tools=[weather_tool],
            max_steps=15,
            screenshot_dir=screenshot_dir,
            plan_dir=plan_dir,
            verbose=True
        )
        
        # Run the agent loop with a task that demonstrates planning and action chaining
        task = "Go to Google, search for 'Elon Musk', and find information about his companies"
        
        print(f"Running task: {task}")
        memory = await agent_loop.run(task)
        
        # Print the memory
        print("\nTask completed! Memory:")
        for i, message in enumerate(memory):
            role = message["role"]
            content = message["content"]
            
            if role == "system" and isinstance(content, str) and content.startswith("{"):
                # Parse JSON content
                try:
                    data = json.loads(content)
                    if "observation" in data:
                        observation = data["observation"]
                        print(f"{i}. OBSERVATION: {observation['status']}")
                        if observation["data"]:
                            print(f"   Data: {observation['data']}")
                        if observation["error"]:
                            print(f"   Error: {observation['error']}")
                    else:
                        print(f"{i}. SYSTEM: {content[:100]}...")
                except:
                    print(f"{i}. SYSTEM: {content[:100]}...")
            elif role == "assistant" and isinstance(content, str) and content.startswith("{"):
                # Parse JSON content
                try:
                    data = json.loads(content)
                    if "action" in data:
                        action = data["action"]
                        print(f"{i}. ACTION: {action['tool']} - {action['parameters']}")
                        if "thinking" in data and data["thinking"]:
                            print(f"   Thinking: {data['thinking']}")
                    elif "actions" in data:
                        actions = data["actions"]
                        print(f"{i}. CHAINED ACTIONS:")
                        for j, action in enumerate(actions):
                            print(f"   {j+1}. {action['tool']} - {action['parameters']}")
                        if "thinking" in data and data["thinking"]:
                            print(f"   Thinking: {data['thinking']}")
                    else:
                        print(f"{i}. ASSISTANT: {content[:100]}...")
                except:
                    print(f"{i}. ASSISTANT: {content[:100]}...")
            else:
                print(f"{i}. {role.upper()}: {content[:100]}...")
        
        # Print the final plan
        if agent_loop.state.plan_path:
            print("\nFinal Plan:")
            with open(agent_loop.state.plan_path, "r") as f:
                print(f.read())


async def get_weather(location: str) -> Dict[str, Any]:
    """
    Mock function to get weather data for a location.
    
    In a real application, this would call a weather API.
    
    Args:
        location: Location to get weather for
        
    Returns:
        Weather data for the location
    """
    # Mock weather data
    weather_data = {
        "New York": {"temperature": 72, "condition": "Sunny", "humidity": 45},
        "London": {"temperature": 62, "condition": "Cloudy", "humidity": 80},
        "Tokyo": {"temperature": 85, "condition": "Partly Cloudy", "humidity": 70},
        "Sydney": {"temperature": 78, "condition": "Clear", "humidity": 50},
    }
    
    # Return weather data for the location or a default
    return weather_data.get(location, {"temperature": 75, "condition": "Unknown", "humidity": 60})


async def main():
    """Main function to run the example."""
    await run_agent_loop()


if __name__ == "__main__":
    asyncio.run(main()) 