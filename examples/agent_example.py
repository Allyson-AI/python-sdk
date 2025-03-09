"""
Example of using Allyson's Agent to make requests to OpenAI-compatible chat completion APIs.
"""

import asyncio
import os
import json
from typing import Dict, List, Any

from allyson import Agent


def simple_chat_example(api_key: str):
    """
    Simple example of using the Agent for chat completions.
    
    Args:
        api_key: OpenAI API key
    """
    print("Running simple chat example...")
    
    # Create an agent instance
    agent = Agent(api_key=api_key)
    
    # Define messages for the chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are the three laws of robotics?"}
    ]
    
    # Make a chat completion request
    response = agent.chat_completion(messages=messages)
    
    # Print the response
    print("\nChat Completion Response:")
    print(f"Model used: {response['model']}")
    assistant_message = response["choices"][0]["message"]["content"]
    print(f"Assistant: {assistant_message}")
    print(f"Finish reason: {response['choices'][0]['finish_reason']}")
    print(f"Usage - Prompt tokens: {response['usage']['prompt_tokens']}, Completion tokens: {response['usage']['completion_tokens']}")


def custom_parameters_example(api_key: str):
    """
    Example with custom parameters for the chat completion.
    
    Args:
        api_key: OpenAI API key
    """
    print("\nRunning example with custom parameters...")
    
    # Create an agent with custom parameters
    agent = Agent(
        api_key=api_key,
        model="gpt-3.5-turbo",  # Using a different model
        timeout=30,  # Shorter timeout
        max_retries=2  # Fewer retries
    )
    
    # Define messages for the chat
    messages = [
        {"role": "system", "content": "You are a creative assistant that writes short poems."},
        {"role": "user", "content": "Write a haiku about programming."}
    ]
    
    # Make a chat completion request with custom parameters
    response = agent.chat_completion(
        messages=messages,
        temperature=0.9,  # Higher temperature for more creativity
        max_tokens=50,  # Limit response length
        top_p=0.95,  # Nucleus sampling parameter
        frequency_penalty=0.5  # Reduce repetition
    )
    
    # Print the response
    print("\nChat Completion Response (Custom Parameters):")
    print(f"Model used: {response['model']}")
    assistant_message = response["choices"][0]["message"]["content"]
    print(f"Assistant: {assistant_message}")
    print(f"Finish reason: {response['choices'][0]['finish_reason']}")
    print(f"Usage - Prompt tokens: {response['usage']['prompt_tokens']}, Completion tokens: {response['usage']['completion_tokens']}")


def custom_server_example(api_key: str, base_url: str):
    """
    Example using a custom OpenAI-compatible server.
    
    Args:
        api_key: API key for the custom server
        base_url: Base URL of the custom server
    """
    print("\nRunning example with custom server...")
    
    # Create an agent with a custom server
    agent = Agent(
        api_key=api_key,
        base_url=base_url,
        model="llama3"  # Example model name for a custom server
    )
    
    # Define messages for the chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain how to make a simple HTTP request in Python."}
    ]
    
    try:
        # Make a chat completion request
        response = agent.chat_completion(messages=messages)
        
        # Print the response
        print("\nChat Completion Response (Custom Server):")
        print(f"Model used: {response.get('model', 'unknown')}")
        assistant_message = response["choices"][0]["message"]["content"]
        print(f"Assistant: {assistant_message}")
        
    except Exception as e:
        print(f"Error with custom server: {str(e)}")
        print("Note: This example requires a running OpenAI-compatible server at the specified URL.")


def conversation_example(api_key: str):
    """
    Example of a multi-turn conversation.
    
    Args:
        api_key: OpenAI API key
    """
    print("\nRunning conversation example...")
    
    # Create an agent instance
    agent = Agent(api_key=api_key)
    
    # Start with system and user messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant that provides concise answers."},
        {"role": "user", "content": "What is machine learning?"}
    ]
    
    # First turn
    print("\nUser: What is machine learning?")
    response = agent.chat_completion(messages=messages)
    assistant_message = response["choices"][0]["message"]["content"]
    print(f"Assistant: {assistant_message}")
    
    # Add the assistant's response to the conversation
    messages.append({"role": "assistant", "content": assistant_message})
    
    # Add the user's follow-up question
    messages.append({"role": "user", "content": "What are some common applications?"})
    print("\nUser: What are some common applications?")
    
    # Second turn
    response = agent.chat_completion(messages=messages)
    assistant_message = response["choices"][0]["message"]["content"]
    print(f"Assistant: {assistant_message}")
    
    # Add the assistant's response to the conversation
    messages.append({"role": "assistant", "content": assistant_message})
    
    # Add another user question
    messages.append({"role": "user", "content": "Which one is the most promising for the future?"})
    print("\nUser: Which one is the most promising for the future?")
    
    # Third turn
    response = agent.chat_completion(messages=messages)
    assistant_message = response["choices"][0]["message"]["content"]
    print(f"Assistant: {assistant_message}")


def main():
    """Run the examples."""
    # Get API key from environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key as an environment variable:")
        print("  export OPENAI_API_KEY='your-api-key'")
        return
    
    # Run the examples
    simple_chat_example(api_key)
    custom_parameters_example(api_key)
    
    # Custom server example (commented out by default)
    # Uncomment and set your custom server URL to run this example
    # custom_server_url = "http://localhost:8000/v1"
    # custom_server_example(api_key, custom_server_url)
    
    conversation_example(api_key)


if __name__ == "__main__":
    main() 