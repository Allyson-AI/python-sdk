"""
Agent module for making requests to OpenAI-compatible chat completion servers.

This module provides a lightweight client for interacting with OpenAI-compatible chat completion APIs
using only the requests library to keep dependencies minimal.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import requests

logger = logging.getLogger(__name__)


class Agent:
    """
    Agent class for making requests to OpenAI-compatible chat completion servers.
    
    This class provides a simple interface for sending chat completion requests to OpenAI API
    or compatible servers (like local LLM servers) using only the requests library.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        max_retries: int = 3,
        model: str = "gpt-4o",
    ):
        """
        Initialize a new Agent instance.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL of the API (default: OpenAI's API)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            model: Default model to use for completions
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
    
    def _make_request(
        self, 
        endpoint: str, 
        method: str = "POST", 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the API.
        
        Args:
            endpoint: API endpoint (without base URL)
            method: HTTP method (GET, POST, etc.)
            data: Request payload
            params: Query parameters
            
        Returns:
            API response as a dictionary
            
        Raises:
            Exception: If the request fails after max_retries
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                if method.upper() == "GET":
                    response = self.session.get(
                        url, 
                        params=params, 
                        timeout=self.timeout
                    )
                else:
                    response = self.session.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        timeout=self.timeout
                    )
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed to make request after {self.max_retries} attempts: {str(e)}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion.
        
        Args:
            messages: List of message objects with role and content
            model: Model to use (defaults to instance default)
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty parameter
            presence_penalty: Presence penalty parameter
            stop: Stop sequences
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response
        """
        data = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        
        # Add optional parameters if provided
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if top_p is not None:
            data["top_p"] = top_p
        if frequency_penalty is not None:
            data["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            data["presence_penalty"] = presence_penalty
        if stop is not None:
            data["stop"] = stop
            
        # Add any additional kwargs
        data.update(kwargs)
        
        return self._make_request("chat/completions", data=data) 