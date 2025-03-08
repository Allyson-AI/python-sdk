# Allyson

AI-powered web browser automation using Playwright.

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 