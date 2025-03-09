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

- **0.1.2** - Updated Description
- **0.1.1** - Test release for GitHub Actions automated publishing
- **0.1.0** - Initial release