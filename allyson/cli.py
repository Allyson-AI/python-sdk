"""
Command-line interface for Allyson.
"""

import argparse
import asyncio
import importlib.metadata
import logging
import sys
from typing import List, Optional

from allyson.browser import Browser

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def get_version() -> str:
    """Get the version of Allyson."""
    try:
        return importlib.metadata.version("allyson")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Allyson: AI-powered web browser automation using Playwright."
    )
    
    parser.add_argument(
        "--version", action="store_true", help="Show version and exit"
    )
    
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Install command
    install_parser = subparsers.add_parser(
        "install", help="Install Playwright browsers"
    )
    
    # Open command
    open_parser = subparsers.add_parser(
        "open", help="Open a browser and navigate to a URL"
    )
    open_parser.add_argument("url", help="URL to navigate to")
    open_parser.add_argument(
        "--browser", 
        choices=["chromium", "firefox", "webkit"], 
        default="chromium",
        help="Browser to use (default: chromium)"
    )
    open_parser.add_argument(
        "--headless", 
        action="store_true", 
        help="Run browser in headless mode"
    )
    
    # Screenshot command
    screenshot_parser = subparsers.add_parser(
        "screenshot", help="Take a screenshot of a webpage"
    )
    screenshot_parser.add_argument("url", help="URL to navigate to")
    screenshot_parser.add_argument(
        "output", help="Path to save the screenshot"
    )
    screenshot_parser.add_argument(
        "--browser", 
        choices=["chromium", "firefox", "webkit"], 
        default="chromium",
        help="Browser to use (default: chromium)"
    )
    screenshot_parser.add_argument(
        "--full-page", 
        action="store_true", 
        help="Take a screenshot of the full page"
    )
    
    # Run script command
    script_parser = subparsers.add_parser(
        "run", help="Run a Python script with Allyson"
    )
    script_parser.add_argument("script", help="Path to the script to run")
    script_parser.add_argument(
        "args", 
        nargs="*", 
        help="Arguments to pass to the script"
    )
    
    return parser.parse_args(args)


async def install_browsers():
    """Install Playwright browsers."""
    import subprocess
    import sys
    
    print("Installing Playwright browsers...")
    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"Error installing browsers: {result.stderr}")
        return False
    
    print("Browsers installed successfully.")
    return True


async def open_browser(url: str, browser_type: str = "chromium", headless: bool = False):
    """Open a browser and navigate to a URL."""
    async with Browser(browser_type=browser_type, headless=headless) as browser:
        print(f"Navigating to {url}...")
        await browser.agoto(url)
        print("Press Ctrl+C to close the browser.")
        
        try:
            # Keep the browser open until interrupted
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Closing browser...")


async def take_screenshot(
    url: str, 
    output_path: str, 
    browser_type: str = "chromium", 
    full_page: bool = False
):
    """Take a screenshot of a webpage."""
    async with Browser(browser_type=browser_type, headless=True) as browser:
        print(f"Navigating to {url}...")
        await browser.agoto(url)
        
        print(f"Taking {'full page ' if full_page else ''}screenshot...")
        await browser.ascreenshot(output_path, full_page=full_page)
        
        print(f"Screenshot saved to {output_path}")


async def run_script(script_path: str, script_args: List[str]):
    """Run a Python script with Allyson."""
    import importlib.util
    import os
    import sys
    
    # Add the script's directory to sys.path
    script_dir = os.path.dirname(os.path.abspath(script_path))
    sys.path.insert(0, script_dir)
    
    # Load the script as a module
    spec = importlib.util.spec_from_file_location("script", script_path)
    if spec is None or spec.loader is None:
        print(f"Error: Could not load script {script_path}")
        return
    
    module = importlib.util.module_from_spec(spec)
    sys.argv = [script_path] + script_args
    
    try:
        spec.loader.exec_module(module)
        
        # If the script has a main function, call it
        if hasattr(module, "main"):
            if asyncio.iscoroutinefunction(module.main):
                await module.main()
            else:
                module.main()
    except Exception as e:
        print(f"Error running script: {e}")
        import traceback
        traceback.print_exc()


async def async_main(args: argparse.Namespace):
    """Main async function."""
    if args.command == "install":
        await install_browsers()
    elif args.command == "open":
        await open_browser(args.url, args.browser, args.headless)
    elif args.command == "screenshot":
        await take_screenshot(args.url, args.output, args.browser, args.full_page)
    elif args.command == "run":
        await run_script(args.script, args.args)


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    if args.version:
        print(f"Allyson version {get_version()}")
        return
    
    if not args.command:
        print("No command specified. Use --help for usage information.")
        return
    
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main() 