#!/usr/bin/env python3
"""
Script to install and run Allyson.
"""

import os
import subprocess
import sys


def run_command(command, description=None):
    """Run a command and print its output."""
    if description:
        print(f"\n{description}...")
    
    process = subprocess.run(
        command,
        shell=True,
        text=True,
        capture_output=True,
    )
    
    if process.returncode != 0:
        print(f"Error: {process.stderr}")
        sys.exit(1)
    
    print(process.stdout)
    return process.stdout


def main():
    """Install and run Allyson."""
    # Check if we're in a virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        print("It's recommended to run this script in a virtual environment.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(0)
    
    # Install the package in development mode
    run_command(
        f"{sys.executable} -m pip install -e .",
        "Installing Allyson in development mode"
    )
    
    # Install Playwright browsers
    run_command(
        f"{sys.executable} -m playwright install",
        "Installing Playwright browsers"
    )
    
    # Run the simple example
    print("\nRunning simple example...")
    example_path = os.path.join("examples", "simple_example.py")
    if not os.path.exists(example_path):
        print(f"Error: Example file not found: {example_path}")
        sys.exit(1)
    
    try:
        subprocess.run(
            [sys.executable, example_path],
            check=True,
        )
    except subprocess.CalledProcessError:
        print("Error running the example.")
        sys.exit(1)
    
    print("\nAllyson has been successfully installed and tested!")
    print("\nYou can now use Allyson in your Python scripts:")
    print("\nfrom allyson import Browser\n")
    print("with Browser() as browser:")
    print("    browser.goto('https://example.com')")
    print("    browser.click('Link text')")
    print("    browser.screenshot('screenshot.png')")


if __name__ == "__main__":
    main() 