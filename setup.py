from setuptools import setup, find_packages

setup(
    name="allyson",
    version="0.1.5",
    packages=find_packages(),
    install_requires=[
        "playwright>=1.40.0",
        "pytest-playwright>=0.4.0",
        "tqdm>=4.66.0",
        "pydantic>=2.0.0",
        "pytest-asyncio>=0.21.0",
        "pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "pytest",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "allyson=allyson.cli:main",
        ],
    },
    author="Allyson Team",
    author_email="info@allyson.ai",
    description="AI-powered web browser automation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Allyson-AI/python-sdk",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
) 