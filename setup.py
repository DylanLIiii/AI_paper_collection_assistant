from setuptools import setup, find_packages

setup(
    name="paper_assistant",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "arxiv>=2.1.3,<3",
        "feedparser>=6.0.11,<7",
        "loguru>=0.7.3,<0.8",
        "instructor>=1.7.2,<2",
        "markitdown>=0.0.1a3,<0.0.1a4",
        "litellm>=1.57.5,<2",
        "retry>=0.9.2,<0.10",
        "slack-sdk>=3.34.0,<4",
        "flask>=3.1.0,<4",
        "markdown>=3.7,<4",
        "python-markdown-math>=0.8,<0.9",
        "pytz>=2024.1"
    ],
    python_requires=">=3.8",
    package_data={
        "paper_assistant": [
            "config/*",
            "templates/*",
        ],
    },
    entry_points={
        "console_scripts": [
            "paper-assistant=paper_assistant.cli.commands:main",
        ],
    },
)
