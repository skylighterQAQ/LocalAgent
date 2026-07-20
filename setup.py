from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="local-agent",
    version="0.1.0",
    description="A local AI agent framework based on Ollama and LangGraph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LocalAgent Team",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "local-agent=local_agent.cli.main:app",
            "la=local_agent.cli.main:app",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
