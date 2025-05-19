#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Setup Script
------------------------------
Script cài đặt cho hệ thống AI y tế đa agent.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="medical_ai_agents",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Hệ thống AI y tế đa agent sử dụng LangGraph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hdd0510/medical_ai_agents",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langgraph>=0.4.5",
        "langgraph-checkpoint>=0.0.1",
        "langchain>=0.0.267",
        "langchain-openai>=0.0.2",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "pillow>=9.0.0",
        "numpy>=1.20.0",
        "opencv-python>=4.5.0",
        "ultralytics>=8.0.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.60.0",
        "pydantic>=2.0.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.5",
    ],
    entry_points={
        "console_scripts": [
            "medical-ai=medical_ai_agents.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt"],
    },
)