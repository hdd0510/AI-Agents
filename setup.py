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
        # Core ML dependencies
        "torch==2.1.2",
        "torchvision==0.16.2",
        "transformers==4.37.2",
        "tokenizers==0.15.1",
        "sentencepiece==0.1.99",
        "accelerate==0.21.0",
        "peft==0.7.1",
        "bitsandbytes==0.41.3",
        
        # LangChain ecosystem - updated versions
        "langchain==0.2.0",
        "langchain-core>=0.2.38",
        "langchain-community>=0.0.40",
        "langchain-openai>=0.1.0",
        "langchain-text-splitters>=0.0.1",
        "langgraph>=0.4.5",
        "langgraph-checkpoint==2.0.26",
        "pydantic>=2.7.4,<3.0.0",  # Updated to match langgraph requirement
        
        # Medical AI + common utils
        "numpy==1.24.3",
        "opencv-python==4.8.1.78",
        "pillow==10.1.0",
        "matplotlib==3.8.2",
        "tqdm==4.66.1",
        "scikit-learn==1.2.2",
        
        "uvicorn==0.22.0",
        "python-multipart==0.0.5",
        "requests==2.31.0",
        "httpx==0.24.0",
        

        
        # Other dependencies
        "protobuf==3.20.0",
        "shortuuid==1.0.11",
        "markdown2[all]==2.4.10",
        "einops==0.6.1",
        "einops-exts==0.0.4",
        "timm==0.6.13"
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