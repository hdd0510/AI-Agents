# setup.py
from setuptools import setup, find_packages

setup(
    name="llava",
    version="0.1",
    packages=find_packages(where="medical_ai_agents/tools/vqa"),
    package_dir={"": "medical_ai_agents/tools/vqa"},
)
