#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Agents Initialization
---------------------------------------
Initialization file for the agents package.
"""

# Import agent classes
from agents.base_agent import BaseAgent
from agents.detector import DetectorAgent
from agents.classifier import ClassifierAgent
from agents.vqa import VQAAgent

__all__ = [
    'BaseAgent',
    'DetectorAgent',
    'ClassifierAgent',
    'VQAAgent'
]