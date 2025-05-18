#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Agents Initialization
---------------------------------------
Initialization file for the agents package.
"""

# Import agent classes
from medical_ai_system.agents.base_agent import BaseAgent
from medical_ai_system.agents.detector import DetectorAgent
from medical_ai_system.agents.classifier import ClassifierAgent
from medical_ai_system.agents.vqa import VQAAgent

__all__ = [
    'BaseAgent',
    'DetectorAgent',
    'ClassifierAgent',
    'VQAAgent'
]