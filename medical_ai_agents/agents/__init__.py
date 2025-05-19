#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Agents Initialization
---------------------------------------
Initialization file for the agents package.
"""

# Import agent classes
from medical_ai_agents.agents.base_agent import BaseAgent
from medical_ai_agents.agents.detector import DetectorAgent
from medical_ai_agents.agents.classifier import ClassifierAgent
from medical_ai_agents.agents.vqa import VQAAgent

__all__ = [
    'BaseAgent',
    'DetectorAgent',
    'ClassifierAgent',
    'VQAAgent'
]