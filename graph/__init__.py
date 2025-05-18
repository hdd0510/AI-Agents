#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Graph Initialization
--------------------------------------
Initialization file for the graph package.
"""

# Import graph components
from medical_ai_system.graph.nodes import task_analyzer, reflection_node, result_synthesizer
from medical_ai_system.graph.routers import (
    task_router, post_detector_router, post_modality_router, 
    post_region_router, post_vqa_router
)
from medical_ai_system.graph.pipeline import create_medical_ai_graph

__all__ = [
    'task_analyzer',
    'reflection_node',
    'result_synthesizer',
    'task_router',
    'post_detector_router',
    'post_modality_router',
    'post_region_router',
    'post_vqa_router',
    'create_medical_ai_graph'
]