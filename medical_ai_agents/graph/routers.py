#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Graph -  Routers (MODIFIED for multi-task support)
-----------------------
 routing logic với multi-task execution support.
"""

import logging
import os
from typing import Dict, Any, List

from medical_ai_agents.config import SystemState, TaskType
from medical_ai_agents.graph.nodes import _mark_task_completed

#  Task Router với Multi-Task Support
def task_router(state: SystemState) -> str:
    """ router với multi-task execution logic."""
    logger = logging.getLogger("graph.routers.task_router")
    
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    required_tasks = state.get("required_tasks", [])
    completed_tasks = state.get("completed_tasks", [])
    execution_order = state.get("execution_order", [])
    image_path = state.get("image_path", "")
    is_text_only = state.get("is_text_only", False)
    
    logger.info(f"Routing - Type: {task_type}, Required: {required_tasks}, Completed: {completed_tasks}")
    
    # Text-only queries always go to VQA
    if is_text_only or not image_path or not os.path.exists(image_path):
        logger.info("Text-only mode, routing to VQA")
        return "vqa"
    
    # Find next task to execute
    next_task = None
    for task in execution_order:
        if task not in completed_tasks:
            next_task = task
            break
    
    if not next_task:
        logger.info("All tasks completed, routing to synthesizer")
        return "synthesizer"
    
    # Route based on next task
    routing_map = {
        "polyp_detection": "detector",
        "modality_classification": "modality_classifier",
        "region_classification": "region_classifier", 
        "medical_qa": "vqa"
    }
    
    target = routing_map.get(next_task, "synthesizer")
    logger.info(f"Next task: {next_task} → routing to: {target}")
    
    return target


#  Post-Agent Routers
def post_detector_router(state: SystemState) -> str:
    """ router after detector với multi-task logic."""
    logger = logging.getLogger("graph.routers.post_detector")
    
    # Mark detector task as completed
    state = _mark_task_completed(state, "polyp_detection")
    
    required_tasks = state.get("required_tasks", [])
    completed_tasks = state.get("completed_tasks", [])
    execution_order = state.get("execution_order", [])
    
    logger.info(f"Post-detector: Required={required_tasks}, Completed={completed_tasks}")
    
    # Find next task
    next_task = None
    for task in execution_order:
        if task not in completed_tasks:
            next_task = task
            break
    
    if not next_task:
        return "synthesizer"
    
    # Route to next task
    routing_map = {
        "modality_classification": "modality_classifier",
        "region_classification": "region_classifier",
        "medical_qa": "vqa"
    }
    
    target = routing_map.get(next_task, "synthesizer")
    logger.info(f"Post-detector next task: {next_task} → {target}")
    
    return target


def post_modality_router(state: SystemState) -> str:
    """Enhanced router after modality classifier."""
    logger = logging.getLogger("graph.routers.post_modality")
    
    # Mark modality task as completed
    state = _mark_task_completed(state, "modality_classification")
    
    required_tasks = state.get("required_tasks", [])
    completed_tasks = state.get("completed_tasks", [])
    execution_order = state.get("execution_order", [])
    
    logger.info(f"Post-modality: Required={required_tasks}, Completed={completed_tasks}")
    
    # Find next task
    next_task = None
    for task in execution_order:
        if task not in completed_tasks:
            next_task = task
            break
    
    if not next_task:
        return "synthesizer"
    
    # Route to next task
    routing_map = {
        "region_classification": "region_classifier",
        "medical_qa": "vqa"
    }
    
    target = routing_map.get(next_task, "synthesizer")
    logger.info(f"Post-modality next task: {next_task} → {target}")
    
    return target


def post_region_router(state: SystemState) -> str:
    """Enhanced router after region classifier."""
    logger = logging.getLogger("graph.routers.post_region")
    
    # Mark region task as completed
    state = _mark_task_completed(state, "region_classification")
    
    required_tasks = state.get("required_tasks", [])
    completed_tasks = state.get("completed_tasks", [])
    execution_order = state.get("execution_order", [])
    
    logger.info(f"Post-region: Required={required_tasks}, Completed={completed_tasks}")
    
    # Find next task
    next_task = None
    for task in execution_order:
        if task not in completed_tasks:
            next_task = task
            break
    
    if not next_task:
        return "synthesizer"
    
    # Route to next task (likely VQA)
    if next_task == "medical_qa":
        return "vqa"
    else:
        return "synthesizer"

def post_vqa_router(state: SystemState) -> str:
    """Enhanced router after VQA - FIXED VERSION"""
    logger = logging.getLogger("graph.routers.post_vqa")
    
    print(f"🔧 DEBUG: post_vqa_router input - completed_tasks: {state.get('completed_tasks', [])}")
    
    # Mark medical_qa as completed
    _mark_task_completed(state, "medical_qa")
    
    print(f"🔧 DEBUG: post_vqa_router after marking - completed_tasks: {state.get('completed_tasks', [])}")
    
    logger.info("Going directly to synthesizer")
    return "synthesizer"