#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Graph - Routers
-----------------------
Các hàm router để điều hướng luồng dữ liệu trong LangGraph.
"""

import logging
from typing import Dict, Any, List

from medical_ai_system.config import SystemState, TaskType

# Router to determine next step based on task type
def task_router(state: SystemState) -> str:
    """Routes to the next step based on task type."""
    logger = logging.getLogger("graph.routers.task_router")
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    
    logger.info(f"Routing based on task type: {task_type}")
    
    if task_type == TaskType.POLYP_DETECTION:
        return "detector"
    elif task_type == TaskType.MODALITY_CLASSIFICATION:
        return "modality_classifier"
    elif task_type == TaskType.REGION_CLASSIFICATION:
        return "region_classifier"
    elif task_type == TaskType.MEDICAL_QA:
        # For medical QA, start with detector for context
        return "detector"
    else:  # COMPREHENSIVE
        return "detector"


# Router after detection
def post_detector_router(state: SystemState) -> str:
    """Routes to the next step after detection based on task type."""
    logger = logging.getLogger("graph.routers.post_detector")
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    
    logger.info(f"Post-detector routing for task type: {task_type}")
    
    if task_type == TaskType.POLYP_DETECTION:
        return "synthesizer"
    elif task_type == TaskType.MEDICAL_QA:
        return "vqa"
    else:  # For other types including COMPREHENSIVE
        # First go to modality classifier
        return "modality_classifier"


# Router after modality classification
def post_modality_router(state: SystemState) -> str:
    """Routes to the next step after modality classification."""
    logger = logging.getLogger("graph.routers.post_modality")
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    
    logger.info(f"Post-modality routing for task type: {task_type}")
    
    if task_type == TaskType.MODALITY_CLASSIFICATION:
        return "synthesizer"
    elif task_type == TaskType.COMPREHENSIVE:
        return "region_classifier"
    else:
        # Should not happen, but just in case
        return "synthesizer"


# Router after region classification
def post_region_router(state: SystemState) -> str:
    """Routes to the next step after region classification."""
    logger = logging.getLogger("graph.routers.post_region")
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    
    logger.info(f"Post-region routing for task type: {task_type}")
    
    if task_type == TaskType.REGION_CLASSIFICATION:
        return "synthesizer"
    elif task_type == TaskType.COMPREHENSIVE and state.get("query"):
        return "vqa"
    else:
        return "synthesizer"


# Router after VQA
def post_vqa_router(state: SystemState) -> str:
    """Routes to the next step after VQA."""
    logger = logging.getLogger("graph.routers.post_vqa")
    
    # Check if reflection is needed and available
    reflection_available = "reflection" in state
    needs_reflection = _needs_reflection(state)
    
    logger.info(f"Post-VQA routing. Needs reflection: {needs_reflection}, Reflection available: {reflection_available}")
    
    if needs_reflection and reflection_available:
        return "reflection"
    else:
        return "synthesizer"


def _needs_reflection(state: SystemState) -> bool:
    """Determines if reflection is needed based on VQA result."""
    vqa_result = state.get("vqa_result", {})
    
    # No reflection needed if VQA failed
    if not vqa_result or not vqa_result.get("success", False):
        return False
    
    # Check confidence - low confidence needs reflection
    confidence = vqa_result.get("confidence", 1.0)
    if confidence < 0.7:
        return True
    
    # Check for uncertainty in answer
    answer = vqa_result.get("answer", "").lower()
    uncertainty_phrases = ["có thể", "không chắc chắn", "khó xác định", "có lẽ"]
    if any(phrase in answer for phrase in uncertainty_phrases):
        return True
    
    return False