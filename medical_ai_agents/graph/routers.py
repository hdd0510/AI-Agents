#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Graph - Routers (MODIFIED for text-only support)
-----------------------
Các hàm router để điều hướng luồng dữ liệu trong LangGraph.
"""

import logging
import os
from typing import Dict, Any, List

from medical_ai_agents.config import SystemState, TaskType

# Router to determine next step based on task type
def task_router(state: SystemState) -> str:
    """Routes to the next step based on task type and input type."""
    logger = logging.getLogger("graph.routers.task_router")
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    image_path = state.get("image_path", "")
    query = state.get("query", "")
    is_text_only = state.get("is_text_only", False)
    
    logger.info(f"Routing - Task: {task_type}, Has Image: {bool(image_path and os.path.exists(image_path))}, Is Text Only: {is_text_only}")
    
    # MODIFIED: Handle text-only mode
    if is_text_only or not image_path or not os.path.exists(image_path):
        if query and query.strip():
            logger.info("Text-only mode detected, routing to VQA for text-based medical consultation")
            return "vqa"  # Route directly to VQA for text-only queries
        else:
            logger.warning("No valid input provided")
            return "synthesizer"  # Go to synthesizer to handle error
    
    # Original logic for image-based analysis
    if task_type == TaskType.POLYP_DETECTION:
        return "detector"
    elif task_type == TaskType.MODALITY_CLASSIFICATION:
        return "modality_classifier"
    elif task_type == TaskType.REGION_CLASSIFICATION:
        return "region_classifier"
    elif task_type == TaskType.MEDICAL_QA:
        return "detector"  # Start with detector for context
    else:  # COMPREHENSIVE
        return "detector"


# Router after detection
def post_detector_router(state: SystemState) -> str:
    """Routes after detection based on task type and query."""
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    query = state.get("query", "")
    
    # Nếu có query (dù task là polyp_detection), vẫn cần VQA để answer
    if query and query.strip():
        return "vqa"
    elif task_type == TaskType.POLYP_DETECTION:
        return "synthesizer"
    else:
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