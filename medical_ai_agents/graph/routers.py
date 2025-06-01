#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Graph - Routers (FIXED: Proper Task-Based Logic)
-----------------------
Các hàm router với logic routing được sửa chữa hợp lý.
"""

import logging
import os
from typing import Dict, Any, List

from medical_ai_agents.config import SystemState, TaskType

# ===== FIXED TASK ROUTER =====
def task_router(state: SystemState) -> str:
    """Routes to the next step based on task type and input type."""
    logger = logging.getLogger("graph.routers.task_router")
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    image_path = state.get("image_path", "")
    query = state.get("query", "")
    is_text_only = state.get("is_text_only", False)
    
    logger.info(f"Routing - Task: {task_type}, Has Image: {bool(image_path and os.path.exists(image_path))}, Is Text Only: {is_text_only}")
    
    # Text-only queries always go to VQA (to use LLaVA)
    if is_text_only or not image_path or not os.path.exists(image_path):
        if query and query.strip():
            logger.info("Text-only mode detected, routing to VQA for LLaVA-based consultation")
            return "vqa"
        else:
            logger.warning("No valid input provided")
            return "synthesizer"
    
    # Image-based analysis - FIXED LOGIC by task type
    if task_type == TaskType.POLYP_DETECTION:
        logger.info("Task: POLYP_DETECTION → detector")
        return "detector"
    elif task_type == TaskType.MODALITY_CLASSIFICATION:
        logger.info("Task: MODALITY_CLASSIFICATION → modality_classifier (skip detector)")
        return "modality_classifier"  # Go directly to modality, no need detector
    elif task_type == TaskType.REGION_CLASSIFICATION:
        logger.info("Task: REGION_CLASSIFICATION → region_classifier (skip detector)")
        return "region_classifier"  # Go directly to region, no need detector
    elif task_type == TaskType.MEDICAL_QA:
        logger.info("Task: MEDICAL_QA → detector (for context)")
        return "detector"  # Need detector for medical context
    else:  # COMPREHENSIVE
        logger.info("Task: COMPREHENSIVE → detector (full pipeline)")
        return "detector"  # Start full pipeline


# ===== FIXED POST-DETECTOR ROUTER =====
def post_detector_router(state: SystemState) -> str:
    """Routes after detection based on task type and query."""
    logger = logging.getLogger("graph.routers.post_detector")
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    query = state.get("query", "")
    
    logger.info(f"Post-detector routing for task: {task_type}, has query: {bool(query)}")
    
    # LOGIC: Detector chỉ chạy cho POLYP_DETECTION, MEDICAL_QA, và COMPREHENSIVE
    # Vì vậy post-detector logic phải match với điều này
    
    if task_type == TaskType.POLYP_DETECTION:
        if query and query.strip():
            logger.info("POLYP_DETECTION with query → VQA for explanation")
            return "vqa"
        else:
            logger.info("POLYP_DETECTION without query → synthesizer")
            return "synthesizer"
    
    elif task_type == TaskType.MEDICAL_QA:
        logger.info("MEDICAL_QA after detection → VQA for question answering")
        return "vqa"  # Always go to VQA for medical Q&A
    
    elif task_type == TaskType.COMPREHENSIVE:
        logger.info("COMPREHENSIVE after detection → modality_classifier (continue pipeline)")
        return "modality_classifier"  # Continue full pipeline
    
    else:
        # Shouldn't reach here if task_router is correct
        logger.warning(f"Unexpected task_type {task_type} after detector, going to synthesizer")
        return "synthesizer"


# ===== FIXED POST-MODALITY ROUTER =====
def post_modality_router(state: SystemState) -> str:
    """Routes to the next step after modality classification."""
    logger = logging.getLogger("graph.routers.post_modality")
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    
    logger.info(f"Post-modality routing for task type: {task_type}")
    
    # LOGIC: Modality classifier chỉ chạy cho COMPREHENSIVE
    # (vì các task khác đi thẳng đến target classifier)
    
    if task_type == TaskType.COMPREHENSIVE:
        logger.info("COMPREHENSIVE after modality → region_classifier")
        return "region_classifier"  # Continue to region classification
    else:
        # Shouldn't reach here normally, but handle gracefully
        logger.warning(f"Unexpected task_type {task_type} after modality, going to synthesizer")
        return "synthesizer"


# ===== FIXED POST-REGION ROUTER =====
def post_region_router(state: SystemState) -> str:
    """Routes to the next step after region classification."""
    logger = logging.getLogger("graph.routers.post_region")
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    query = state.get("query", "")
    
    logger.info(f"Post-region routing for task type: {task_type}")
    
    if task_type == TaskType.REGION_CLASSIFICATION:
        logger.info("REGION_CLASSIFICATION complete → synthesizer")
        return "synthesizer"  # Task complete
    
    elif task_type == TaskType.COMPREHENSIVE:
        if query and query.strip():
            logger.info("COMPREHENSIVE with query → VQA for final analysis")
            return "vqa"  # Answer user's question
        else:
            logger.info("COMPREHENSIVE without query → synthesizer")
            return "synthesizer"  # Just synthesis detection + classification
    
    else:
        logger.warning(f"Unexpected task_type {task_type} after region, going to synthesizer")
        return "synthesizer"


# ===== POST-VQA ROUTER (unchanged) =====
def post_vqa_router(state: SystemState) -> str:
    """Routes to the next step after VQA."""
    logger = logging.getLogger("graph.routers.post_vqa")
    
    # Check if reflection is needed and available
    reflection_available = "reflection" in state
    needs_reflection = _needs_reflection(state)
    
    logger.info(f"Post-VQA routing. Needs reflection: {needs_reflection}, Reflection available: {reflection_available}")
    
    # For LLaVA-based processing, consider reflection less critical
    vqa_result = state.get("vqa_result", {})
    query_type = vqa_result.get("query_type", "unknown")
    
    if query_type == "text_only":
        # Text-only queries processed by LLaVA typically don't need reflection
        logger.info("Text-only LLaVA processing complete, going to synthesizer")
        return "synthesizer"
    elif needs_reflection and reflection_available:
        logger.info("Image-based query needs reflection")
        return "reflection"
    else:
        logger.info("Going directly to synthesizer")
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