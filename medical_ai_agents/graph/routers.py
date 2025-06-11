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
    current_task = state.get("current_task")
    image_path = state.get("image_path", "")
    is_text_only = state.get("is_text_only", False)
    uploaded_docs = state.get("uploaded_documents", [])
    
    logger.info(f"Routing - Type: {task_type}, Required: {required_tasks}, Completed: {completed_tasks}")
    
    # Check for document-related tasks
    if "document_qa" in required_tasks and not uploaded_docs:
        logger.info("Document QA requested but no documents uploaded, removing from tasks")
        required_tasks.remove("document_qa")
        if "document_qa" in execution_order:
            execution_order.remove("document_qa")
    
    # If documents are present, prioritize RAG
    if uploaded_docs:
        valid_docs = []
        for doc in uploaded_docs:
            if os.path.exists(doc):
                valid_docs.append(doc)
            else:
                logger.warning(f"Document not found: {doc}")
        
        if valid_docs:
            logger.info(f"Found {len(valid_docs)} valid documents, prioritizing RAG")
            state["uploaded_documents"] = valid_docs
            return "rag"
    
    # Handle non-medical general queries - route directly to synthesizer
    if "general_query" in required_tasks:
        logger.info("General (non-medical) query detected, routing directly to synthesizer")
        return "synthesizer"
    
    # Text-only medical queries go to VQA
    if is_text_only or not image_path or not os.path.exists(image_path):
        logger.info("Text-only medical query, routing to VQA")
        return "vqa"
    
    # Use the current_task if available, otherwise find next task
    next_task = current_task
    if not next_task:
        for task in execution_order:
            if task not in completed_tasks:
                next_task = task
                break
    
    if not next_task:
        logger.info("All tasks completed or no tasks defined, routing to synthesizer")
        return "synthesizer"
    
    # Route based on next task
    routing_map = {
        "polyp_detection": "detector",
        "modality_classification": "modality_classifier",
        "region_classification": "region_classifier", 
        "medical_qa": "vqa",
        "document_qa": "rag"
    }
    
    target = routing_map.get(next_task, "synthesizer")
    logger.info(f"Next task: {next_task} → routing to: {target}")
    
    return target


#  Post-Agent Routers
def post_detector_router(state: SystemState) -> str:
    """Router after detector - returns only routing decision."""
    logger = logging.getLogger("graph.routers.post_detector")
    
    # DEBUG: Inspect state but DO NOT modify it (LangGraph handles this)
    required_tasks = state.get("required_tasks", [])
    completed_tasks = state.get("completed_tasks", [])
    execution_order = state.get("execution_order", [])
    current_task = state.get("current_task")
    
    logger.info(f"Post-detector: Required={required_tasks}, Completed={completed_tasks}")
    
    # Use current_task if available (coming from _mark_task_completed), otherwise find next
    next_task = current_task
    if not next_task:
        for task in execution_order:
            if task not in completed_tasks and task != "polyp_detection":
                next_task = task
                break
    
    if not next_task:
        logger.info("All tasks completed or no more tasks, routing to synthesizer")
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
    """Router after modality classifier - returns only routing decision."""
    logger = logging.getLogger("graph.routers.post_modality")
    
    # DEBUG: Inspect state but DO NOT modify it (LangGraph handles this)
    required_tasks = state.get("required_tasks", [])
    completed_tasks = state.get("completed_tasks", [])
    execution_order = state.get("execution_order", [])
    current_task = state.get("current_task")
    
    logger.info(f"Post-modality: Required={required_tasks}, Completed={completed_tasks}")
    
    # Use current_task if available, otherwise find next task
    next_task = current_task
    if not next_task:
        for task in execution_order:
            if task not in completed_tasks and task != "modality_classification":
                next_task = task
                break
    
    # Check if we have modality_result to determine if we need synthesis
    modality_result = state.get("modality_result", {})
    
    # Route to synthesizer if we have modality result with low confidence
    # or there are no more tasks
    if modality_result.get("is_low_confidence", False) or not next_task:
        logger.info("Routing to synthesizer for modality result analysis or no more tasks")
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
    """Router after region classifier - returns only routing decision."""
    logger = logging.getLogger("graph.routers.post_region")
    
    # DEBUG: Inspect state but DO NOT modify it (LangGraph handles this)
    required_tasks = state.get("required_tasks", [])
    completed_tasks = state.get("completed_tasks", [])
    execution_order = state.get("execution_order", [])
    current_task = state.get("current_task")
    
    logger.info(f"Post-region: Required={required_tasks}, Completed={completed_tasks}")
    
    # Use current_task if available, otherwise find next task
    next_task = current_task
    if not next_task:
        for task in execution_order:
            if task not in completed_tasks and task != "region_classification":
                next_task = task
                break
    
    # Check if we have region_result to determine if we need synthesis
    region_result = state.get("region_result", {})
    
    # Route to synthesizer if we have region result with low confidence
    # or there are no more tasks
    if region_result.get("is_low_confidence", False) or not next_task:
        logger.info("Routing to synthesizer for region result analysis or no more tasks")
        return "synthesizer"
    
    # Route to next task (likely VQA)
    if next_task == "medical_qa":
        logger.info(f"Post-region next task: medical_qa → vqa")
        return "vqa"
    else:
        logger.info(f"Post-region next task: {next_task} → synthesizer")
        return "synthesizer"


def post_vqa_router(state: SystemState) -> str:
    """Router after VQA - returns only routing decision."""
    logger = logging.getLogger("graph.routers.post_vqa")
    
    # DEBUG: Inspect state but DO NOT modify it (LangGraph handles this)
    completed_tasks = state.get("completed_tasks", [])
    
    logger.info(f"Post-vqa: Completed={completed_tasks}")
    
    # Always route to synthesizer
    logger.info("Routing to synthesizer after VQA")
    return "synthesizer"


def post_vqa_router_with_rag(state: SystemState) -> str:
    """Router after VQA that checks for uploaded documents and routes to RAG if any are found."""
    logger = logging.getLogger("graph.routers.post_vqa_rag")
    
    # Check for uploaded documents
    uploaded_docs = state.get("uploaded_documents", [])
    
    if uploaded_docs:
        # Verify document existence
        valid_docs = []
        for doc in uploaded_docs:
            if os.path.exists(doc):
                valid_docs.append(doc)
            else:
                logger.warning(f"Document not found: {doc}")
        
        if valid_docs:
            logger.info(f"Found {len(valid_docs)} valid documents, routing to RAG")
            return "rag"
    
    # No valid documents, proceed to synthesizer
    logger.info("No valid documents found, proceeding to synthesizer")
    return "synthesizer"


def post_rag_router(state: SystemState) -> str:
    """Router after RAG - returns only routing decision."""
    logger = logging.getLogger("graph.routers.post_rag")
    
    # DEBUG: Inspect state but DO NOT modify it (LangGraph handles this)
    completed_tasks = state.get("completed_tasks", [])
    
    logger.info(f"Post-rag: Completed={completed_tasks}")
    
    # Check if we have RAG results and query complexity
    rag_result = state.get("rag_result", {})
    query_complexity = rag_result.get("query_complexity", "simple")
    
    # Route based on complexity
    if query_complexity == "complex":
        logger.info("Complex medical query detected, routing to VQA for specialized knowledge")
        return "vqa"
    else:
        logger.info("Simple document query, skipping VQA and going to synthesizer")
        return "synthesizer"