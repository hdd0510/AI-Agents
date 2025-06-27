#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Graph -  Routers (MODIFIED for multi-task support)
-----------------------
 routing logic với multi-task execution support.
"""

import logging
import os
from typing import Dict, Any, List, Annotated, TypedDict, Optional

from medical_ai_agents.config import SystemState, TaskType
from medical_ai_agents.graph.nodes import _mark_task_completed

#  Task Router với Multi-Task Support
def task_router(state: SystemState) -> str:
    """ router với multi-task execution logic."""
    logger = logging.getLogger("graph.routers.task_router")
    
    # Kiểm tra xem conversation_history được truyền đúng hay không
    if "conversation_history" in state:
        conv_history = state["conversation_history"]
        logger.info(f"[DEBUG] Router received conversation history with {len(conv_history)} entries")
        if conv_history:
            # Log một vài entry đầu và cuối
            for i, entry in enumerate(conv_history[:2]):
                logger.info(f"[DEBUG] History entry {i}: query='{entry.get('query', '')[:30]}...', response='{entry.get('response', '')[:30]}...'")
            
            if len(conv_history) > 2:
                for i, entry in enumerate(conv_history[-2:], start=len(conv_history)-2):
                    logger.info(f"[DEBUG] History entry {i}: query='{entry.get('query', '')[:30]}...', response='{entry.get('response', '')[:30]}...'")
    else:
        logger.warning("[DEBUG] Router did not receive conversation_history in state")
    
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    required_tasks = state.get("required_tasks", [])
    completed_tasks = state.get("completed_tasks", [])
    execution_order = state.get("execution_order", [])
    current_task = state.get("current_task")
    image_path = state.get("image_path", "")
    is_text_only = state.get("is_text_only", False)
    uploaded_docs = state.get("uploaded_documents", [])
    embedding_result = state.get("embedding_result", {})
    
    logger.info(f"Routing - Type: {task_type}, Required: {required_tasks}, Completed: {completed_tasks}")
    
    # BƯỚC 1: Kiểm tra xem có tài liệu kèm theo và đã được xử lý chưa
    if uploaded_docs or embedding_result.get("success", False):
        # Kiểm tra xem có tài liệu hợp lệ
        valid_docs = []
        if uploaded_docs:
            for doc in uploaded_docs:
                if os.path.exists(doc):
                    valid_docs.append(doc)
                else:
                    logger.warning(f"Document not found: {doc}")
        
        # Nếu có tài liệu hợp lệ hoặc embedding thành công, ưu tiên RAG
        if valid_docs or embedding_result.get("success", False) or embedding_result.get("has_existing_documents", False):
            logger.info(f"Documents available - prioritizing RAG processing")
            # Thêm document_qa vào required tasks nếu chưa có
            if "document_qa" not in required_tasks:
                required_tasks.append("document_qa")
                if "document_qa" not in execution_order:
                    execution_order.append("document_qa")
            
            # Luôn route đến RAG trước khi tài liệu có sẵn
            if "document_qa" not in completed_tasks:
                state["current_task"] = "document_qa"
                return "rag"
    
    # BƯỚC 2: Xem xét các tác vụ khác sau khi ưu tiên RAG
    # Xử lý các truy vấn không y tế - route trực tiếp đến synthesizer
    if "general_query" in required_tasks:
        logger.info("General (non-medical) query detected, routing directly to synthesizer")
        return "synthesizer"
    
    # BƯỚC 3: Xử lý các truy vấn y tế dạng văn bản
    if is_text_only or not image_path or not os.path.exists(image_path):
        logger.info("Text-only medical query, routing to VQA")
        return "vqa"
    
    # BƯỚC 4: Xử lý theo current_task hoặc tìm tác vụ tiếp theo
    next_task = current_task
    if not next_task:
        for task in execution_order:
            if task not in completed_tasks:
                next_task = task
                break
    
    if not next_task:
        logger.info("All tasks completed or no tasks defined, routing to synthesizer")
        return "synthesizer"
    
    # Route dựa trên next task
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
    uploaded_docs = state.get("uploaded_documents", [])
    embedding_result = state.get("embedding_result", {})
    
    logger.info(f"Post-detector: Required={required_tasks}, Completed={completed_tasks}")
    
    # MODIFICATION: Check if RAG is needed and not yet completed
    if ("document_qa" in required_tasks or uploaded_docs or embedding_result.get("success", False)) and "document_qa" not in completed_tasks:
        logger.info("Documents available, routing to RAG first")
        return "rag"
    
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
    uploaded_docs = state.get("uploaded_documents", [])
    embedding_result = state.get("embedding_result", {})
    
    logger.info(f"Post-modality: Required={required_tasks}, Completed={completed_tasks}")
    
    # MODIFICATION: Check if RAG is needed and not yet completed
    if ("document_qa" in required_tasks or uploaded_docs or embedding_result.get("success", False)) and "document_qa" not in completed_tasks:
        logger.info("Documents available, routing to RAG first")
        return "rag"
    
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
    uploaded_docs = state.get("uploaded_documents", [])
    embedding_result = state.get("embedding_result", {})
    
    logger.info(f"Post-region: Required={required_tasks}, Completed={completed_tasks}")
    
    # MODIFICATION: Check if RAG is needed and not yet completed
    if ("document_qa" in required_tasks or uploaded_docs or embedding_result.get("success", False)) and "document_qa" not in completed_tasks:
        logger.info("Documents available, routing to RAG first")
        return "rag"
    
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
    required_tasks = state.get("required_tasks", [])
    completed_tasks = state.get("completed_tasks", [])
    uploaded_docs = state.get("uploaded_documents", [])
    embedding_result = state.get("embedding_result", {})
    
    logger.info(f"Post-vqa: Completed={completed_tasks}")
    
    # MODIFICATION: Check if RAG is needed and not yet completed
    if ("document_qa" in required_tasks or uploaded_docs or embedding_result.get("success", False)) and "document_qa" not in completed_tasks:
        logger.info("Documents available, routing to RAG")
        return "rag"
    
    # Always route to synthesizer if no RAG required
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
    conversation_history = state.get("conversation_history", [])
    uploaded_docs = state.get("uploaded_documents", [])
    embedding_result = state.get("embedding_result", {})
    rag_result = state.get("rag_result", {})
    
    logger.info(f"Post-rag: Completed={completed_tasks}")
    
    # Kiểm tra kết quả RAG
    if rag_result:
        # Kiểm tra độ phức tạp của truy vấn
        query_complexity = rag_result.get("query_complexity", "simple")
        vqa_output = rag_result.get("vqa_output")  # RAG có thể đề xuất chuyển đến VQA
        
        # Kiểm tra xem có thông tin về tài liệu có liên quan không
        has_relevant_info = len(rag_result.get("sources", [])) > 0
        
        # Kiểm tra lịch sử hội thoại để xác định xem query hiện tại có phải là theo sau query trước đó không
        is_followup_query = False
        query = state.get("query", "")
        if conversation_history and len(conversation_history) > 1 and query:
            # Tính toán độ liên quan giữa query hiện tại và query trước đó
            prev_query = conversation_history[-2].get("query", "")
            prev_response = conversation_history[-2].get("response", "")
            
            if prev_query and (
                query.lower().startswith("tại sao") or
                query.lower().startswith("như thế nào") or
                query.lower().startswith("giải thích") or
                "?" in query or
                len(query.split()) < 10  # Câu ngắn thường là câu hỏi tiếp theo
            ):
                is_followup_query = True
                logger.info(f"Detected potential follow-up query based on conversation history")
        
        # Đưa ra quyết định routing
        if query_complexity == "complex" or vqa_output or (not has_relevant_info and not is_followup_query):
            logger.info("Routing to VQA due to complex medical question or lack of relevant document info")
            return "vqa"
        else:
            logger.info("Simple document query with relevant info found, routing to synthesizer")
            return "synthesizer"
    
    # Mặc định route đến synthesizer nếu không có thông tin RAG
    logger.info("No RAG result available, defaulting to synthesizer")
    return "synthesizer"