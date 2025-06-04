#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Configuration (MODIFIED for multi-task support)
--------------------------------
configuration với multi-task selection support.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal, Union, cast
from enum import Enum
from dataclasses import dataclass, field

# Task Types
class TaskType(str, Enum):
    """Các loại task trong hệ thống - với multi-task."""
    POLYP_DETECTION = "polyp_detection"
    MODALITY_CLASSIFICATION = "modality_classification"
    REGION_CLASSIFICATION = "region_classification"
    MEDICAL_QA = "medical_qa"
    COMPREHENSIVE = "comprehensive"
    TEXT_ONLY = "text_only"
    MULTI_TASK = "multi_task"  # NEW: For combination tasks

# Keep existing TypedDict definitions...
class DetectionResult(TypedDict, total=False):
    """Kết quả từ Detector Agent."""
    success: bool
    objects: List[Dict[str, Any]]
    count: int
    error: Optional[str]

class ClassificationResult(TypedDict, total=False):
    """Kết quả từ Classifier Agent."""
    success: bool
    class_name: str
    confidence: float
    error: Optional[str]

class VQAResult(TypedDict, total=False):
    """Kết quả từ VQA Agent."""
    success: bool
    answer: str
    error: Optional[str]

class RAGResult(TypedDict, total=False):
    """Kết quả từ RAG Agent."""
    success: bool
    answer: str
    sources: List[str]
    error: Optional[str]

# SystemState với multi-task support
class SystemState(TypedDict, total=False):
    """State của hệ thống LangGraph với multi-task support."""
    # Input
    image_path: str
    query: Optional[str]
    medical_context: Optional[Dict[str, Any]]
    is_text_only: bool
    
    # Task Management
    task_type: TaskType
    required_tasks: List[str]  # NEW: List of required tasks
    completed_tasks: List[str]  # NEW: List of completed tasks
    current_task: Optional[str]  # NEW: Currently executing task
    execution_order: List[str]  # NEW: Optimal execution order
    
    # Processing
    session_id: str
    start_time: float
    
    # Results (unchanged)
    detector_result: Optional[DetectionResult]
    modality_result: Optional[ClassificationResult]
    region_result: Optional[ClassificationResult]
    vqa_result: Optional[VQAResult]
    rag_result: Optional[RAGResult]
    
    # Output
    final_result: Optional[Dict[str, Any]]
    error: Optional[str]

@dataclass
class MedicalGraphConfig:
    """cấu hình cho Medical AI Graph với multi-task support."""
    name: str = "Multi-Task Medical AI Graph"
    device: str = "cuda"
    parallel_execution: bool = True

    # Multi-task configuration
    enable_multi_task: bool = True  # NEW
    max_concurrent_tasks: int = 3   # NEW
    task_priority_order: List[str] = field(default_factory=lambda: [  # NEW
        "polyp_detection",
        "modality_classification", 
        "region_classification",
        "medical_qa"
    ])
    
    # Paths to models (unchanged)
    detector_model_path: str = "medical_ai_agents/weights/detect_best.pt"
    modality_classifier_path: str = "medical_ai_agents/weights/modal_best.pt"
    region_classifier_path: str = "medical_ai_agents/weights/location_best.pt"
    vqa_model_path: str = "medical_ai_agents/weights/llava-med-mistral-v1.5-7b"
    
    # LLM config (unchanged)
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.5
    
    # LangGraph config (unchanged)
    checkpoint_dir: str = "checkpoints"
    consistency_threshold: float = 0.7
    output_path: str = "results"