#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Configuration
--------------------------------
Cấu hình và định nghĩa kiểu cho hệ thống AI y tế đa agent.
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated, Literal, Union, cast
from enum import Enum
from dataclasses import dataclass, field

# Type Definitions
class TaskType(str, Enum):
    """Các loại task trong hệ thống."""
    POLYP_DETECTION = "polyp_detection"
    MODALITY_CLASSIFICATION = "modality_classification"
    REGION_CLASSIFICATION = "region_classification"
    MEDICAL_QA = "medical_qa"
    COMPREHENSIVE = "comprehensive"

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
    confidence: float
    error: Optional[str]

class RAGResult(TypedDict, total=False):
    """Kết quả từ RAG Agent."""
    success: bool
    answer: str
    sources: List[str]
    error: Optional[str]

class ReflectionResult(TypedDict, total=False):
    """Kết quả từ Reflection."""
    original_answer: str
    improved_answer: str
    bias_detected: bool
    confidence: float

class SystemState(TypedDict, total=False):
    """State của hệ thống LangGraph."""
    # Input
    image_path: str
    query: Optional[str]
    medical_context: Optional[Dict[str, Any]]
    
    # Processing
    session_id: str
    task_type: TaskType
    start_time: float
    
    # Results
    detector_result: Optional[DetectionResult]
    modality_result: Optional[ClassificationResult]
    region_result: Optional[ClassificationResult]
    vqa_result: Optional[VQAResult]
    rag_result: Optional[RAGResult]
    reflection_result: Optional[ReflectionResult]
    
    # Output
    final_result: Optional[Dict[str, Any]]
    error: Optional[str]

@dataclass
class MedicalGraphConfig:
    """Cấu hình cho Medical AI Graph."""
    name: str = "Medical AI Graph"
    device: str = "cuda"
    parallel_execution: bool = True
    use_reflection: bool = True
    
    # Paths to models
    detector_model_path: str = "weights/detect_best.pt"
    modality_classifier_path: str = "weights/modal_best.pt"
    region_classifier_path: str = "weights/location_best.pt"
    vqa_model_path: str = "weights/llava-med-mistral-v1.5-7b"
    
    # LLM config
    llm_model: str = "gpt-4"
    llm_temperature: float = 0.2
    
    # LangGraph config
    checkpoint_dir: str = "checkpoints"
    
    # Other configs
    consistency_threshold: float = 0.7
    output_path: str = "results"