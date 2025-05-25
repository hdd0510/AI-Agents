#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Graph - Pipeline
-------------------------
Tạo và thiết lập LangGraph cho hệ thống AI y tế.
"""

import os
import logging

from langgraph.graph import StateGraph, END
# Bỏ import này vì nó không tồn tại trong phiên bản 0.4.5
# from langgraph.checkpoint import checkpoint_graph
from langchain_openai import ChatOpenAI

# Thêm import cho checkpointer
from langgraph.checkpoint.memory import MemorySaver

from medical_ai_agents.config import MedicalGraphConfig, SystemState
from medical_ai_agents.agents.detector import DetectorAgent
from medical_ai_agents.agents.classifier import ClassifierAgent
from medical_ai_agents.agents.vqa import VQAAgent
from medical_ai_agents.graph.nodes import task_analyzer, reflection_node, result_synthesizer
from medical_ai_agents.graph.routers import (
    task_router, post_detector_router, post_modality_router, 
    post_region_router, post_vqa_router
)

def create_medical_ai_graph(config: MedicalGraphConfig):
    """Creates and returns the Medical AI LangGraph."""
    logger = logging.getLogger("graph.pipeline")
    
    # Initialize agents
    logger.info("Initializing agents...")
    
    # Detector agent
    detector_agent = DetectorAgent(
        model_path=config.detector_model_path,
        device=config.device
    )
    
    # Modality classifier agent
    modality_classifier_agent = ClassifierAgent(
        model_path=config.modality_classifier_path,
        class_names=["WLI", "BLI", "FICE", "LCI"],
        classifier_type="modality",
        device=config.device
    )
    
    # Region classifier agent
    region_classifier_agent = ClassifierAgent(
        model_path=config.region_classifier_path,
        class_names=[
            "Hau_hong", "Thuc_quan", "Tam_vi", "Than_vi", 
            "Phinh_vi", "Hang_vi", "Bo_cong_lon", "Bo_cong_nho", 
            "Hanh_ta_trang", "Ta_trang"
        ],
        classifier_type="region",
        device=config.device
    )
    
    # VQA agent
    vqa_agent = VQAAgent(
        model_path=config.vqa_model_path,
        device=config.device
    )
    
    # Initialize LLM for task analyzer and reflection
    llm = ChatOpenAI(model=config.llm_model, temperature=config.llm_temperature)
    
    # Create workflow graph
    logger.info("Creating StateGraph...")
    workflow = StateGraph(SystemState)
    
    # Add nodes
    workflow.add_node("task_analyzer", lambda state: task_analyzer(state, llm))
    workflow.add_node("detector", detector_agent)
    workflow.add_node("modality_classifier", modality_classifier_agent)
    workflow.add_node("region_classifier", region_classifier_agent)
    workflow.add_node("vqa", vqa_agent)
    if config.use_reflection:
        workflow.add_node("reflection", lambda state: reflection_node(state, llm))
    workflow.add_node("synthesizer", lambda state: result_synthesizer(state, llm))
    
    # Add edges
    workflow.set_entry_point("task_analyzer")
    workflow.add_conditional_edges(
        "task_analyzer",
        task_router,
        {
            "detector": "detector",
            "modality_classifier": "modality_classifier",
            "region_classifier": "region_classifier",
            "vqa": "vqa",
            "synthesizer": "synthesizer"
        }
    )
    workflow.add_conditional_edges(
        "detector",
        post_detector_router,
        {
            "modality_classifier": "modality_classifier",
            "vqa": "vqa",
            "synthesizer": "synthesizer"
        }
    )
    workflow.add_conditional_edges(
        "modality_classifier",
        post_modality_router,
        {
            "region_classifier": "region_classifier",
            "synthesizer": "synthesizer"
        }
    )
    workflow.add_conditional_edges(
        "region_classifier",
        post_region_router,
        {
            "vqa": "vqa",
            "synthesizer": "synthesizer"
        }
    )
    workflow.add_conditional_edges(
        "vqa",
        post_vqa_router,
        {
            "reflection": "reflection",
            "synthesizer": "synthesizer"
        }
    )
    if config.use_reflection:
        workflow.add_edge("reflection", "synthesizer")
    workflow.add_edge("synthesizer", END)
    
    # Thêm checkpointing và compile graph - sửa phương pháp ở đây
    if config.checkpoint_dir and os.path.exists(config.checkpoint_dir):
        logger.info(f"Adding checkpointing to {config.checkpoint_dir}")
        # Tạo checkpointer bằng MemorySaver cho lưu trong RAM
        checkpointer = MemorySaver()
        
        # Hoặc sử dụng SqliteSaver cho lưu vào file
        # import sqlite3
        # conn = sqlite3.connect(os.path.join(config.checkpoint_dir, "checkpoints.db"), check_same_thread=False)
        # checkpointer = SqliteSaver(conn)
        
        # Compile graph với checkpointer
        return workflow.compile(checkpointer=checkpointer)
    else:
        # Compile bình thường không có checkpoint
        return workflow.compile()