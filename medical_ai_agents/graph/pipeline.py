"""
Medical AI Graph - Pipeline (MODIFIED for multi-task support)
-------------------------
LangGraph pipeline với multi-task execution support.
"""

import os
import logging

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from medical_ai_agents.config import MedicalGraphConfig, SystemState
from medical_ai_agents.agents.detector import DetectorAgent
from medical_ai_agents.agents.classifier import ClassifierAgent
from medical_ai_agents.agents.vqa import VQAAgent
from medical_ai_agents.agents.rag import RAGAgent
from medical_ai_agents.graph.nodes import (
    task_analyzer, result_synthesizer
)
from medical_ai_agents.graph.routers import (
    task_router, post_detector_router, 
    post_modality_router, post_region_router, 
    post_vqa_router, post_rag_router
)

def create_medical_ai_graph(config: MedicalGraphConfig):
    """Create Medical AI LangGraph với multi-task support."""
    logger = logging.getLogger("graph.pipeline")
    
    # Initialize agents (unchanged)
    logger.info("Initializing agents...")
    
    rag_agent = RAGAgent(
        storage_path=config.rag_storage_path,  # Add to config
        device=config.device
    )

    detector_agent = DetectorAgent(
        model_path=config.detector_model_path,
        device=config.device
    )
    
    modality_classifier_agent = ClassifierAgent(
        model_path=config.modality_classifier_path,
        class_names=["WLI", "BLI", "FICE", "LCI"],
        classifier_type="modality",
        device=config.device
    )
    
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
    
    vqa_agent = VQAAgent(
        model_path=config.vqa_model_path,
        device=config.device
    )
    
    # Initialize LLM for nodes
    llm = ChatOpenAI(model=config.llm_model, temperature=config.llm_temperature)
    
    # Create workflow graph
    logger.info("Creating StateGraph...")
    workflow = StateGraph(SystemState)
    
    # Add nodes
    workflow.add_node("task_analyzer", lambda state: task_analyzer(state, llm))
    workflow.add_node("detector", detector_agent)
    workflow.add_node("rag", rag_agent)
    workflow.add_node("modality_classifier", modality_classifier_agent)
    workflow.add_node("region_classifier", region_classifier_agent)
    workflow.add_node("vqa", vqa_agent)
    workflow.add_node("synthesizer", lambda state: result_synthesizer(state, llm))
    
    # Add edges with multi-task routing
    workflow.set_entry_point("task_analyzer")
    
    # conditional edges
    workflow.add_conditional_edges(
        "task_analyzer",
        task_router,
        {
            "detector": "detector",
            "modality_classifier": "modality_classifier",
            "region_classifier": "region_classifier",
            "vqa": "vqa",
            "rag": "rag",
            "synthesizer": "synthesizer"
        }
    )

    workflow.add_conditional_edges(
        "rag",
        post_rag_router,
        {
            "vqa": "vqa"
        }
    )
    
    workflow.add_conditional_edges(
        "vqa",
        post_vqa_router,
        {
            "synthesizer": "synthesizer"
        }
    )
    
    workflow.add_conditional_edges(
        "detector",
        post_detector_router,
        {
            "modality_classifier": "modality_classifier",
            "region_classifier": "region_classifier",
            "vqa": "vqa",
            "synthesizer": "synthesizer"
        }
    )
    
    workflow.add_conditional_edges(
        "modality_classifier",
        post_modality_router,
        {
            "region_classifier": "region_classifier",
            "vqa": "vqa",
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
    
    workflow.add_edge("synthesizer", END)
    
    # Compile with checkpointing
    if config.checkpoint_dir and os.path.exists(config.checkpoint_dir):
        logger.info(f"Adding checkpointing to {config.checkpoint_dir}")
        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    else:
        return workflow.compile()
