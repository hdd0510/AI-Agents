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
    task_analyzer, result_synthesizer, task_completion_wrapper, document_embedding
)
from medical_ai_agents.graph.routers import (
    task_router, post_detector_router, 
    post_modality_router, post_region_router, 
    post_vqa_router, post_rag_router, post_vqa_router_with_rag
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
    
    # Add nodes - MODIFIED with task completion wrappers
    workflow.add_node("document_embedding", lambda state: document_embedding(state, rag_agent))
    workflow.add_node("task_analyzer", lambda state: task_analyzer(state, llm))
    # Add with task completion wrapper for proper state management
    workflow.add_node("detector", task_completion_wrapper(detector_agent, "polyp_detection"))
    workflow.add_node("rag", task_completion_wrapper(rag_agent, "document_qa"))
    workflow.add_node("modality_classifier", task_completion_wrapper(modality_classifier_agent, "modality_classification"))
    workflow.add_node("region_classifier", task_completion_wrapper(region_classifier_agent, "region_classification"))
    workflow.add_node("vqa", task_completion_wrapper(vqa_agent, "medical_qa"))
    workflow.add_node("synthesizer", lambda state: result_synthesizer(state, llm))
    
    # Add edges with multi-task routing
    # MODIFIED: Luôn bắt đầu với document_embedding để kiểm tra và xử lý tài liệu trước
    workflow.set_entry_point("document_embedding")
    
    # Add edge from document_embedding to task_analyzer
    workflow.add_edge("document_embedding", "task_analyzer")
    
    # conditional edges với logging cải tiến
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

    # MODIFICATION: Cập nhật routing để đảm bảo RAG luôn được ưu tiên khi có tài liệu
    workflow.add_conditional_edges(
        "detector",
        post_detector_router,
        {
            "modality_classifier": "modality_classifier", 
            "region_classifier": "region_classifier",
            "vqa": "vqa",
            "rag": "rag",  # Added RAG as possible next step
            "synthesizer": "synthesizer"
        }
    )
    
    workflow.add_conditional_edges(
        "modality_classifier",
        post_modality_router,
        {
            "region_classifier": "region_classifier",
            "vqa": "vqa",
            "rag": "rag",  # Added RAG as possible next step
            "synthesizer": "synthesizer",
            "detector": "detector"  # Thêm để xử lý trường hợp đặc biệt khi polyp_detection chưa hoàn thành
        }
    )
    
    workflow.add_conditional_edges(
        "region_classifier",
        post_region_router,
        {
            "vqa": "vqa",
            "rag": "rag",  # Added RAG as possible next step
            "synthesizer": "synthesizer",
            "detector": "detector",  # Thêm để xử lý trường hợp đặc biệt khi polyp_detection chưa hoàn thành
            "modality_classifier": "modality_classifier"  # Đường dẫn dự phòng khác
        }
    )
    
    # Sử dụng post_vqa_router đã cập nhật để kiểm tra đúng tài liệu có sẵn
    workflow.add_conditional_edges(
        "vqa",
        post_vqa_router,
        {
            "rag": "rag",  # Luôn xem xét RAG như một bước tiếp theo có thể
            "synthesizer": "synthesizer"
        }
    )

    # MODIFIED: Cập nhật logic routing sau RAG để xem xét kết quả embedding và phức tạp của truy vấn
    workflow.add_conditional_edges(
        "rag",
        post_rag_router,
        {
            "vqa": "vqa",  # Có thể chuyển đến VQA nếu là câu hỏi y tế phức tạp
            "synthesizer": "synthesizer"  # Trực tiếp đến synthesizer nếu là truy vấn tài liệu đơn giản
        }
    )
    
    workflow.add_edge("synthesizer", END)
    
    # Compile với checkpointing
    return workflow.compile()
