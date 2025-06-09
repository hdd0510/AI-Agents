"""
Medical AI System - Main Entry Point (MODIFIED for multi-task support)
-----------------------------------
điểm vào chính với multi-task execution support.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

from medical_ai_agents.config import MedicalGraphConfig
from medical_ai_agents.graph.pipeline import create_medical_ai_graph

class EnhancedMedicalAISystem:
    """
    Medical AI System với multi-task execution support.
    """
    
    def __init__(self, config: MedicalGraphConfig = None):
        """Initialize the Medical AI System."""
        self.config = config or MedicalGraphConfig()
        self.graph = create_medical_ai_graph(self.config)
        self.logger = logging.getLogger("enhanced-medical-ai-system")
        self.logger.info(f"Initialized Medical AI System: {self.config.name}")
    
    def analyze(self, 
               image_path: Optional[str] = None,
               query: Optional[str] = None, 
               medical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        analyze với multi-task execution support.
        
        Args:
            image_path: Optional path to the image file
            query: Optional question or request  
            medical_context: Optional medical context information
            
        Returns:
            Dict with analysis results including multi-task info
        """
        # Validate inputs
        if not image_path and not query:
            return {
                "error": "Either image_path or query must be provided", 
                "success": False
            }
        
        # Only validate image if provided
        if image_path and not os.path.exists(image_path):
            return {
                "error": f"Image not found: {image_path}", 
                "success": False
            }
        
        # Create initial state
        from medical_ai_agents.config import SystemState
        import uuid
        import time
        
        initial_state: SystemState = {
            "image_path": image_path or "",
            "query": query,
            "medical_context": medical_context,
            "session_id": str(uuid.uuid4()),
            "start_time": time.time(),
            "is_text_only": image_path is None,
            
            # Multi-task initialization
            "required_tasks": [],
            "completed_tasks": [],
            "execution_order": [],
            "current_task": None
        }
        
        try:
            # Run the graph
            final_state = None
            for event in self.graph.stream(initial_state):
                self.logger.debug(f"step completed: {list(event.keys())}")
                for node_name, state in event.items():
                    final_state = state
            
            # Check final state
            if final_state is None:
                return {
                    "error": "graph execution failed - no final state",
                    "success": False,
                    "session_id": initial_state["session_id"]
                }
            
            # Create output directory if needed
            if self.config.output_path and image_path:
                session_dir = os.path.join(self.config.output_path, final_state.get("session_id", "unknown"))
                os.makedirs(session_dir, exist_ok=True)
                
                # Save result
                if "final_result" in final_state:
                    result_path = os.path.join(session_dir, "result.json")
                    with open(result_path, 'w', encoding='utf-8') as f:
                        json.dump(final_state["final_result"], f, ensure_ascii=False, indent=2)
            
            # Return final result
            if "final_result" in final_state and final_state["final_result"]:
                final_result = final_state["final_result"]
                
                # Thêm final_answer để app.py có thể sử dụng
                if "response" in final_result and final_result["response"]:
                    final_result["final_answer"] = final_result["response"]
                elif "agent_results" in final_result:
                    # Fallback từ các kết quả agent nếu không có response tổng hợp
                    agent_results = final_result["agent_results"]
                    if "region_result" in agent_results and agent_results["region_result"].get("success", False):
                        region = agent_results["region_result"]
                        final_result["final_answer"] = f"Vùng giải phẫu: {region.get('class_name', 'Unknown')} ({region.get('confidence', 0.0):.1%})\n\n{region.get('analysis', '')}"
                    elif "vqa_result" in agent_results and agent_results["vqa_result"].get("success", False):
                        final_result["final_answer"] = agent_results["vqa_result"].get("answer", "")
                
                return final_result
            else:
                # fallback result
                fallback_result = {
                    "success": True,
                    "session_id": final_state.get("session_id", ""),
                    "task_type": final_state.get("task_type", "unknown"),
                    "query": final_state.get("query", ""),
                    "is_text_only": final_state.get("is_text_only", False),
                    "processing_time": time.time() - initial_state["start_time"],
                    
                    # Multi-task fallback info
                    "multi_task_analysis": {
                        "required_tasks": final_state.get("required_tasks", []),
                        "completed_tasks": final_state.get("completed_tasks", []),
                        "execution_order": final_state.get("execution_order", [])
                    }
                }
                
                # Add agent results if available
                agent_results = {}
                if "detector_result" in final_state:
                    agent_results["detector_result"] = final_state["detector_result"]
                if "modality_result" in final_state:
                    agent_results["modality_result"] = final_state["modality_result"]
                if "region_result" in final_state:
                    agent_results["region_result"] = final_state["region_result"]
                if "vqa_result" in final_state:
                    print('-'*100)
                    print(final_state["vqa_result"])
                    print('-'*100)
                    agent_results["vqa_result"] = final_state["vqa_result"]
                
                fallback_result["agent_results"] = agent_results
                
                # Add legacy fields for compatibility
                if "vqa_result" in agent_results:
                    vqa_result = agent_results["vqa_result"]
                    if vqa_result and vqa_result.get("success", False):
                        fallback_result["answer"] = vqa_result.get("answer", "")
                        fallback_result["answer_confidence"] = vqa_result.get("confidence", 0.0)
                
                if "detector_result" in agent_results:
                    detector_result = agent_results["detector_result"]
                    if detector_result and detector_result.get("success", False):
                        fallback_result["polyps"] = detector_result.get("objects", [])
                        fallback_result["polyp_count"] = len(detector_result.get("objects", []))
                    else:
                        fallback_result["polyps"] = []
                        fallback_result["polyp_count"] = 0
                
                # Check for errors
                if "error" in final_state:
                    fallback_result["error"] = final_state["error"]
                    fallback_result["success"] = False
                
                return fallback_result
            
        except Exception as e:
            import traceback
            error_msg = f"analysis failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return {
                "error": error_msg,
                "success": False,
                "session_id": initial_state["session_id"]
            }