#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Main Entry Point (MODIFIED for text-only support)
-----------------------------------
Điểm vào chính của hệ thống AI y tế đa agent sử dụng LangGraph.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional

from medical_ai_agents.config import MedicalGraphConfig
from medical_ai_agents.graph.pipeline import create_medical_ai_graph

class MedicalAISystem:
    """
    Medical AI System using LangGraph for orchestration.
    Now supports both image+text and text-only modes.
    """
    
    def __init__(self, config: MedicalGraphConfig = None):
        """Initialize the Medical AI System."""
        self.config = config or MedicalGraphConfig()
        self.graph = create_medical_ai_graph(self.config)
        self.logger = logging.getLogger("medical-ai-system")
        self.logger.info(f"Initialized Medical AI System with config: {self.config.name}")
    
    def analyze(self, 
               image_path: Optional[str] = None,  # CHANGED: Now optional 
               query: Optional[str] = None, 
               medical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a medical image and/or answer a query.
        
        Args:
            image_path: Optional path to the image file
            query: Optional question or request  
            medical_context: Optional medical context information
            
        Returns:
            Dict with analysis results
        """
        # MODIFIED: Validate inputs - at least one must be provided
        if not image_path and not query:
            return {
                "error": "Either image_path or query must be provided", 
                "success": False
            }
        
        # MODIFIED: Only validate image if provided
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
            "image_path": image_path or "",  # CHANGED: Empty string if None
            "query": query,
            "medical_context": medical_context,
            "session_id": str(uuid.uuid4()),
            "start_time": time.time(),
            "is_text_only": image_path is None  # NEW: Flag for text-only mode
        }
        
        try:
            # Run the graph và lấy state cuối cùng
            final_state = None
            for event in self.graph.stream(initial_state):
                self.logger.debug(f"Step completed: {list(event.keys())}")
                for node_name, state in event.items():
                    final_state = state
            
            # Kiểm tra final_state
            if final_state is None:
                return {
                    "error": "Graph execution failed - no final state",
                    "success": False,
                    "session_id": initial_state["session_id"]
                }
            
            # Create output directory if needed (only for image analysis)
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
                return final_state["final_result"]
            else:
                # Fallback: tạo result từ các thành phần có sẵn
                fallback_result = {
                    "success": True,
                    "session_id": final_state.get("session_id", ""),
                    "task_type": final_state.get("task_type", "unknown"),
                    "query": final_state.get("query", ""),
                    "is_text_only": final_state.get("is_text_only", False),
                    "processing_time": time.time() - initial_state["start_time"]
                }
                
                # Add VQA result for text-only queries
                if "vqa_result" in final_state:
                    vqa_result = final_state["vqa_result"]
                    if vqa_result and vqa_result.get("success", False):
                        fallback_result["answer"] = vqa_result.get("answer", "")
                        fallback_result["answer_confidence"] = vqa_result.get("confidence", 0.0)
                
                # Add detector result if image was provided
                if image_path and "detector_result" in final_state:
                    detector_result = final_state["detector_result"]
                    if detector_result and detector_result.get("success", False):
                        fallback_result["polyps"] = detector_result.get("objects", [])
                        fallback_result["polyp_count"] = len(detector_result.get("objects", []))
                    else:
                        fallback_result["polyps"] = []
                        fallback_result["polyp_count"] = 0
                
                # Check for any errors
                if "error" in final_state:
                    fallback_result["error"] = final_state["error"]
                    fallback_result["success"] = False
                
                return fallback_result
            
        except Exception as e:
            import traceback
            error_msg = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return {
                "error": error_msg,
                "success": False,
                "session_id": initial_state["session_id"]
            }