#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Main Entry Point (FIXED)
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
    
    Usage:
    ```python
    system = MedicalAISystem()
    result = system.analyze(
        image_path="path/to/image.jpg",
        query="Is there a polyp in this image?",
        medical_context={"patient_history": "Family history of colon cancer"}
    )
    ```
    """
    
    def __init__(self, config: MedicalGraphConfig = None):
        """Initialize the Medical AI System."""
        self.config = config or MedicalGraphConfig()
        self.graph = create_medical_ai_graph(self.config)
        self.logger = logging.getLogger("medical-ai-system")
        self.logger.info(f"Initialized Medical AI System with config: {self.config.name}")
    
    def analyze(self, 
               image_path: str, 
               query: Optional[str] = None, 
               medical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze a medical image and optionally answer a query.
        
        Args:
            image_path: Path to the image file
            query: Optional question or request
            medical_context: Optional medical context information
            
        Returns:
            Dict with analysis results
        """
        # Validate image path
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}", "success": False}
        
        # Create initial state
        from medical_ai_agents.config import SystemState
        import uuid
        import time
        
        initial_state: SystemState = {
            "image_path": image_path,
            "query": query,
            "medical_context": medical_context,
            "session_id": str(uuid.uuid4()),
            "start_time": time.time()
        }
        
        try:
            # Run the graph và lấy state cuối cùng
            final_state = None
            for event in self.graph.stream(initial_state):
                # event là một dict với key là tên node và value là state sau khi node đó chạy
                self.logger.debug(f"Step completed: {list(event.keys())}")
                # Lưu state cuối cùng
                for node_name, state in event.items():
                    final_state = state
            
            # Kiểm tra final_state
            if final_state is None:
                return {
                    "error": "Graph execution failed - no final state",
                    "success": False,
                    "session_id": initial_state["session_id"]
                }
            
            # Create output directory if needed
            if self.config.output_path:
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
                    "processing_time": time.time() - initial_state["start_time"]
                }
                
                # Thêm detector result nếu có
                if "detector_result" in final_state:
                    detector_result = final_state["detector_result"]
                    if detector_result and detector_result.get("success", False):
                        fallback_result["polyps"] = detector_result.get("objects", [])
                        fallback_result["polyp_count"] = len(detector_result.get("objects", []))
                    else:
                        fallback_result["polyps"] = []
                        fallback_result["polyp_count"] = 0
                
                # Thêm classification results nếu có
                if "modality_result" in final_state:
                    modality_result = final_state["modality_result"]
                    if modality_result and modality_result.get("success", False):
                        fallback_result["modality"] = {
                            "class_name": modality_result.get("class_name", "unknown"),
                            "confidence": modality_result.get("confidence", 0.0)
                        }
                
                if "region_result" in final_state:
                    region_result = final_state["region_result"]
                    if region_result and region_result.get("success", False):
                        fallback_result["region"] = {
                            "class_name": region_result.get("class_name", "unknown"),
                            "confidence": region_result.get("confidence", 0.0)
                        }
                
                # Thêm VQA result nếu có
                if "vqa_result" in final_state:
                    vqa_result = final_state["vqa_result"]
                    if vqa_result and vqa_result.get("success", False):
                        fallback_result["answer"] = vqa_result.get("answer", "")
                        fallback_result["answer_confidence"] = vqa_result.get("confidence", 0.0)
                
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

# Simple example to demonstrate usage
def example():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical AI System Example")
    parser.add_argument("--image", required=True, help="Path to medical image")
    parser.add_argument("--query", help="Medical question")
    
    args = parser.parse_args()
    
    # Initialize system
    system = MedicalAISystem()
    
    # Analyze image
    result = system.analyze(args.image, args.query)
    
    # Print result
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    example()