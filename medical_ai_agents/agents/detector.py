#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - Detector Agent
--------------------------------
Agent phát hiện polyp với LLM controller và YOLO detection tool.
"""

import json
from typing import Dict, Any, List
import logging

from medical_ai_agents.agents.base_agent import BaseAgent
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.detection.yolo_tools import YOLODetectionTool

class DetectorAgent(BaseAgent):
    """Agent phát hiện polyp trong hình ảnh nội soi sử dụng LLM controller."""
    
    def __init__(self, model_path: str, llm_model: str = "gpt-4o-mini", device: str = "cuda"):
        """
        Khởi tạo Detector Agent với LLM controller.
        
        Args:
            model_path: Đường dẫn đến YOLO model weights
            llm_model: Mô hình LLM sử dụng làm controller
            device: Device để chạy model (cuda/cpu)
        """
        self.model_path = model_path
        super().__init__(name="Detector Agent", llm_model=llm_model, device=device)
        self.yolo_tool = None
    
    def _register_tools(self) -> List[BaseTool]:
        """Register tools for this agent."""
        self.yolo_tool = YOLODetectionTool(model_path=self.model_path, device=self.device)
        return [self.yolo_tool]
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines this agent's role."""
        return """Bạn là một AI chuyên gia về phát hiện polyp trong hình ảnh nội soi tiêu hóa. 
Nhiệm vụ của bạn là phân tích hình ảnh để xác định vị trí, kích thước và đặc điểm của các polyp.

Bạn có thể sử dụng các công cụ sau theo thứ tự:
1. yolo_detection: Công cụ phát hiện polyp sử dụng mô hình YOLO
   - Tham số: image_path (str), conf_thresh (float, optional)

Quy trình làm việc của bạn PHẢI theo thứ tự sau:
1. Xác định hình ảnh cần phân tích
2. Sử dụng công cụ yolo_detection để phát hiện polyp
"""

    def initialize(self) -> bool:
        """Khởi tạo agent và các công cụ."""
        try:
            # Tools are already initialized in _register_tools
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize detector agent: {str(e)}")
            self.initialized = False
            return False
    
    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract task-specific input from state."""
        return {
            "image_path": state.get("image_path", ""),
            "query": state.get("query", ""),
            "medical_context": state.get("medical_context", {})
        }
    
    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for LLM prompt."""
        image_path = task_input.get("image_path", "")
        query = task_input.get("query", "")
        context = task_input.get("medical_context", {})
        
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()]) if context else "None"
        
        return f"""Hình ảnh cần phân tích: {image_path}
        
        Yêu cầu: {query if query else "Phát hiện polyp trong hình ảnh"}

        Thông tin y tế bổ sung:
        {context_str}

        Hãy phân tích hình ảnh này để tìm polyp. Sử dụng các công cụ có sẵn để phát hiện và phân tích.
        Trả lời theo định dạng để tool có thể sử dụng:

        Tool: yolo_detection (tên công cụ)
        Parameters: ({{"image_path": "path/to/image.jpg", "conf_thresh": 0.5}}) (tham số dưới dạng JSON)

        Sau khi sử dụng công cụ, hãy phân tích kết quả và đưa ra nhận xét chuyên môn.
        """
    def _format_synthesis_input(self) -> str:
        return """
        Dựa trên kết quả từ tools, bạn phải xác định:
        - Mô tả chi tiết các polyp được phát hiện (vị trí, kích thước, đặc điểm)
        - Đưa ra đánh giá về mức độ tin cậy của phát hiện
        - Nếu không phát hiện polyp, hãy xác nhận điều đó và giải thích lý do có thể

        Bạn phải trả về JSON với định dạng:
        ```json
        {
        "detector_result": {
            "success": true/false,
            "count": number_of_polyps,
            "objects": [...list of objects...],
            "analysis": "nhận xét chuyên môn về kết quả phát hiện"
        }
        }
        ```
        """
    
    def _extract_agent_result(self, synthesis: str) -> Dict[str, Any]:
        """Extract agent result from LLM synthesis."""
        try:
            # Enhanced JSON extraction with multiple attempts
            json_str = None
            
            # Method 1: Find first complete JSON object
            json_start = synthesis.find('{')
            if json_start >= 0:
                brace_count = 0
                json_end = json_start
                
                for i, char in enumerate(synthesis[json_start:], json_start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if brace_count == 0:  # Found complete JSON
                    json_str = synthesis[json_start:json_end]
            
            # Method 2: Extract between ```json and ``` if exists
            if not json_str:
                import re
                json_block = re.search(r'```json\s*(\{.*?\})\s*```', synthesis, re.DOTALL)
                if json_block:
                    json_str = json_block.group(1)
            
            # Parse the JSON
            if json_str:
                detector_result = json.loads(json_str)
                self.logger.info(f"Successfully extracted JSON result")
                return detector_result
            
            # Method 3: Fallback - create result from synthesis text
            self.logger.warning("No JSON found, creating fallback result")
            return {
                "detector_result": {
                    "success": True,
                    "analysis": synthesis,
                    "count": 0,
                    "objects": []
                }
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {str(e)}")
            self.logger.error(f"Attempted to parse: {json_str}")
            
            # Fallback result
            return {
                "detector_result": {
                    "success": False,
                    "error": f"JSON parsing failed: {str(e)}",
                    "analysis": synthesis
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to extract agent result: {str(e)}")
            return {
                "detector_result": {
                    "success": False,
                    "error": str(e),
                    "analysis": synthesis
                }
            }