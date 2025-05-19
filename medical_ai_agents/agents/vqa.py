#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - VQA Agent
---------------------------
Agent trả lời câu hỏi về hình ảnh với LLM controller và LLaVA tool.
"""

import json
from typing import Dict, Any, List
import logging

from medical_ai_agents.agents.base_agent import BaseAgent
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.vqa.llava_tools import LLaVATool

class VQAAgent(BaseAgent):
    """Agent trả lời câu hỏi về hình ảnh y tế sử dụng LLM controller."""
    
    def __init__(self, model_path: str, llm_model: str = "gpt-4", device: str = "cuda"):
        """
        Khởi tạo VQA Agent với LLM controller.
        
        Args:
            model_path: Đường dẫn đến LLaVA model
            llm_model: Mô hình LLM sử dụng làm controller
            device: Device để chạy model (cuda/cpu)
        """
        super().__init__(name="VQA Agent", llm_model=llm_model, device=device)
        self.model_path = model_path
        self.llava_tool = None
    
    def _register_tools(self) -> List[BaseTool]:
        """Register tools for this agent."""
        self.llava_tool = LLaVATool(model_path=self.model_path, device=self.device)
        return [self.llava_tool]
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines this agent's role."""
        return """Bạn là một AI chuyên gia y tế chuyên trả lời câu hỏi dựa trên hình ảnh nội soi tiêu hóa.
Nhiệm vụ của bạn là phân tích câu hỏi y tế và sử dụng công cụ thị giác để trả lời chính xác.

Bạn có thể sử dụng công cụ sau:
1. llava_vqa: Công cụ trả lời câu hỏi dựa trên hình ảnh sử dụng mô hình LLaVA
   - Tham số: image_path (str), question (str), medical_context (Dict, optional)
   - Kết quả: câu trả lời và độ tin cậy

Quy trình làm việc của bạn:
1. Phân tích câu hỏi của người dùng
2. Chuẩn bị câu hỏi chi tiết cho mô hình LLaVA
3. Sử dụng công cụ llava_vqa để trả lời câu hỏi
4. Phân tích câu trả lời và độ tin cậy
5. Nâng cao chất lượng câu trả lời với kiến thức y tế chuyên môn

Khi trả lời:
- Đảm bảo câu trả lời có tính chuyên môn y tế cao
- Chỉ ra những điểm không chắc chắn nếu có
- Sử dụng ngôn ngữ phù hợp với chuyên gia y tế

Bạn phải trả về JSON với định dạng:
```json
{
  "vqa_result": {
    "success": true/false,
    "answer": "câu trả lời chi tiết",
    "confidence": confidence_value,
    "analysis": "phân tích chuyên môn về độ tin cậy và chất lượng câu trả lời"
  }
}
```"""

    def initialize(self) -> bool:
        """Khởi tạo agent và các công cụ."""
        try:
            # Tools are already initialized in _register_tools
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize VQA agent: {str(e)}")
            self.initialized = False
            return False
    
    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract task-specific input from state."""
        # Get detector results if available
        detector_result = state.get("detector_result", {})
        modality_result = state.get("modality_result", {})
        region_result = state.get("region_result", {})
        
        medical_context = {}
        
        # Add detection info to context
        if detector_result and detector_result.get("success", False):
            objects = detector_result.get("objects", [])
            medical_context["detected_polyps"] = len(objects)
            
            if objects:
                polyp_descriptions = []
                for i, obj in enumerate(objects[:3]):  # Top 3 objects
                    desc = f"Polyp {i+1}: {obj.get('confidence', 0):.2f} confidence, "
                    desc += f"location: {obj.get('position_description', 'unknown')}"
                    polyp_descriptions.append(desc)
                
                medical_context["polyp_details"] = "; ".join(polyp_descriptions)
        
        # Add modality info
        if modality_result and modality_result.get("success", False):
            medical_context["imaging_modality"] = modality_result.get("class_name", "Unknown")
        
        # Add region info
        if region_result and region_result.get("success", False):
            medical_context["anatomical_region"] = region_result.get("class_name", "Unknown")
        
        # Add user-provided context
        user_context = state.get("medical_context", {})
        if user_context:
            medical_context.update(user_context)
        
        return {
            "image_path": state.get("image_path", ""),
            "query": state.get("query", ""),
            "medical_context": medical_context
        }
    
    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for LLM prompt."""
        image_path = task_input.get("image_path", "")
        query = task_input.get("query", "")
        context = task_input.get("medical_context", {})
        
        context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()]) if context else "None"
        
        return f"""Hình ảnh cần phân tích: {image_path}
        
Câu hỏi: {query if query else "Mô tả những gì bạn thấy trong hình ảnh này"}

Thông tin y tế bổ sung:
{context_str}

Hãy sử dụng công cụ llava_vqa để trả lời câu hỏi này dựa trên hình ảnh.
Cần đảm bảo câu trả lời có tính chuyên môn cao và chính xác về mặt y tế.

Trả lời theo định dạng:

Tool: [tên công cụ]
Parameters: [tham số dưới dạng JSON]

Sau khi sử dụng công cụ, hãy phân tích kết quả và đưa ra câu trả lời cuối cùng với độ tin cậy.
"""
    
    def _extract_agent_result(self, synthesis: str) -> Dict[str, Any]:
        """Extract agent result from LLM synthesis."""
        try:
            # Try to extract JSON
            json_start = synthesis.find('{')
            json_end = synthesis.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = synthesis[json_start:json_end]
                vqa_result = json.loads(json_str)
                return vqa_result
            
            # Fallback: Create result from synthesis text
            return {
                "vqa_result": {
                    "success": True,
                    "answer": synthesis,
                    "confidence": 0.7,
                    "analysis": "Generated from LLM synthesis"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract agent result: {str(e)}")
            return {
                "vqa_result": {
                    "success": False,
                    "error": str(e),
                    "answer": synthesis,
                    "confidence": 0.5
                }
            }