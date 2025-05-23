#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - Classifier Agent
----------------------------------
Agent phân loại hình ảnh với LLM controller và classifier tool.
"""

import json
from typing import Dict, Any, List
import logging

from medical_ai_agents.agents.base_agent import BaseAgent
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.classifier.cls_tools import ClassifierTool

class ClassifierAgent(BaseAgent):
    """Agent phân loại hình ảnh nội soi sử dụng LLM controller."""
    
    def __init__(self, model_path: str, class_names: List[str], 
                classifier_type: str = "modality", llm_model: str = "gpt-4", device: str = "cuda"):
        """
        Khởi tạo Classifier Agent với LLM controller.
        
        Args:
            model_path: Đường dẫn đến model weights
            class_names: Danh sách tên các lớp
            classifier_type: Loại classifier ('modality' hoặc 'region')
            llm_model: Mô hình LLM sử dụng làm controller
            device: Device để chạy model (cuda/cpu)
        """
        # Set attributes before super().__init__()
        self.model_path = model_path
        self.class_names = class_names
        self.classifier_type = classifier_type
        
        super().__init__(name=f"{classifier_type.capitalize()} Classifier Agent", llm_model=llm_model, device=device)
        self.classifier_tool = None
    
    def _register_tools(self) -> List[BaseTool]:
        """Register tools for this agent."""
        self.classifier_tool = ClassifierTool(
            model_path=self.model_path,
            class_names=self.class_names,
            classifier_type=self.classifier_type,
            device=self.device
        )
        return [self.classifier_tool]
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines this agent's role."""
        if self.classifier_type == "modality":
            type_desc = "kỹ thuật chụp nội soi (WLI, BLI, FICE, LCI)"
            outcome_key = "modality_result"
        else:
            type_desc = "vị trí giải phẫu trong đường tiêu hóa"
            outcome_key = "region_result"
        
        return f"""Bạn là một AI chuyên gia về phân loại hình ảnh nội soi tiêu hóa, chuyên về {type_desc}.
Nhiệm vụ của bạn là phân tích hình ảnh để xác định {type_desc} chính xác.

Bạn có thể sử dụng công cụ sau:
1. {self.classifier_type}_classifier: Công cụ phân loại hình ảnh
   - Tham số: image_path (str)
   - Kết quả: lớp được phân loại, độ tin cậy, và thông tin bổ sung

Quy trình làm việc của bạn:
1. Xác định hình ảnh cần phân loại
2. Sử dụng công cụ {self.classifier_type}_classifier để phân loại hình ảnh
3. Phân tích kết quả phân loại và độ tin cậy
4. Nếu độ tin cậy thấp, giải thích lý do có thể
5. Đưa ra kết luận cuối cùng về {type_desc}

Khi trả lời:
- Xác nhận lớp được phân loại với độ tin cậy
- Giải thích đặc điểm của lớp phân loại trong ngữ cảnh y tế
- Đề xuất các bước tiếp theo nếu phù hợp

Bạn phải trả về JSON với định dạng:
```json
{{
  "{outcome_key}": {{
    "success": true/false,
    "class_name": "tên lớp",
    "confidence": confidence_value,
    "description": "mô tả chi tiết về lớp phân loại",
    "analysis": "phân tích chuyên môn về kết quả phân loại"
  }}
}}
```"""

    def initialize(self) -> bool:
        """Khởi tạo agent và các công cụ."""
        try:
            # Tools are already initialized in _register_tools
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize classifier agent: {str(e)}")
            self.initialized = False
            return False
    
    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract task-specific input from state."""
        return {
            "image_path": state.get("image_path", ""),
            "query": state.get("query", ""),
            "classifier_type": self.classifier_type
        }
    
    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for LLM prompt."""
        image_path = task_input.get("image_path", "")
        classifier_type = task_input.get("classifier_type", "")
        
        return f"""Hình ảnh cần phân loại: {image_path}
        
Loại phân loại: {classifier_type}

Hãy phân loại hình ảnh này theo {classifier_type}. Sử dụng công cụ có sẵn để phân loại và phân tích.
Trả lời theo định dạng:

Tool: [tên công cụ]
Parameters: [tham số dưới dạng JSON]

Sau khi sử dụng công cụ, hãy phân tích kết quả và đưa ra nhận xét chuyên môn.
"""
    
    def _extract_agent_result(self, synthesis: str) -> Dict[str, Any]:
        """Extract agent result from LLM synthesis."""
        try:
            # Try to extract JSON
            json_start = synthesis.find('{')
            json_end = synthesis.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = synthesis[json_start:json_end]
                result = json.loads(json_str)
                return result
            
            # Fallback: Create result from synthesis text
            outcome_key = f"{self.classifier_type}_result"
            return {
                outcome_key: {
                    "success": True,
                    "class_name": "Unknown",
                    "confidence": 0.5,
                    "description": "Could not determine class from analysis",
                    "analysis": synthesis
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract agent result: {str(e)}")
            outcome_key = f"{self.classifier_type}_result"
            return {
                outcome_key: {
                    "success": False,
                    "error": str(e),
                    "analysis": synthesis
                }
            }