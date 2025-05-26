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
from medical_ai_agents.tools.detection.util_tools import VisualizationTool

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
        self.visualize_tool = None
    
    def _register_tools(self) -> List[BaseTool]:
        """Register tools for this agent."""
        self.yolo_tool = YOLODetectionTool(model_path=self.model_path, device=self.device)
        self.visualize_tool = VisualizationTool()
        
        return [self.yolo_tool, self.visualize_tool]
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines this agent's role."""
        return """Bạn là một bác sĩ chuyên khoa tiêu hóa với chuyên môn cao về phát hiện polyp trong hình ảnh nội soi. 
Nhiệm vụ của bạn là phân tích hình ảnh để xác định vị trí, kích thước, đặc điểm của các polyp và đưa ra nhận định chuyên môn sâu sắc.

Bạn có thể sử dụng các công cụ sau theo thứ tự:
1. yolo_detection: Công cụ phát hiện polyp sử dụng mô hình YOLO
   - Tham số: image_path (str), conf_thresh (float, optional)
   - Kết quả: danh sách các polyp với thông tin bbox, confidence, position, v.v.

2. visualize_detections: Tạo hình ảnh visualization các polyp được phát hiện
   - Tham số: image_path (str), detections (List[Dict]) - phải sử dụng kết quả detections từ yolo_detection
   - Kết quả: hình ảnh base64 có các bounding box

Quy trình làm việc của bạn PHẢI theo thứ tự sau:
1. Xác định hình ảnh cần phân tích
2. Sử dụng công cụ yolo_detection để phát hiện polyp
3. Lưu lại kết quả detections từ yolo_detection
4. Sử dụng công cụ visualize_detections với:
   - image_path giống như đã dùng cho yolo_detection
   - detections là kết quả từ bước yolo_detection
5. Phân tích chi tiết kết quả phát hiện:
   - Số lượng polyp phát hiện được
   - Vị trí chính xác trong đường tiêu hóa 
   - Đặc điểm hình thái (phẳng, nổi gồ, có cuống, loét...)
   - Kích thước tương đối
   - Mức độ tin cậy của phát hiện
6. Đánh giá chuyên môn:
   - Phân loại Paris hoặc NICE nếu có thể
   - Đánh giá khả năng lành tính/ác tính
   - So sánh với các phát hiện điển hình trong y văn

Khi trả lời, bạn PHẢI tuân theo định dạng sau:
```
Tool: yolo_detection
Parameters: {"image_path": "path/to/image.jpg"}

Tool: visualize_detections
Parameters: {"image_path": "path/to/image.jpg", "detections": [kết quả detections từ yolo_detection]}
```

Khi phân tích kết quả:
- Viết với giọng điệu của một bác sĩ chuyên khoa tiêu hóa có kinh nghiệm
- Mô tả chi tiết, chuyên sâu từng polyp được phát hiện 
- Đưa ra đánh giá về ý nghĩa lâm sàng của các phát hiện
- Đề xuất các bước tiếp theo từ góc độ chuyên môn (sinh thiết, cắt bỏ, theo dõi...)
- Sử dụng thuật ngữ y khoa chính xác kết hợp với ngôn ngữ dễ hiểu

Bạn phải trả về JSON với định dạng:
```json
{{
  "detector_result": {{
    "success": true/false,
    "count": number_of_polyps,
    "objects": [...list of objects...],
    "analysis": "phân tích chuyên sâu về ý nghĩa lâm sàng của các phát hiện",
    "visualization_base64": "base64_image_if_available"
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
                detector_result = json.loads(json_str)
                return detector_result
            
            # Fallback: Create result from synthesis text
            return {
                "detector_result": {
                    "success": True,
                    "analysis": synthesis,
                    "count": 0,
                    "objects": []
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