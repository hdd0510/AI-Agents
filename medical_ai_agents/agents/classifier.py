#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - Enhanced Classifier Agent
--------------------------------------------
Agent phân loại hình ảnh với LLM controller và classifier tool - Enhanced flow.
"""

import json
from typing import Dict, Any, List
import logging
import re

from medical_ai_agents.agents.base_agent import BaseAgent
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.classifier.cls_tools import ClassifierTool

class ClassifierAgent(BaseAgent):
    """Agent phân loại hình ảnh nội soi sử dụng LLM controller."""
    
    def __init__(self, model_path: str, class_names: List[str], 
                classifier_type: str = "modality", llm_model: str = "gpt-4o-mini", device: str = "cuda"):
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
            examples = """
            - WLI (White Light Imaging): Ánh sáng trắng tiêu chuẩn
            - BLI (Blue Light Imaging): Ánh sáng xanh tăng cường mạch máu
            - FICE: Công nghệ cải thiện màu sắc
            - LCI (Linked Color Imaging): Tăng cường độ tương phản màu
            """
        else:
            type_desc = "vị trí giải phẫu trong đường tiêu hóa"
            outcome_key = "region_result"
            examples = """
            - Hau_hong: Hầu họng
            - Thuc_quan: Thực quản  
            - Tam_vi: Tâm vị (giao giữa thực quản và dạ dày)
            - Than_vi: Thân vị (thân dạ dày)
            - Phinh_vi: Phình vị (đáy dạ dày)
            - Hang_vi: Hang vị (phần dưới dạ dày)
            - Bo_cong_lon/Bo_cong_nho: Bờ cong lớn/nhỏ của dạ dày
            - Hanh_ta_trang: Hành tá tràng
            - Ta_trang: Tá tràng
            """
        
        return f"""Bạn là một AI chuyên gia về phân loại hình ảnh nội soi tiêu hóa, chuyên về {type_desc}.
Nhiệm vụ của bạn là phân tích hình ảnh để xác định {type_desc} chính xác.

Các lớp phân loại có thể:
{examples}

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

Khi sử dụng công cụ, trả lời theo định dạng:
Tool: {self.classifier_type}_classifier
Parameters: {{"image_path": "path/to/image.jpg"}}

Sau khi có kết quả từ công cụ, hãy phân tích và đưa ra nhận xét chuyên môn."""

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

Tool: {classifier_type}_classifier
Parameters: {{"image_path": "{image_path}"}}

Sau khi sử dụng công cụ, hãy phân tích kết quả và đưa ra nhận xét chuyên môn.
"""
    
    def _format_synthesis_input(self) -> str:
        """Format synthesis input for LLM prompt."""
        outcome_key = f"{self.classifier_type}_result"
        type_desc = "kỹ thuật chụp nội soi" if self.classifier_type == "modality" else "vị trí giải phẫu"
        
        return f"""
Dựa trên kết quả từ tools, bạn phải xác định:
- Lớp phân loại được xác định cho {type_desc}
- Độ tin cậy của kết quả phân loại
- Giải thích về đặc điểm của lớp phân loại
- Nếu độ tin cậy thấp, hãy giải thích lý do có thể

Bạn phải trả về JSON với định dạng:
```json
{{
  "{outcome_key}": {{
    "success": true/false,
    "class_name": "tên lớp được phân loại",
    "confidence": confidence_value,
    "description": "mô tả chi tiết về lớp phân loại",
    "analysis": "phân tích chuyên môn về kết quả phân loại"
  }}
}}
```
        """
    
    def _parse_tool_calls(self, plan: str) -> List[Dict[str, Any]]:
        """Enhanced parsing for tool calls with better error handling."""
        tool_calls = []
        
        # Method 1: Standard format parsing
        lines = plan.split("\n")
        current_tool = None
        current_params = {}
        
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith("Tool:"):
                # Save previous tool if exists
                if current_tool:
                    tool_calls.append({
                        "tool_name": current_tool,
                        "params": current_params
                    })
                
                # Start new tool
                current_tool = line_stripped.replace("Tool:", "").strip()
                current_params = {}
            
            elif line_stripped.startswith("Parameters:"):
                # Try to parse JSON parameters
                try:
                    params_text = line_stripped.replace("Parameters:", "").strip()
                    
                    # Handle both single line and multiline JSON
                    if params_text.startswith("{"):
                        # Try to find complete JSON
                        json_str = params_text
                        if not params_text.endswith("}"):
                            # Look for closing brace in subsequent lines
                            line_idx = lines.index(line)
                            for next_line in lines[line_idx + 1:]:
                                json_str += " " + next_line.strip()
                                if "}" in next_line:
                                    break
                        
                        current_params = json.loads(json_str)
                        
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.warning(f"Failed to parse parameters: {line_stripped}, error: {e}")
                    current_params = {}
        
        # Add last tool
        if current_tool:
            tool_calls.append({
                "tool_name": current_tool,
                "params": current_params
            })
        
        # Method 2: Regex extraction as fallback
        if not tool_calls:
            import re
            
            # Look for tool patterns specific to classifier
            tool_pattern = rf'Tool:\s*({self.classifier_type}_classifier)'
            param_pattern = r'Parameters:\s*(\{[^}]*\})'
            
            tools = re.findall(tool_pattern, plan)
            params = re.findall(param_pattern, plan, re.DOTALL)
            
            for i, tool in enumerate(tools):
                param_dict = {}
                if i < len(params):
                    try:
                        param_dict = json.loads(params[i])
                    except:
                        pass
                
                tool_calls.append({
                    "tool_name": tool,
                    "params": param_dict
                })
        
        # Method 3: Fallback - if still no tools, create default call
        if not tool_calls:
            self.logger.warning("No tool calls parsed, creating default classifier call")
            # Try to extract image_path from the plan text
            image_path = ""
            image_matches = re.findall(r'image_path["\s]*:["\s]*([^"]+)', plan)
            if image_matches:
                image_path = image_matches[0]
            elif "image_path" in plan:
                # Fallback extraction
                lines = plan.split('\n')
                for line in lines:
                    if 'image_path' in line and ':' in line:
                        try:
                            image_path = line.split(':')[1].strip().strip('"')
                            break
                        except:
                            pass
            
            # If we found image_path, create the tool call
            if image_path:
                tool_calls.append({
                    "tool_name": f"{self.classifier_type}_classifier",
                    "params": {
                        "image_path": image_path
                    }
                })
        
        self.logger.info(f"[Classifier] Parsed {len(tool_calls)} tool calls")
        return tool_calls

    def _extract_agent_result(self, synthesis: str) -> Dict[str, Any]:
        """Extract agent result from LLM synthesis - Enhanced like detector/VQA."""
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
                result = json.loads(json_str)
                self.logger.info(f"[Classifier] Successfully extracted JSON result")
                return result
            
            # Method 3: Fallback - create result from synthesis text
            self.logger.warning("[Classifier] No JSON found, creating fallback result")
            
            # Try to extract class name and confidence from text
            class_name = "Unknown"
            confidence = 0.5
            
            # Look for class name patterns
            class_patterns = [
                r'class[_\s]*name["\s]*:["\s]*([^"\n,]+)',
                r'classified[:\s]+as[:\s]+([^\n,]+)',
                r'result[:\s]+([^\n,]+)',
            ]
            for pattern in class_patterns:
                matches = re.findall(pattern, synthesis, re.IGNORECASE)
                if matches:
                    class_name = matches[0].strip()
                    break
            
            # Look for confidence patterns
            conf_patterns = [
                r'confidence["\s]*:["\s]*([0-9.]+)',
                r'confidence[:\s]+([0-9.]+)',
                r'([0-9.]+)%?\s*confidence',
            ]
            for pattern in conf_patterns:
                matches = re.findall(pattern, synthesis, re.IGNORECASE)
                if matches:
                    try:
                        confidence = float(matches[0])
                        if confidence > 1.0:  # Handle percentage
                            confidence = confidence / 100.0
                        break
                    except:
                        pass
            
            # Create fallback result
            outcome_key = f"{self.classifier_type}_result"
            return {
                outcome_key: {
                    "success": True,
                    "class_name": class_name,
                    "confidence": confidence,
                    "description": f"Classified as {class_name}",
                    "analysis": "Generated from LLM synthesis parsing"
                }
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"[Classifier] JSON decode error: {str(e)}")
            self.logger.error(f"[Classifier] Attempted to parse: {json_str}")
            
            # Fallback result
            outcome_key = f"{self.classifier_type}_result"
            return {
                outcome_key: {
                    "success": False,
                    "error": f"JSON parsing failed: {str(e)}",
                    "analysis": synthesis
                }
            }
        except Exception as e:
            self.logger.error(f"[Classifier] Failed to extract agent result: {str(e)}")
            outcome_key = f"{self.classifier_type}_result"
            return {
                outcome_key: {
                    "success": False,
                    "error": str(e),
                    "analysis": synthesis
                }
            }