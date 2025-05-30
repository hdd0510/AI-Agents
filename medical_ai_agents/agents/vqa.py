#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - VQA Agent (MODIFIED for text-only support)
---------------------------
Agent trả lời câu hỏi về hình ảnh với LLM controller và LLaVA tool.
"""

import json
import os
from typing import Dict, Any, List
import logging
import re
import base64

from medical_ai_agents.agents.base_agent import BaseAgent
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.vqa.llava_tools import LLaVATool
from langchain.schema import SystemMessage, HumanMessage

class VQAAgent(BaseAgent):
    """Agent trả lời câu hỏi về hình ảnh y tế và câu hỏi text-only sử dụng LLM controller."""
    
    def __init__(self, model_path: str, llm_model: str = "gpt-4o-mini", device: str = "cuda"):
        """
        Khởi tạo VQA Agent với LLM controller.
        
        Args:
            model_path: Đường dẫn đến LLaVA model
            llm_model: Mô hình LLM sử dụng làm controller
            device: Device để chạy model (cuda/cpu)
        """
        # Set attributes before super().__init__()
        self.model_path = model_path
        
        super().__init__(name="VQA Agent", llm_model=llm_model, device=device)
        self.llava_tool = None
    
    def _register_tools(self) -> List[BaseTool]:
        """Register tools for this agent."""
        self.llava_tool = LLaVATool(model_path=self.model_path, device=self.device)
        return [self.llava_tool]
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines this agent's role."""
        prompt = """
Bạn là một AI chuyên gia y tế với khả năng chuyên sâu về y tế và bạn có khả năng:
1. Trả lời câu hỏi dựa trên hình ảnh nội soi tiêu hóa (sử dụng công cụ llava_vqa)
2. Tư vấn y tế chuyên khoa dựa trên input (sử dụng kiến thức chuyên môn của bạn)

Công cụ có sẵn:
- llava_vqa: Công cụ trả lời câu hỏi dựa trên hình ảnh sử dụng mô hình LLaVA
  - Tham số: image_path (str), question (str), medical_context (Dict, optional)
  - CHỈ sử dụng khi có hình ảnh

Quy trình làm việc:
1. Kiểm tra xem có hình ảnh không
2. Nếu có hình ảnh: sử dụng công cụ llava_vqa
3. Nếu không có hình ảnh: dựa vào kiến thức y khoa chuyên môn để trả lời trực tiếp
4. Tổng hợp và đưa ra câu trả lời cuối cùng

Khi trả lời câu hỏi text-only, hãy:
- Sử dụng kiến thức y khoa chuyên sâu
- Đưa ra lời khuyên phù hợp với vai trò bác sĩ chuyên khoa
- Khuyến nghị khám trực tiếp khi cần thiết
- Không chẩn đoán chính xác 100% qua text
"""
        return prompt

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
                for i, obj in enumerate(objects):
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
            "medical_context": medical_context,
            "is_text_only": state.get("is_text_only", False)
        }
    
    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for LLM prompt."""
        image_path = task_input.get("image_path", "")
        query = task_input.get("query", "")
        context = task_input.get("medical_context", {})
        is_text_only = task_input.get("is_text_only", False)
        
        context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()]) if context else "None"
        
        # MODIFIED: Handle text-only mode
        if is_text_only or not image_path or not os.path.exists(image_path):
            return f"""Đây là câu hỏi y tế TEXT-ONLY (không có hình ảnh):

Câu hỏi: {query}

Thông tin y tế bổ sung:
{context_str}

Hãy trả lời câu hỏi này dựa trên kiến thức y khoa chuyên môn của bạn.
KHÔNG sử dụng công cụ llava_vqa vì không có hình ảnh.
Trả lời trực tiếp như một bác sĩ chuyên khoa giàu kinh nghiệm.
"""
        else:
            return f"""Hình ảnh cần phân tích: {image_path}
                
Câu hỏi: {query if query else "Mô tả những gì bạn thấy trong hình ảnh này"}

Thông tin y tế bổ sung:
{context_str}

Hãy sử dụng công cụ llava_vqa để trả lời câu hỏi này dựa trên hình ảnh.
Trả lời theo định dạng sau:

Tool: llava_vqa
Parameters: {{"image_path": "{image_path}", "question": "{query if query else 'Mô tả những gì bạn thấy trong hình ảnh này'}", "medical_context": {json.dumps(context)}}}
"""
    
    def _format_synthesis_input(self) -> str:
        return """
Dựa trên kết quả từ tools hoặc kiến thức y khoa của bạn, hãy tổng hợp:
- Câu trả lời chi tiết và chuyên môn
- Phân tích dựa trên thông tin có sẵn
- Khuyến nghị y tế phù hợp
- Lời khuyên về các bước tiếp theo nếu cần

Trả lời theo định dạng sau:
```json
{
    "vqa_result": {
        "success": true/false,
        "answer": "câu trả lời chi tiết từ tools hoặc kiến thức chuyên môn",
        "analysis": "phân tích tổng hợp cuối cùng"
    }
}
```
"""
    
    def _parse_tool_calls(self, plan: str) -> List[Dict[str, Any]]:
        """Enhanced parsing for VQA agent - handle text-only mode."""
        tool_calls = []
        
        # Check if this is text-only mode (no tool calls expected)
        if "KHÔNG sử dụng công cụ" in plan or "không có hình ảnh" in plan:
            self.logger.info("[VQA] Text-only mode detected, no tool calls needed")
            return []  # No tool calls for text-only
        
        # Method 1: Standard format parsing
        lines = plan.split("\n")
        current_tool = None
        current_params = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("Tool:"):
                # Save previous tool if exists
                if current_tool:
                    tool_calls.append({
                        "tool_name": current_tool,
                        "params": current_params
                    })
                
                # Start new tool
                current_tool = line.replace("Tool:", "").strip()
                current_params = {}
            
            elif line.startswith("Parameters:"):
                # Try to parse JSON parameters
                try:
                    params_text = line.replace("Parameters:", "").strip()
                    if params_text.startswith("{") and params_text.endswith("}"):
                        current_params = json.loads(params_text)
                except Exception as e:
                    self.logger.warning(f"Failed to parse parameters: {line}, error: {e}")
        
        # Add last tool
        if current_tool:
            tool_calls.append({
                "tool_name": current_tool,
                "params": current_params
            })
        
        # Method 2: If no standard format found, try regex extraction
        if not tool_calls:
            # Look for llava_vqa tool mentions
            llava_pattern = r'llava_vqa.*?[\{]([^}]+)[\}]'
            matches = re.findall(llava_pattern, plan, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                try:
                    # Clean up the match and try to parse as JSON
                    param_str = "{" + match + "}"
                    params = json.loads(param_str)
                    tool_calls.append({
                        "tool_name": "llava_vqa",
                        "params": params
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to parse regex match: {match}, error: {e}")
        
        self.logger.info(f"[VQA] Parsed {len(tool_calls)} tool calls")
        return tool_calls
    
    def _extract_agent_result(self, synthesis: str) -> Dict[str, Any]:
        """Extract agent result from LLM synthesis."""
        try:
            # Try to extract JSON from synthesis
            json_start = synthesis.find('{')
            json_end = synthesis.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = synthesis[json_start:json_end]
                vqa_result = json.loads(json_str)
                self.logger.info(f"[VQA] Successfully extracted JSON result")
                return vqa_result
            
            # If no JSON found, create result from synthesis text
            self.logger.warning("[VQA] No JSON found in synthesis, parsing text")
            
            # Try to extract answer from text
            answer = ""
            # Look for answer patterns
            answer_patterns = [
                r'answer["\s]*:["\s]*([^"\n]+)',
                r'Answer[:\s]+([^\n]+)',
                r'câu trả lời[:\s]+([^\n]+)',
            ]
            for pattern in answer_patterns:
                matches = re.findall(pattern, synthesis, re.IGNORECASE)
                if matches:
                    answer = matches[0].strip()
                    break
            
            # If still no answer found, use the whole synthesis as answer
            if not answer:
                answer = synthesis
                
            # Fallback: Create result from synthesis text
            return {
                "vqa_result": {
                    "success": True,
                    "answer": answer,
                    "analysis": "Generated from LLM synthesis (text-only or parsing fallback)"
                }
            }
        except Exception as e:
            self.logger.error(f"[VQA] Failed to extract agent result: {str(e)}")
            return {
                "vqa_result": {
                    "success": False,
                    "error": str(e),
                    "answer": synthesis
                }
            }

    def _process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state using LLM controller and tools."""
        # Extract relevant information from state
        task_input = self._extract_task_input(state)
        self.logger.info(f"[VQA] Task input: {json.dumps(task_input, indent=2)}")
        
        # Let LLM decide which tools to use and how
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._format_task_input(task_input))
        ]
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        plan = response.content
        self.logger.info(f"[VQA] LLM plan: {plan}")
        
        # Parse the plan and execute tools
        tool_calls = self._parse_tool_calls(plan)
        self.logger.info(f"[VQA] Parsed tool calls: {json.dumps(tool_calls, indent=2)}")
        
        results = {}
        tool_outputs = {}
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            params = tool_call.get("params", {})
            
            if tool_name:
                self.logger.info(f"[VQA] Processing tool call: {tool_name}")
                self.logger.info(f"[VQA] Initial parameters: {json.dumps(params, indent=2)}")
                
                # Execute the tool
                tool_result = self.execute_tool(tool_name, **params)
                results[tool_name] = tool_result
                tool_outputs[tool_name] = tool_result
                self.logger.info(f"[VQA] Tool {tool_name} completed with result: {json.dumps(tool_result, indent=2)}")
        
        # Let LLM synthesize final result with actual image
        image_path = task_input.get("image_path", "")
        synthesis_text = f"""Tool results: {json.dumps(results, indent=2)}

Original task input:
Query: {task_input.get('query', '')}
Medical context: {json.dumps(task_input.get('medical_context', {}), indent=2)}

{self._format_synthesis_input()}
"""
        
        # Create the synthesis message with image using base64 encoding
        img_base64 = None
        if image_path and os.path.exists(image_path):
            try:
                # Convert image to base64
                with open(image_path, "rb") as img_file:
                    img_bytes = img_file.read()
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                # Create message with both text and base64-encoded image
                content = [
                    {"type": "text", "text": synthesis_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
                
                # Add the message with image to messages
                messages.append(HumanMessage(content=content))
                self.logger.info(f"[VQA] Including base64 image in synthesis prompt")
            except Exception as e:
                self.logger.error(f"[VQA] Failed to encode image for synthesis: {str(e)}")
                messages.append(HumanMessage(content=synthesis_text))
        else:
            self.logger.warning(f"[VQA] Image path not found or invalid: {image_path}")
            messages.append(HumanMessage(content=synthesis_text))
            
        # Invoke LLM for synthesis
        synthesis_response = self.llm.invoke(messages)
        self.logger.info(f"[VQA] Synthesis response: {synthesis_response.content}")
        
        # Return agent result
        agent_result = self._extract_agent_result(synthesis_response.content)
        self.logger.info(f"[VQA] Final agent result: {json.dumps(agent_result, indent=2)}")
        return {**state, **agent_result}