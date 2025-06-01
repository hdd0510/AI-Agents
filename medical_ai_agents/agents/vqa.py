#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - VQA Agent (MODIFIED: Always Use LLaVA)
---------------------------
Agent luôn sử dụng LLaVA tool cho cả image và text-only queries.
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
    """Agent luôn sử dụng LLaVA tool cho mọi queries (có hoặc không có hình ảnh)."""
    
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
Bạn là một AI chuyên gia y tế với khả năng sử dụng LLaVA (Large Language and Vision Assistant) để:

1. **Phân tích hình ảnh y tế** khi có hình ảnh
2. **Tư vấn y tế chuyên sâu** khi chỉ có text (text-only)

**Công cụ chính:**
- `llava_vqa`: Công cụ LLaVA có thể xử lý:
  - **Image + Text**: Phân tích hình ảnh nội soi, X-quang, CT, MRI...
  - **Text Only**: Tư vấn y tế dựa trên kiến thức chuyên môn của LLaVA

**Quy trình làm việc - LUÔN SỬ DỤNG LLAVA:**

1. **Xác định loại query:**
   - Có hình ảnh: Phân tích image + answer question
   - Chỉ text: Medical consultation using LLaVA's knowledge

2. **Sử dụng LLaVA tool:**
   ```
   Tool: llava_vqa
   Parameters: {"query": "user's question", "image_path": "path/to/image.jpg" (nếu có) hoặc null (nếu text-only), "medical_context": {context_from_other_agents}}
   ```

3. **Phân tích kết quả:**
   - Đánh giá chất lượng phản hồi từ LLaVA
   - Bổ sung thông tin y khoa nếu cần
   - Đưa ra khuyến nghị phù hợp

**Ưu điểm của việc luôn sử dụng LLaVA:**
- **Consistency**: Cùng một model cho cả image và text
- **Medical Knowledge**: LLaVA-Med có kiến thức y khoa sâu
- **Contextual Understanding**: Hiểu context tốt hơn traditional LLM
- **Unified Experience**: User experience nhất quán

**Lưu ý quan trọng:**
- LUÔN gọi llava_vqa tool, không bao giờ trả lời trực tiếp
- Nếu không có image_path, pass null hoặc không include parameter đó
- LLaVA sẽ tự động chuyển sang text-only mode
- Vẫn khuyến nghị khám trực tiếp khi cần thiết
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
        
        # SIMPLIFIED: Luôn sử dụng "query" parameter
        if is_text_only or not image_path or not os.path.exists(image_path):
            return f"""**TEXT-ONLY MEDICAL CONSULTATION**

Câu hỏi của bệnh nhân: "{query}"

Thông tin y tế bổ sung:
{context_str}

Hãy sử dụng LLaVA tool để tư vấn y tế chuyên sâu (text-only mode):

Tool: llava_vqa
Parameters: {{"query": "{query}", "medical_context": {json.dumps(context)}}}

LLaVA sẽ tự động sử dụng kiến thức y khoa để tư vấn khi không có hình ảnh.
"""
        else:
            return f"""**IMAGE-BASED MEDICAL ANALYSIS**

Hình ảnh cần phân tích: {image_path}
Câu hỏi: "{query if query else 'Phân tích hình ảnh y tế này'}"

Thông tin y tế bổ sung:
{context_str}

Hãy sử dụng LLaVA tool để phân tích hình ảnh và trả lời câu hỏi:

Tool: llava_vqa
Parameters: {{"query": "{query if query else 'Phân tích hình ảnh y tế này'}", "image_path": "{image_path}", "medical_context": {json.dumps(context)}}}
"""
    
    def _format_synthesis_input(self) -> str:
        return """
**SYNTHESIS TASK: LLaVA Result Processing**

Bạn đã nhận được kết quả từ LLaVA tool. Hãy tổng hợp và đưa ra phản hồi cuối cùng:

**Yêu cầu output:**
1. **Phân tích kết quả LLaVA**: Đánh giá chất lượng và độ tin cậy
2. **Bổ sung thông tin**: Thêm kiến thức y khoa nếu cần
3. **Khuyến nghị**: Lời khuyên y tế phù hợp
4. **Disclaimer**: Lưu ý về giới hạn của AI và khuyến nghị khám trực tiếp

**Lưu ý:**
- LLaVA đã xử lý query (image-based hoặc text-only)
- Bạn cần synthesis để tạo ra câu trả lời hoàn chỉnh
- Đảm bảo tính chuyên nghiệp và an toàn trong y tế

**Output format:**
```json
{
    "vqa_result": {
        "success": true/false,
        "answer": "câu trả lời hoàn chỉnh sau khi synthesis kết quả LLaVA",
        "analysis": "phân tích về kết quả từ LLaVA và bổ sung thông tin",
        "llava_response": "original response from LLaVA",
        "query_type": "image_based/text_only",
        "confidence": 0.0-1.0
    }
}
```
"""
    
    def _parse_tool_calls(self, plan: str) -> List[Dict[str, Any]]:
        """SIMPLIFIED parsing for LLaVA tool calls."""
        tool_calls = []
        
        self.logger.info(f"[VQA] Parsing LLM plan:\n{plan}")
        
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
        
        # Method 2: Regex extraction as fallback
        if not tool_calls:
            llava_pattern = r'llava_vqa.*?[\{]([^}]+)[\}]'
            matches = re.findall(llava_pattern, plan, re.DOTALL | re.IGNORECASE)
            
            for match in matches:
                try:
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
                    "analysis": "Generated from LLaVA tool execution",
                    "query_type": "llava_processed"
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
        """SIMPLIFIED process state - no complex fallbacks needed."""
        # Extract relevant information from state
        task_input = self._extract_task_input(state)
        self.logger.info(f"[VQA] Processing task: {json.dumps(task_input, indent=2)}")
        
        # Let LLM decide how to use LLaVA tool
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._format_task_input(task_input))
        ]
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        plan = response.content
        self.logger.info(f"[VQA] LLM plan: {plan}")
        
        # Parse the plan and execute LLaVA tool
        tool_calls = self._parse_tool_calls(plan)
        self.logger.info(f"[VQA] Parsed tool calls: {json.dumps(tool_calls, indent=2)}")
        
        # SIMPLE FALLBACK: If no tool calls, create one from task_input
        if not tool_calls:
            self.logger.warning("[VQA] No tool calls found, creating simple fallback")
            fallback_params = {"query": task_input.get("query", "Medical consultation")}
            
            # Add image_path if available
            if task_input.get("image_path") and os.path.exists(task_input["image_path"]):
                fallback_params["image_path"] = task_input["image_path"]
            
            # Add medical context
            if task_input.get("medical_context"):
                fallback_params["medical_context"] = task_input["medical_context"]
            
            tool_calls = [{"tool_name": "llava_vqa", "params": fallback_params}]
            self.logger.info(f"[VQA] Simple fallback created")
        
        # Execute tool calls
        results = {}
        llava_success = False
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            params = tool_call.get("params", {})
            
            if tool_name == "llava_vqa":
                self.logger.info(f"[VQA] Executing LLaVA tool with params: {json.dumps(params, indent=2)}")
                
                # Execute LLaVA tool
                tool_result = self.execute_tool(tool_name, **params)
                results[tool_name] = tool_result
                
                # Check success and response quality
                llava_success = tool_result.get("success", False)
                llava_answer = tool_result.get("answer", "")
                
                self.logger.info(f"[VQA] LLaVA completed: success={llava_success}, answer_length={len(llava_answer)}")
                
                # Quality check
                if llava_success and llava_answer:
                    meaningless_indicators = [
                        "I cannot", "I am not able", "I don't have", 
                        "cannot analyze", "unable to", "no information"
                    ]
                    
                    is_meaningless = any(indicator in llava_answer.lower() for indicator in meaningless_indicators)
                    is_too_short = len(llava_answer.strip()) < 50
                    
                    if is_meaningless or is_too_short:
                        self.logger.warning(f"[VQA] Poor quality response detected")
                        llava_success = False
                        tool_result["success"] = False
                        tool_result["error"] = "LLaVA response quality insufficient"
        
        # SAFETY CHECK: If LLaVA failed, return error immediately
        if not llava_success:
            self.logger.error("[VQA] LLaVA failed - returning safety error")
            return {
                **state,
                "vqa_result": {
                    "success": False,
                    "error": "LLaVA medical consultation failed",
                    "answer": "❌ **Lỗi hệ thống tư vấn y tế**\n\nHệ thống LLaVA gặp sự cố và không thể thực hiện tư vấn an toàn.\n\n🏥 **Khuyến nghị:** Vui lòng thử lại hoặc tham khảo bác sĩ chuyên khoa trực tiếp.",
                    "safety_action": "Refused to generate medical advice when tool failed"
                }
            }
        
        # LLaVA succeeded - proceed with synthesis
        self.logger.info("[VQA] LLaVA succeeded - proceeding with synthesis")
        
        synthesis_text = f"""**MEDICAL CONSULTATION SYNTHESIS**

Task: Query="{task_input.get('query')}", Image={bool(task_input.get('image_path'))}, Text-only={task_input.get('is_text_only', False)}

**LLaVA Tool Status: ✅ SUCCESS**
LLaVA Response: {results['llava_vqa'].get('answer', '')[:200]}...

{self._format_synthesis_input()}
"""
        
        synthesis_messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=synthesis_text)
        ]
            
        # Invoke LLM for synthesis
        synthesis_response = self.llm.invoke(synthesis_messages)
        
        # Return agent result
        agent_result = self._extract_agent_result(synthesis_response.content)
        return {**state, **agent_result}