#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - VQA Agent with ReAct Pattern (FIXED)
-------------------------------------------------------
Visual Question Answering Agent sử dụng ReAct framework với đầy đủ abstract methods.
"""

import json
import os
from typing import Dict, Any, List
import logging
import re
from langchain_core.messages import SystemMessage, HumanMessage

from medical_ai_agents.agents.base_agent import BaseAgent, ThoughtType
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.vqa.llava_tools import LLaVATool


class VQAAgent(BaseAgent):
    """VQA Agent với ReAct pattern cho medical question answering."""

    def __init__(self, model_path: str, llm_model: str = "gpt-4o-mini", device: str = "cuda"):
        self.model_path = model_path
        super().__init__(name="VQA ReAct Agent", llm_model=llm_model, device=device)

        self.max_iterations = 5  # VQA thường ít vòng hơn
        self.llava_tool = None

    def _register_tools(self) -> List[BaseTool]:
        """Register LLaVA tool."""
        self.llava_tool = LLaVATool(model_path=self.model_path, device=self.device)
        return [self.llava_tool]

    def _get_agent_description(self) -> str:
        """Get VQA agent description."""
        return (
            "I am a medical Visual Question Answering specialist with dual capabilities:\n\n"
            "1. Medical Image Analysis: I analyze medical images (endoscopy, X-rays, CT, MRI) with expert precision using LLaVA.\n"
            "2. Medical Consultation: I provide professional medical advice and consultation for text queries.\n"
            "My expertise includes interpreting findings, answering complex medical questions, and offering recommendations."
        )

    def initialize(self) -> bool:
        """Initialize VQA agent."""
        try:
            self.initialized = True
            self.logger.info("VQA ReAct Agent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize VQA agent: {str(e)}")
            self.initialized = False
            return False

    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract VQA task input với medical context."""
        medical_context = state.get("medical_context", {}).copy()

        # Thêm dữ liệu từ các agent khác (detector, modality, region)
        if "detector_result" in state:
            det = state["detector_result"]
            if det.get("success", False):
                medical_context["polyp_detection"] = {
                    "count": det.get("count", 0),
                    "objects": det.get("objects", [])[:3],
                    "clinical_summary": det.get("analysis", "")
                }
        if "modality_result" in state:
            mod = state["modality_result"]
            if mod.get("success", False):
                medical_context["imaging_modality"] = {
                    "type": mod.get("class_name", "Unknown"),
                    "confidence": mod.get("confidence", 0.0),
                    "description": mod.get("description", "")
                }
        if "region_result" in state:
            reg = state["region_result"]
            if reg.get("success", False):
                medical_context["anatomical_region"] = {
                    "location": reg.get("class_name", "Unknown"),
                    "confidence": reg.get("confidence", 0.0),
                    "description": reg.get("description", "")
                }

        return {
            "image_path": state.get("image_path", ""),
            "query": state.get("query", ""),
            "medical_context": medical_context,
            "is_text_only": state.get("is_text_only", False),
            "user_type": state.get("user_type", "patient"),
            "language_preference": state.get("language_preference", "vi")
        }

    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input cho ReAct processing (FIXED abstract method)."""
        image_path = task_input.get("image_path", "")
        query = task_input.get("query", "")
        context = task_input.get("medical_context", {})
        is_text_only = task_input.get("is_text_only", False)
        user_type = task_input.get("user_type", "patient")

        # Analyze query characteristics
        query_lower = query.lower()
        is_greeting = any(word in query_lower for word in ["hello", "hi", "xin chào", "chào"])
        is_medical = any(word in query_lower for word in ["polyp", "nội soi", "đau", "pain", "symptom"])

        # Build medical context string
        context_parts = []
        if "polyp_detection" in context:
            det = context["polyp_detection"]
            context_parts.append(f"- Polyp detection: {det['count']} polyp(s) found")
            if det.get("clinical_summary"):
                context_parts.append(f"- Clinical assessment: {det['clinical_summary'][:100]}...")
        if "imaging_modality" in context:
            mod = context["imaging_modality"]
            context_parts.append(f"- Imaging technique: {mod['type']} ({mod['confidence']:.1%} confidence)")
        if "anatomical_region" in context:
            reg = context["anatomical_region"]
            context_parts.append(f"- Anatomical location: {reg['location']} ({reg['confidence']:.1%} confidence)")
        
        context_str = "\n".join(context_parts) if context_parts else "No additional medical context"

        # Format based on mode (text-only vs image-based)
        if is_text_only or not image_path or not os.path.exists(image_path):
            prompt = f"""**Medical Consultation Task (Text-Only)**

User query: "{query}"
User type: {user_type}
Query characteristics: {"Greeting" if is_greeting else "Medical question" if is_medical else "General inquiry"}

Medical context from previous analysis:
{context_str}

Requirements:
1. Use llava_vqa tool for medical consultation
2. Provide appropriate response based on query type
3. For medical questions: Give professional consultation with assessment and recommendations
4. For greetings: Respond warmly and explain your capabilities
5. Adapt language and complexity to {user_type} level
6. Include appropriate medical disclaimers

Please proceed with the consultation using ReAct pattern."""
        else:
            prompt = f"""**Medical Visual Question Answering Task**

Image to analyze: {image_path}
User question: "{query if query else 'Please analyze this medical image comprehensively'}"
User type: {user_type}

Medical context from other analyses:
{context_str}

Requirements:
1. Use llava_vqa tool to analyze the image and answer the question
2. Integrate findings with the provided medical context
3. Provide interpretation at appropriate level for {user_type}
4. Include clinical significance and recommendations
5. Maintain medical accuracy while being accessible

Please proceed with the analysis using ReAct pattern."""
        
        return prompt

    def _format_agent_result(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format ReAct result into VQA agent output (FIXED abstract method)."""
        if not react_result.get("success", False):
            return {
                "vqa_result": {
                    "success": False,
                    "error": react_result.get("error", "VQA analysis failed"),
                    "reasoning_steps": len(self.react_history) if hasattr(self, 'react_history') else 0
                }
            }

        # Extract LLaVA response from ReAct history
        llava_response = ""
        query_type = "unknown"
        confidence = 0.5

        for step in self.react_history:
            if step.observation and "llava_vqa" in str(step.action):
                try:
                    obs_data = json.loads(step.observation)
                    if obs_data.get("success", False):
                        llava_response = obs_data.get("answer", "")
                        query_type = obs_data.get("query_type", "unknown")
                        # Estimate confidence based on response quality
                        confidence = self._estimate_confidence(llava_response)
                        break
                except json.JSONDecodeError:
                    continue

        # Get comprehensive analysis from final answer
        final_answer = react_result.get("answer", "")
        
        # If no good final answer, use LLaVA response
        if not final_answer and llava_response:
            final_answer = llava_response

        return {
            "vqa_result": {
                "success": True,
                "answer": final_answer,
                "analysis": f"VQA analysis completed using LLaVA with ReAct reasoning. Query type: {query_type}",
                "llava_response": llava_response,
                "query_type": query_type,
                "confidence": confidence,
                "reasoning_steps": len(self.react_history) if hasattr(self, 'react_history') else 0
            }
        }

    def _estimate_confidence(self, answer: str) -> float:
        """Estimate confidence based on answer quality."""
        if not answer or len(answer.strip()) < 10:
            return 0.3

        # Lower confidence indicators
        uncertainty_terms = [
            "i'm not sure", "unclear", "cannot determine", "difficult to say",
            "may be", "might be", "possibly", "uncertain", "not clear"
        ]

        answer_lower = answer.lower()
        confidence = 0.9

        for term in uncertainty_terms:
            if term in answer_lower:
                confidence -= 0.1
                break

        # Higher confidence indicators
        certainty_terms = [
            "clearly shows", "definitely", "obviously", "certain",
            "confident", "evident", "apparent"
        ]

        for term in certainty_terms:
            if term in answer_lower:
                confidence += 0.1
                break

        return max(0.0, min(1.0, confidence))

    # ===== HELPER METHODS (kept from original) =====
    
    def _format_synthesis_input(self) -> str:
        """Format synthesis input for LLM prompt."""
        return """**SYNTHESIS TASK: LLaVA Result Processing**

Bạn đã nhận được kết quả từ LLaVA tool. Hãy tổng hợp và đưa ra phản hồi cuối cùng:

**Yêu cầu output:**
1. Phân tích kết quả LLaVA: đánh giá chất lượng và độ tin cậy
2. Bổ sung thông tin: thêm kiến thức y khoa nếu cần
3. Khuyến nghị: lời khuyên y tế phù hợp
4. Disclaimer: lưu ý về giới hạn của AI và khuyến nghị khám trực tiếp

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
```"""

    def _parse_tool_calls(self, plan: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM plan."""
        tool_calls = []

        self.logger.info(f"[VQA] Parsing LLM plan:\n{plan}")

        lines = plan.split("\n")
        current_tool = None
        current_params = {}

        for line in lines:
            line = line.strip()
            if line.startswith("Tool:"):
                if current_tool:
                    tool_calls.append({"tool_name": current_tool, "params": current_params})
                current_tool = line.replace("Tool:", "").strip()
                current_params = {}
            elif line.startswith("Parameters:"):
                try:
                    params_text = line.replace("Parameters:", "").strip()
                    if params_text.startswith("{") and params_text.endswith("}"):
                        current_params = json.loads(params_text)
                except Exception as e:
                    self.logger.warning(f"Failed to parse parameters: {line}, error: {e}")

        if current_tool:
            tool_calls.append({"tool_name": current_tool, "params": current_params})

        # Fallback regex parsing
        if not tool_calls:
            llava_pattern = r'llava_vqa.*?[\{]([^}]+)[\}]'
            matches = re.findall(llava_pattern, plan, re.DOTALL | re.IGNORECASE)
            for match in matches:
                try:
                    param_str = "{" + match + "}"
                    params = json.loads(param_str)
                    tool_calls.append({"tool_name": "llava_vqa", "params": params})
                except Exception as e:
                    self.logger.warning(f"Failed to parse regex match: {match}, error: {e}")

        self.logger.info(f"[VQA] Parsed {len(tool_calls)} tool calls")
        return tool_calls

    def _extract_agent_result(self, synthesis: str) -> Dict[str, Any]:
        """Extract agent result from synthesis (legacy method for compatibility)."""
        try:
            json_start = synthesis.find('{')
            json_end = synthesis.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = synthesis[json_start:json_end]
                vqa_result = json.loads(json_str)
                self.logger.info(f"[VQA] Successfully extracted JSON result")
                return vqa_result

            self.logger.warning("[VQA] No JSON found in synthesis, parsing text")
            
            # Extract answer from text
            answer = ""
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
            
            if not answer:
                answer = synthesis
                
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