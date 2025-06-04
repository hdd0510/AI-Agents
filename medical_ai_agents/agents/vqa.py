#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TIERED VQA AGENT - APPROACH (Supportive)
================================================
Full reasoning freedom with goal-oriented guidance
Synthesis WITH image viewing to evaluate tool results
"""

import json
import os
import logging
import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from medical_ai_agents.agents.base_agent import BaseAgent, ThoughtType, ReActStep
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.vqa.llava_tools import LLaVATool

class VQAAgent(BaseAgent):
    """
    VQA Agent - Supportive Tier
    
    GOAL-ORIENTED REASONING with full freedom:
    - Choose optimal strategy based on query type
    - Adapt approach based on available context
    - Multiple reasoning paths available
    - Synthesis WITH image viewing for validation
    """
    
    def __init__(self, model_path: str, llm_model: str = "gpt-4o-mini", device: str = "cuda"):
        self.model_path = model_path
        super().__init__(name="VQA Agent", llm_model=llm_model, device=device)
        
        # configuration
        self.max_iterations = 6  # Most flexible: allow complex reasoning
        
        # Goal-oriented criteria
        self.primary_goal = "provide_comprehensive_medical_consultation"
        self.success_criteria = [
            "query_understood_and_analyzed",
            "relevant_medical_analysis_completed",
            "professional_advice_provided", 
            "safety_disclaimers_included"
        ]
        
        # strategies
        self.available_strategies = [
            "direct_llava_consultation",      # Simple image+text → answer
            "context_enhanced_analysis",      # Use previous results + LLaVA
            "multi_step_reasoning",          # Break down complex queries
            "comparative_analysis",          # Compare with medical standards
            "consultation_synthesis"         # Synthesize multiple perspectives
        ]
        
        # Termination criteria
        self.termination_conditions = [
            "confident_comprehensive_answer_provided",
            "safety_concern_requires_human_expert",
            "max_iterations_reached_with_best_effort"
        ]
        
        self.llava_tool = None
    
    def _register_tools(self) -> List[BaseTool]:
        """Register VQA tools."""
        self.llava_tool = LLaVATool(
            model_path=self.model_path,
            device=self.device
        )
        return [self.llava_tool]
    
    def _get_agent_description(self) -> str:
        """VQA agent description."""
        return """I am an medical consultation AI with full reasoning freedom.

PRIMARY GOAL: Provide comprehensive, accurate medical consultation

STRATEGIES (I choose based on context):
🎯 Direct Consultation: Straightforward image+text analysis via LLaVA
🔍 Context-Enhanced: Incorporate previous detection/classification results  
🧠 Multi-Step Reasoning: Break complex queries into manageable parts
📊 Comparative Analysis: Compare findings with medical standards
🏥 Consultation Synthesis: Integrate multiple medical perspectives

SYNTHESIS CAPABILITY: I can review input images during synthesis to validate my tool results and ensure accuracy.

TERMINATION: I stop when I've provided a confident, comprehensive answer or identified safety concerns requiring human expertise."""

    def _get_system_prompt(self) -> str:
        """system prompt with strategic guidance."""
        return f"""You are an medical consultation AI with full reasoning freedom.

PRIMARY GOAL: Provide comprehensive, accurate medical consultation for user queries.

STRATEGIES (choose based on situation):

STRATEGY 1 - Direct LLaVA Consultation:
Thought: This is a straightforward medical question, I'll use LLaVA directly
Action: llava_vqa
Action Input: {{"query": "<user_question>", "image_path": "<path>", "medical_context": {{}}}}

STRATEGY 2 - Context-Enhanced Analysis:
Thought: I have previous analysis results, I'll enhance LLaVA with this context
Action: llava_vqa  
Action Input: {{"query": "<enhanced_query>", "image_path": "<path>", "medical_context": {{previous_results}}}}

STRATEGY 3 - Multi-Step Reasoning:
Thought: This is complex, I'll break it down into steps
Action: llava_vqa
Action Input: {{"query": "<step_1_question>", ...}}
[Continue with additional steps as needed]

STRATEGY 4 - Comparative Analysis:
Thought: I need to compare findings with medical standards
[Multiple tool calls for comprehensive analysis]

STRATEGY 5 - Consultation Synthesis:
Thought: I'll gather multiple perspectives and synthesize them
[Multiple approaches combined]

TERMINATION CRITERIA:
✓ Confident, comprehensive answer provided
✓ Safety concern identified requiring human expert
✓ Maximum {self.max_iterations} iterations with best effort

QUALITY STANDARDS:
- Medical accuracy and appropriateness
- Clear explanations for patient understanding  
- Appropriate disclaimers and recommendations
- Professional tone with empathy
- Safety-first approach

Available tools: {self.tool_descriptions}

Analyze the situation and choose your strategy:"""

    def initialize(self) -> bool:
        """Initialize VQA agent."""
        try:
            self.initialized = True
            self.logger.info("VQA Agent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize VQA Agent: {str(e)}")
            return False

    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract VQA task input with comprehensive medical context."""
        # Build rich medical context from all previous agents
        medical_context = state.get("medical_context", {}).copy()
        
        # Add detection context
        if "detector_result" in state:
            detector = state["detector_result"]
            if detector.get("success", False):
                medical_context["polyp_detection"] = {
                    "count": detector.get("count", 0),
                    "objects": detector.get("objects", [])[:3],  # Top 3
                    "analysis": detector.get("analysis", ""),
                    "visualization_available": detector.get("visualization_available", False)
                }
        
        # Add classification context
        if "modality_result" in state:
            modality = state["modality_result"]
            if modality.get("success", False):
                medical_context["imaging_technique"] = {
                    "type": modality.get("class_name", "unknown"),
                    "confidence": modality.get("confidence", 0),
                    "clinical_advantages": modality.get("clinical_advantages", [])
                }
        
        if "region_result" in state:
            region = state["region_result"]
            if region.get("success", False):
                medical_context["anatomical_region"] = {
                    "location": region.get("class_name", "unknown"),
                    "confidence": region.get("confidence", 0),
                    "significance": region.get("anatomical_significance", ""),
                    "pathology_risk": region.get("pathology_risk", {})
                }
        
        return {
            "image_path": state.get("image_path", ""),
            "query": state.get("query", ""),
            "medical_context": medical_context,
            "is_text_only": state.get("is_text_only", False),
            "user_type": state.get("user_type", "patient"),
            "language_preference": state.get("language_preference", "vi")
        }

    def _analyze_query_characteristics(self, query: str, medical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze query to determine optimal strategy."""
        if not query:
            return {"strategy": "direct_llava_consultation", "complexity": "simple"}
        
        query_lower = query.lower()
        
        # Query type analysis
        is_greeting = any(word in query_lower for word in ["hello", "hi", "xin chào", "chào"])
        is_explanation = any(word in query_lower for word in ["explain", "why", "how", "what", "giải thích", "tại sao"])
        is_diagnosis = any(word in query_lower for word in ["diagnose", "disease", "condition", "bệnh", "chẩn đoán"])
        is_comparison = any(word in query_lower for word in ["compare", "difference", "vs", "so sánh", "khác"])
        is_recommendation = any(word in query_lower for word in ["recommend", "should", "advice", "khuyên", "nên"])
        
        # Context richness
        has_detection_context = bool(medical_context.get("polyp_detection"))
        has_classification_context = bool(medical_context.get("imaging_technique") or medical_context.get("anatomical_region"))
        context_richness = "rich" if (has_detection_context and has_classification_context) else "moderate" if (has_detection_context or has_classification_context) else "minimal"
        
        # Strategy selection logic
        if is_greeting:
            strategy = "direct_llava_consultation"
            complexity = "simple"
        elif is_comparison or (is_explanation and context_richness == "rich"):
            strategy = "comparative_analysis"
            complexity = "complex"
        elif (is_diagnosis or is_recommendation) and context_richness in ["moderate", "rich"]:
            strategy = "context_enhanced_analysis"
            complexity = "moderate"
        elif len(query.split()) > 20 or (is_explanation and is_diagnosis):
            strategy = "multi_step_reasoning"
            complexity = "complex"
        elif context_richness == "rich":
            strategy = "consultation_synthesis"
            complexity = "moderate"
        else:
            strategy = "direct_llava_consultation"
            complexity = "simple"
        
        return {
            "strategy": strategy,
            "complexity": complexity,
            "query_type": {
                "is_greeting": is_greeting,
                "is_explanation": is_explanation,
                "is_diagnosis": is_diagnosis,
                "is_comparison": is_comparison,
                "is_recommendation": is_recommendation
            },
            "context_richness": context_richness,
            "estimated_steps": 1 if complexity == "simple" else 2 if complexity == "moderate" else 3
        }

    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for reasoning."""
        image_path = task_input.get("image_path", "")
        query = task_input.get("query", "")
        medical_context = task_input.get("medical_context", {})
        is_text_only = task_input.get("is_text_only", False)
        user_type = task_input.get("user_type", "patient")
        
        # Analyze query characteristics
        analysis = self._analyze_query_characteristics(query, medical_context)
        
        # Build context summary
        context_parts = []
        if "polyp_detection" in medical_context:
            det = medical_context["polyp_detection"]
            context_parts.append(f"- Detection: {det.get('count', 0)} polyp(s) found")
        
        if "imaging_technique" in medical_context:
            img = medical_context["imaging_technique"]
            context_parts.append(f"- Imaging: {img.get('type', 'unknown')} technique ({img.get('confidence', 0):.1%} confidence)")
        
        if "anatomical_region" in medical_context:
            reg = medical_context["anatomical_region"]
            context_parts.append(f"- Location: {reg.get('location', 'unknown')} region ({reg.get('confidence', 0):.1%} confidence)")
        
        context_summary = "\n".join(context_parts) if context_parts else "No previous medical analysis available"
        
        # Mode-specific instructions
        mode_instruction = "TEXT-ONLY medical consultation (no image analysis)" if is_text_only else "IMAGE + TEXT medical consultation"
        
        return f"""**MEDICAL CONSULTATION TASK**

Mode: {mode_instruction}
Image: {image_path if not is_text_only else "Not provided (text-only mode)"}
User Query: "{query if query else 'Provide medical consultation'}"
User Type: {user_type}

Previous Medical Analysis:
{context_summary}

Query Analysis:
- Recommended Strategy: {analysis['strategy']}
- Complexity Level: {analysis['complexity']}
- Context Richness: {analysis['context_richness']}
- Estimated Steps: {analysis['estimated_steps']}

STRATEGY SELECTION:
Based on the analysis above, choose and implement the optimal strategy from your available options.
You have full freedom to adapt your approach, but aim for efficiency while ensuring comprehensive coverage.

Remember:
- Provide medical accuracy with appropriate disclaimers
- Adapt complexity to user type ({user_type})
- Use previous analysis context to enhance your consultation
- Stop when you have a confident, comprehensive answer

Begin your reasoning:"""

    def _check_termination_criteria(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        """Check if termination criteria are met."""
        if not react_result.get("success"):
            return {
                "should_terminate": True,
                "reason": "react_execution_failed",
                "confidence": 0.0
            }
        
        answer = react_result.get("answer", "")
        
        # Safety concern detection
        safety_keywords = [
            "emergency", "urgent", "immediate", "hospital", "doctor", "specialist",
            "khẩn cấp", "cấp cứu", "ngay lập tức", "bác sĩ", "chuyên khoa"
        ]
        
        has_safety_concern = any(keyword in answer.lower() for keyword in safety_keywords)
        
        # Confidence assessment based on answer quality
        confidence = 0.5  # Base confidence
        
        if len(answer) > 100:
            confidence += 0.2  # Substantial answer
        if "confidence" in answer.lower() or "recommend" in answer.lower():
            confidence += 0.1  # Shows medical reasoning
        if has_safety_concern:
            confidence += 0.2  # Appropriate safety awareness
        if any(phrase in answer.lower() for phrase in ["based on", "analysis shows", "findings indicate"]):
            confidence += 0.1  # Evidence-based reasoning
        
        confidence = min(1.0, confidence)
        
        # Termination decision
        should_terminate = (
            confidence >= 0.7 or  # High confidence answer
            has_safety_concern or  # Safety concern identified
            len(self.react_history) >= self.max_iterations  # Max iterations reached
        )
        
        termination_reason = "confident_answer" if confidence >= 0.7 else \
                           "safety_concern" if has_safety_concern else \
                           "max_iterations" if len(self.react_history) >= self.max_iterations else \
                           "continue"
        
        return {
            "should_terminate": should_terminate,
            "reason": termination_reason,
            "confidence": confidence,
            "has_safety_concern": has_safety_concern
        }

    def _run_react_loop(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override ReAct loop for approach with intelligent termination.
        """
        self.react_history = []
        
        for i in range(1, self.max_iterations + 1):
            # Create messages for this iteration
            messages = self._create_react_messages(task_input)
            print(messages)
            # Get LLM response
            response = self.llm.invoke(messages)
            resp_content = response.content
            print(resp_content)
            
            # Parse response
            thought, action, action_input = self._parse_llm_response(resp_content)
            
            if not thought or not action:
                self.logger.warning(f"Iteration {i}: Invalid LLM response, continuing...")
                continue
            
            # Create step
            step = ReActStep(
                thought=thought,
                thought_type=ThoughtType.REASONING,
                action=action,
                action_input=action_input
            )
            
            # Check for final answer
            if action.lower() in ["final answer", "final_answer"]:
                step.thought_type = ThoughtType.CONCLUSION
                self.react_history.append(step)
                
                answer = action_input.get("answer") if action_input else thought
                result = {
                    "success": True,
                    "answer": answer,
                    "history": self._serialize_history(),
                    "iterations_used": i,
                    "termination_reason": "final_answer_provided"
                }
                
                # Check termination criteria for quality assessment
                termination_check = self._check_termination_criteria(result)
                result.update(termination_check)
                
                return result
            
            # Execute tool
            if action in [tool.name for tool in self.tools]:
                observation = self._execute_tool(action, action_input or {})
                step.observation = observation
                step.thought_type = ThoughtType.OBSERVATION
                self.react_history.append(step)
                
                # After tool execution, check if we should continue or can terminate
                temp_result = {
                    "success": True,
                    "answer": f"Completed {action} tool execution",
                    "history": self._serialize_history()
                }
                
                termination_check = self._check_termination_criteria(temp_result)
                
                # Continue unless we have a critical issue
                if termination_check["reason"] == "safety_concern":
                    return {
                        "success": True,
                        "answer": "Safety concern identified. Please consult with a medical professional immediately.",
                        "history": self._serialize_history(),
                        "iterations_used": i,
                        "termination_reason": "safety_concern_detected"
                    }
            else:
                step.observation = f"Error: Unknown action '{action}'"
                self.react_history.append(step)
        
        # Max iterations reached - provide best effort result
        return {
            "success": True,  # Still successful, just reached limit
            "answer": "Analysis completed with available information. Recommend consulting healthcare professional for comprehensive evaluation.",
            "history": self._serialize_history(),
            "iterations_used": self.max_iterations,
            "termination_reason": "max_iterations_reached"
        }

    def _perform_image_synthesis(self, react_result: Dict[str, Any], task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform synthesis WITH image viewing to validate tool results.
        This is the key difference from Classifier agents.
        """
        try:
            image_path = task_input.get("image_path", "")
            query = task_input.get("query", "")
            is_text_only = task_input.get("is_text_only", False)
            
            # Extract LLaVA results from history
            llava_responses = []
            for step in self.react_history:
                if step.observation and "llava_vqa" in str(step.action):
                    try:
                        obs_data = json.loads(step.observation)
                        if obs_data.get("success", False):
                            llava_responses.append({
                                "answer": obs_data.get("answer", ""),
                                "query_type": obs_data.get("query_type", "unknown"),
                                "has_image": obs_data.get("has_image", False)
                            })
                    except json.JSONDecodeError:
                        continue
            
            # Prepare synthesis prompt
            synthesis_prompt = f"""**VQA SYNTHESIS WITH IMAGE VALIDATION**

Original Query: "{query}"
Mode: {"Text-only consultation" if is_text_only else "Image-based analysis"}

LLaVA Tool Results:
{json.dumps(llava_responses, indent=2)}

ReAct Final Answer: {react_result.get("answer", "")}

SYNTHESIS TASK:
1. Review the tool results above
2. {"Since this is text-only, validate the medical reasoning" if is_text_only else "Look at the provided image to validate the analysis"}
3. Assess consistency between tool results and final reasoning
4. Provide enhanced final consultation with:
   - Validation of tool accuracy
   - Comprehensive medical assessment
   - Appropriate recommendations and disclaimers
   - Professional tone suitable for patient consultation

Enhanced consultation:"""
            
            # Create synthesis message
            if is_text_only or not image_path or not os.path.exists(image_path):
                # Text-only synthesis
                messages = [
                    SystemMessage(content=self._get_agent_description()),
                    HumanMessage(content=synthesis_prompt)
                ]
            else:
                # Image + text synthesis (KEY DIFFERENCE from Classifier)
                messages = [
                    SystemMessage(content=self._get_agent_description()),
                    HumanMessage(
                        content=[
                            {"type": "text", "text": synthesis_prompt},
                            {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
                        ]
                    )
                ]
            
            # Get enhanced synthesis
            synthesis_response = self.llm.invoke(messages)
            enhanced_answer = synthesis_response.content.strip()
            
            return {
                "success": True,
                "enhanced_answer": enhanced_answer,
                "original_answer": react_result.get("answer", ""),
                "tool_results": llava_responses,
                "synthesis_method": "text_only" if is_text_only else "image_enhanced",
                "validation_performed": True
            }
            
        except Exception as e:
            self.logger.error(f"Image synthesis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "enhanced_answer": react_result.get("answer", ""),
                "synthesis_method": "fallback"
            }

    def _format_agent_result(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format VQA result with image synthesis."""
        if not react_result.get("success", False):
            return {
                "vqa_result": {
                    "success": False,
                    "error": react_result.get("error", "VQA failed"),
                    "termination_reason": react_result.get("termination_reason", "unknown")
                }
            }
        
        # Get task input for synthesis
        task_input = getattr(self, '_current_task_input', {})
        
        # Perform image synthesis validation
        synthesis_result = self._perform_image_synthesis(react_result, task_input)
        
        # Determine strategy used
        iterations = react_result.get("iterations_used", len(self.react_history))
        strategy_used = "direct_consultation" if iterations <= 2 else \
                      "enhanced_analysis" if iterations <= 4 else \
                      "complex_reasoning"
        
        # Build comprehensive result
        vqa_result = {
            "success": True,
            "strategy_used": strategy_used,
            "answer": synthesis_result.get("enhanced_answer", react_result.get("answer", "")),
            "original_llm_answer": react_result.get("answer", ""),
            "analysis": f"VQA consultation completed using {strategy_used} strategy",
            "confidence": react_result.get("confidence", 0.8),
            "iterations_used": iterations,
            "termination_reason": react_result.get("termination_reason", "completed"),
            "synthesis_validation": {
                "method": synthesis_result.get("synthesis_method", "unknown"),
                "performed": synthesis_result.get("validation_performed", False),
                "tool_results_count": len(synthesis_result.get("tool_results", []))
            },
            "safety_assessment": {
                "concern_detected": react_result.get("has_safety_concern", False),
                "requires_human_expert": react_result.get("termination_reason") == "safety_concern"
            }
        }
        
        # Add tool results summary
        if synthesis_result.get("tool_results"):
            vqa_result["llava_executions"] = len(synthesis_result["tool_results"])
            vqa_result["query_types_handled"] = [
                result.get("query_type", "unknown") 
                for result in synthesis_result["tool_results"]
            ]
        
        return {"vqa_result": vqa_result}

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Override process to store task input for synthesis."""
        # Store task input for later synthesis use
        task_input = self._extract_task_input(state)
        self._current_task_input = task_input
        
        # Call parent process
        return super().process(state)

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
    }
}
```
"""

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
        
        # LLaVA succeeded - use result directly without synthesis
        self.logger.info("[VQA] LLaVA succeeded - returning result directly")
        
        # Create result directly from LLaVA response
        vqa_result = {
            "vqa_result": {
                "success": True,
                "answer": results['llava_vqa'].get('answer', ''),
                "query_type": "image_based" if task_input.get("image_path") else "text_only",
                "llava_response": results['llava_vqa'].get('answer', '')
            }
        }
        
        # Log the final result
        self.logger.info(f"[VQA] Final result: {json.dumps(vqa_result, indent=2)}")
        
        # Return agent result
        return {**state, **vqa_result}
