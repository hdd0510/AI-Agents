#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VQA AGENT - SIMPLE & WORKING VERSION
====================================
Fix parsing issues và thêm synthesis function
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
    """VQA Agent với parsing fix và synthesis function."""
    
    def __init__(self, model_path: str, llm_model: str = "gpt-4o-mini", device: str = "cuda"):
        self.model_path = model_path
        super().__init__(name="VQA Agent", llm_model=llm_model, device=device)
        
        # Simple config
        self.max_iterations = 3  # Reduced from 6
        
        self.llava_tool = None
    
    def _register_tools(self) -> List[BaseTool]:
        """Register VQA tools."""
        self.llava_tool = LLaVATool(
            model_path=self.model_path,
            device=self.device
        )
        return [self.llava_tool]
    
    def _get_agent_description(self) -> str:
        """Agent description."""
        return "I am a medical consultation AI that provides accurate healthcare advice using available tools."

    def _get_system_prompt(self) -> str:
        """Clear system prompt với correct format."""
        return f"""You are a medical consultation AI with expert knowledge in medical imaging and pathology. You have ability of 
        reasoning and handle questions, so just use tools to provide more accurate medical advice to your answer

RESPONSE FORMAT (REQUIRED):
Thought: [your reasoning]
Action: [tool_name or Final Answer]
Action Input: {{"param": "value"}}

EXAMPLES:
Thought: I need to use LLaVA to analyze this medical query
Action: llava_vqa
Action Input: {{"query": "user question", "medical_context": {{}}, "image_path": "path"}}

Thought: I have the medical analysis and can provide final consultation
Action: Final Answer
Action Input: {{"answer": "professional medical response"}}

Available tools: {self.tool_descriptions}

Always use the exact format above. Start with "Thought:"."""

    def _parse_llm_response(self, response: str) -> tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Simple and robust parsing."""
        response = response.strip()
        
        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|$)", response, re.DOTALL | re.IGNORECASE)
        thought = thought_match.group(1).strip() if thought_match else None
        
        # Extract Action
        action_match = re.search(r"Action:\s*(.*?)(?=Action Input:|$)", response, re.DOTALL | re.IGNORECASE)
        action = action_match.group(1).strip() if action_match else None
        
        # Extract Action Input
        input_match = re.search(r"Action Input:\s*(\{.*?\})", response, re.DOTALL)
        action_input = None
        if input_match:
            try:
                action_input = json.loads(input_match.group(1))
            except json.JSONDecodeError:
                # Fallback: create basic structure
                if action and action.lower() in ["final answer", "final_answer"]:
                    # Try to extract answer content
                    answer_content = input_match.group(1).replace("{", "").replace("}", "").replace('"answer":', "").replace('"', "").strip()
                    action_input = {"answer": answer_content}
                else:
                    # For tool calls, use basic query
                    action_input = {"query": "Medical consultation request"}
        
        return thought, action, action_input

    def _run_react_loop(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Simple ReAct loop với fallback."""
        self.react_history = []
        
        for i in range(1, self.max_iterations + 1):
            try:
                # Get LLM response
                messages = self._create_react_messages(task_input)
                response = self.llm.invoke(messages)
                resp_content = response.content
                # Parse response
                thought, action, action_input = self._parse_llm_response(resp_content)
                
                if not thought or not action:
                    # If parsing fails, try direct tool call
                    if i == 1:  # First attempt, try direct call
                        return self._direct_tool_call(task_input)
                    else:
                        continue
                
                # Create step
                step = ReActStep(
                    thought=thought,
                    thought_type=ThoughtType.REASONING,
                    action=action,
                    action_input=action_input
                )
                
                # Handle final answer
                if action.lower() in ["final answer", "final_answer"]:
                    step.thought_type = ThoughtType.CONCLUSION
                    self.react_history.append(step)
                    
                    answer = action_input.get("answer") if action_input else thought
                    return {
                        "success": True,
                        "answer": answer,
                        "iterations_used": i,
                        "termination_reason": "final_answer"
                    }
                
                # Execute tool
                if action in [tool.name for tool in self.tools]:
                    observation = self._execute_tool(action, action_input or {})
                    step.observation = observation
                    step.thought_type = ThoughtType.OBSERVATION
                    self.react_history.append(step)
                    
                    # After tool execution, do synthesis
                    if i >= 1:  # At least 1 iteration done
                        synthesis_result = self._synthesis_step(task_input)
                        if synthesis_result.get("success"):
                            return synthesis_result
                else:
                    step.observation = f"Unknown action: {action}"
                    self.react_history.append(step)
                    
            except Exception as e:
                self.logger.error(f"Iteration {i} failed: {str(e)}")
                continue
        
        # Max iterations reached, try synthesis
        return self._synthesis_step(task_input)
    
    def _direct_tool_call(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Direct tool call fallback."""
        try:
            query = task_input.get("query", "Medical consultation")
            image_path = task_input.get("image_path", "")
            medical_context = task_input.get("medical_context", {})
            
            params = {
                "query": query,
                "medical_context": medical_context
            }
            
            if image_path and os.path.exists(image_path):
                params["image_path"] = image_path
            
            result = self.llava_tool._run(**params)
            
            if result.get("success"):
                return {
                    "success": True,
                    "answer": result.get("answer", ""),
                    "iterations_used": 0,
                    "termination_reason": "direct_call"
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Direct call failed"),
                    "iterations_used": 0
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Direct call error: {str(e)}",
                "iterations_used": 0
            }
    
    def _synthesis_step(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        SYNTHESIS FUNCTION - LLM nhìn vào kết quả tools và đưa ra kết quả cuối
        """
        try:
            # Collect tool results from history
            tool_results = []
            for step in self.react_history:
                if step.observation and step.action in [tool.name for tool in self.tools]:
                    try:
                        obs_data = json.loads(step.observation)
                        tool_results.append({
                            "tool": step.action,
                            "result": obs_data,
                            "thought": step.thought
                        })
                    except json.JSONDecodeError:
                        tool_results.append({
                            "tool": step.action,
                            "result": {"raw_output": step.observation},
                            "thought": step.thought
                        })
            
            if not tool_results:
                return {
                    "success": False,
                    "error": "No tool results to synthesize",
                    "iterations_used": len(self.react_history)
                }
            
            # Create synthesis prompt
            synthesis_prompt = self._create_synthesis_prompt(task_input, tool_results)
            
            # Get synthesis from LLM
            synthesis_messages = [
                SystemMessage(content="You are a medical expert providing final consultation synthesis."),
                HumanMessage(content=synthesis_prompt)
            ]
            
            synthesis_response = self.llm.invoke(synthesis_messages)
            final_answer = synthesis_response.content.strip()
            
            return {
                "success": True,
                "answer": final_answer,
                "iterations_used": len(self.react_history),
                "termination_reason": "synthesis_complete",
                "tool_results_count": len(tool_results)
            }
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {str(e)}")
            
            # Fallback: return best tool result
            for step in reversed(self.react_history):
                if step.observation:
                    try:
                        obs_data = json.loads(step.observation)
                        if obs_data.get("success") and obs_data.get("answer"):
                            return {
                                "success": True,
                                "answer": obs_data["answer"],
                                "iterations_used": len(self.react_history),
                                "termination_reason": "fallback_from_synthesis_error"
                            }
                    except:
                        continue
            
            return {
                "success": False,
                "error": f"Synthesis error: {str(e)}",
                "iterations_used": len(self.react_history)
            }
    
    def _create_synthesis_prompt(self, task_input: Dict[str, Any], tool_results: List[Dict[str, Any]]) -> str:
        """Create synthesis prompt for final answer."""
        query = task_input.get("query", "")
        has_image = bool(task_input.get("image_path") and os.path.exists(task_input.get("image_path", "")))
        
        # Format tool results
        results_text = ""
        for i, result in enumerate(tool_results, 1):
            tool_name = result["tool"]
            tool_data = result["result"]
            
            results_text += f"\n{i}. Tool: {tool_name}\n"
            results_text += f"   Thought: {result['thought']}\n"
            
            if tool_data.get("success"):
                if "answer" in tool_data:
                    results_text += f"   Answer: {tool_data['answer'][:300]}...\n"
                if "query_type" in tool_data:
                    results_text += f"   Type: {tool_data['query_type']}\n"
            else:
                results_text += f"   Error: {tool_data.get('error', 'Unknown error')}\n"
        
        return f"""**MEDICAL CONSULTATION SYNTHESIS**\n\nOriginal Query: \"{query}\"\nHas Image: {has_image}\n\nTool Execution Results:\n{results_text}\n\n**YOUR TASK:**\nAnalyze the tool results above and provide a comprehensive final medical consultation response.\n\n**Requirements:**\n1. Synthesize information from all successful tool executions\n2. Provide clear, professional medical advice\n4. Recommend next steps or follow-up if needed\n5. Be concise but thorough\n\n**Format your response as a complete medical consultation answer.**"""

    def _format_agent_result(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format VQA result."""
        if not react_result.get("success", False):
            return {
                "vqa_result": {
                    "success": False,
                    "error": react_result.get("error", "VQA processing failed"),
                    "approach": "llava"
                }
            }
        
        # Extract answer and confidence
        answer = react_result.get("answer", "")
        confidence = react_result.get("confidence", 0.0)
        
        # Check if RAG information was used
        used_rag = False
        if hasattr(self, 'react_history'):
            for step in self.react_history:
                if step.observation and "rag_result" in step.observation:
                    used_rag = True
                    break
        
        # Create final result
        vqa_result = {
            "success": True,
            "answer": answer,
            "confidence": confidence,
            "approach": "llava",
            "used_rag": used_rag
        }
        
        return {"vqa_result": vqa_result}

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
        """Extract task input."""
        medical_context = state.get("medical_context")
        medical_context = medical_context.copy() if medical_context else {}
        
        # Add context from other agents
        if "detector_result" in state:
            detector = state["detector_result"]
            if detector.get("success", False):
                medical_context["polyp_detection"] = {
                    "count": detector.get("count", 0),
                    "objects": detector.get("objects", [])
                }
        
        if "modality_result" in state:
            modality = state["modality_result"]
            if modality.get("success", False):
                medical_context["imaging_technique"] = {
                    "type": modality.get("class_name", "unknown"),
                    "confidence": modality.get("confidence", 0)
                }
        
        if "region_result" in state:
            region = state["region_result"]
            if region.get("success", False):
                medical_context["anatomical_region"] = {
                    "location": region.get("class_name", "unknown"),
                    "confidence": region.get("confidence", 0)
                }
        
        # Get RAG results if available
        rag_result = state.get("rag_result", {})
        
        return {
            "image_path": state.get("image_path", ""),
            "query": state.get("query", ""),
            "medical_context": medical_context,
            "is_text_only": state.get("is_text_only", False),
            "rag_result": rag_result
        }

    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for VQA processing."""
        query = task_input.get("query", "")
        image_path = task_input.get("image_path", "")
        medical_context = task_input.get("medical_context", {})
        rag_result = task_input.get("rag_result", {})
        
        # Build context string
        context_parts = []
        if medical_context:
            if "polyp_findings" in medical_context:
                findings = medical_context["polyp_findings"]
                context_parts.append(f"- Polyp detection: {findings['count']} polyp(s) found")
            if "imaging_type" in medical_context:
                context_parts.append(f"- Imaging type: {medical_context['imaging_type']}")
            if "anatomical_region" in medical_context:
                context_parts.append(f"- Anatomical region: {medical_context['anatomical_region']}")
        
        context_str = "\n".join(context_parts) if context_parts else "No additional context"
        
        # Add RAG information if available
        rag_info = ""
        if rag_result and rag_result.get("success", False):
            if "vqa_summary" in rag_result and rag_result["vqa_summary"]:
                rag_info = f"\nRelevant information from documents:\n{rag_result['vqa_summary']}"
        
        return f"""**VISUAL QUESTION ANSWERING TASK**

User Query: "{query}"

Medical Context:
{context_str}
{rag_info}

Your task:
1. Analyze the image and provide a detailed answer
2. If document information is provided, incorporate it into your answer
3. Ensure your answer is comprehensive and well-supported
4. Use medical terminology appropriately

Begin with image analysis:"""
