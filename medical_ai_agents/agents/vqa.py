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
        reasoning and handle questions with tools (provide image path if available) to provide more accurate medical advice to your answer. You should reason and act step by step. If results from tools are not good, you should try again.

RESPONSE FORMAT (REQUIRED):
Thought: [your reasoning]
Action: [tool_name or Final Answer]
Action Input: {{"param": "value"}}

EXAMPLES :
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
        # Process the query to extract only useful information
        if "query" in task_input:
            task_input["query"] = self._process_user_query(task_input["query"])
        # Store current task input for use in _execute_tool and _synthesis_step
        self.current_task_input = task_input
        
        self.logger.info("="*50)
        self.logger.info("STARTING VQA REACT LOOP")
        self.logger.info(f"Query: '{task_input.get('query', '')[:100]}...'")
        self.logger.info(f"Image path: {task_input.get('image_path', 'None')}")
        self.logger.info("="*50)
        
        for i in range(1, self.max_iterations + 1):
            try:
                self.logger.info(f"VQA ITERATION {i}/{self.max_iterations}")
                
                # Get LLM response
                messages = self._create_react_messages(task_input)
                response = self.llm.invoke(messages)
                resp_content = response.content
                # Parse response
                thought, action, action_input = self._parse_llm_response(resp_content)
                
                self.logger.info(f"THOUGHT: {thought[:150]}..." if thought and len(thought) > 150 else f"THOUGHT: {thought}")
                self.logger.info(f"ACTION: {action}")
                self.logger.info(f"ACTION INPUT: {json.dumps(action_input, ensure_ascii=False)[:150]}..." if action_input else "ACTION INPUT: None")
                
                if not thought or not action:
                    # If parsing fails, try direct tool call
                    self.logger.info("Parsing failed, trying direct tool call")
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
                    self.logger.info("="*50)
                    self.logger.info("VQA FINAL ANSWER (from direct final answer)")
                    self.logger.info(f"Answer: {answer[:200]}..." if len(answer) > 200 else f"Answer: {answer}")
                    self.logger.info("="*50)
                    
                    return {
                        "success": True,
                        "answer": answer,
                        "iterations_used": i,
                        "termination_reason": "final_answer"
                    }
                
                # Execute tool
                if action in [tool.name for tool in self.tools]:
                    self.logger.info(f"Executing tool: {action}")
                    observation = self._execute_tool(action, action_input or {})
                    step.observation = observation
                    step.thought_type = ThoughtType.OBSERVATION
                    self.react_history.append(step)
                    
                    # Log observation (truncated if too long)
                    obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
                    self.logger.info(f"OBSERVATION: {obs_preview}")
                    
                    # After tool execution, do synthesis
                    if i >= 1:  # At least 1 iteration done
                        self.logger.info("Attempting synthesis after tool execution")
                        synthesis_result = self._synthesis_step(task_input)
                        if synthesis_result.get("success"):
                            return synthesis_result
                else:
                    step.observation = f"Unknown action: {action}"
                    self.react_history.append(step)
                    self.logger.info(f"Unknown action: {action}")
                    
            except Exception as e:
                self.logger.error(f"Iteration {i} failed: {str(e)}")
                continue
        
        # Max iterations reached, try synthesis
        self.logger.info("Max iterations reached, attempting final synthesis")
        return self._synthesis_step(task_input)
    
    def _direct_tool_call(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """Direct tool call fallback."""
        try:
            self.logger.info("="*50)
            self.logger.info("USING DIRECT TOOL CALL")
            
            query = task_input.get("query", "Medical consultation")
            # Process the query to extract only useful information
            query = self._process_user_query(query)
            image_path = task_input.get("image_path", "")
            medical_context = task_input.get("medical_context", {})
            
            # Log important parameters for debugging
            self.logger.info(f"VQA direct_tool_call - Query: '{query[:50]}...', Image: {image_path}")
            
            params = {
                "query": query,
                "medical_context": medical_context
            }
            
            # Explicitly check image path and only add if it exists
            if image_path and os.path.exists(image_path):
                self.logger.info(f"Adding verified image_path to LLaVA tool params: {image_path}")
                params["image_path"] = image_path
            else:
                self.logger.warning(f"Image path not added to params - Path: '{image_path}', Exists: {bool(image_path and os.path.exists(image_path))}")
            
            # Call LLaVA tool with necessary parameters
            self.logger.info("Calling LLaVA tool directly")
            result = self.llava_tool._run(**params)
            
            if result.get("success"):
                answer = result.get("answer", "")
                self.logger.info("="*50)
                self.logger.info("VQA FINAL ANSWER (from direct tool call)")
                self.logger.info(f"Answer: {answer[:200]}..." if len(answer) > 200 else f"Answer: {answer}")
                self.logger.info("="*50)
                
                return {
                    "success": True,
                    "answer": answer,
                    "iterations_used": 0,
                    "termination_reason": "direct_call"
                }
            else:
                self.logger.error(f"Direct tool call failed: {result.get('error', 'Unknown error')}")
                return {
                    "success": False,
                    "error": result.get("error", "Direct call failed"),
                    "iterations_used": 0
                }
                
        except Exception as e:
            self.logger.error(f"Direct call error: {str(e)}")
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
            self.logger.info("="*50)
            self.logger.info("STARTING VQA SYNTHESIS")
            
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
            
            self.logger.info(f"Collected {len(tool_results)} tool results for synthesis")
            
            if not tool_results:
                self.logger.warning("No tool results to synthesize")
                return {
                    "success": False,
                    "error": "No tool results to synthesize",
                    "iterations_used": len(self.react_history)
                }
            
            # Create synthesis prompt
            synthesis_prompt = self._create_synthesis_prompt(task_input, tool_results)
            
            # Check if we have images to include in synthesis
            image_path = task_input.get("image_path", "")
            has_original_image = bool(image_path and os.path.exists(image_path))
            
            # Check for visualization from detector
            detector_viz_base64 = None
            medical_context = task_input.get("medical_context", {})
            if "detector_result" in getattr(self, 'current_state', {}):
                detector_result = self.current_state.get("detector_result", {})
                if detector_result.get("visualization_base64"):
                    detector_viz_base64 = detector_result["visualization_base64"]
                    self.logger.info("Found detector visualization with bounding boxes")
            
            # Get synthesis from LLM with images if available
            synthesis_messages = [
                SystemMessage(content="You are a medical expert providing final consultation synthesis.")
            ]
            
            # Prepare the message content
            message_content = []
            message_content.append({"type": "text", "text": synthesis_prompt})
            
            # Add images to message content if available
            images_included = False
            
            if has_original_image or detector_viz_base64:
                import base64
                from PIL import Image
                from io import BytesIO
                
                def image_to_base64(img_path):
                    with Image.open(img_path) as img:
                        buffered = BytesIO()
                        img.save(buffered, format="PNG")
                        return base64.b64encode(buffered.getvalue()).decode()
                
                # Add detector visualization if available
                if detector_viz_base64:
                    try:
                        message_content.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{detector_viz_base64}"}}
                        )
                        self.logger.info("Successfully included detector visualization in synthesis message")
                        images_included = True
                    except Exception as e:
                        self.logger.error(f"Failed to include detector visualization in synthesis: {str(e)}")

                # Add original image if available
                elif has_original_image:
                    try:
                        img_b64 = image_to_base64(image_path)
                        message_content.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        )
                        self.logger.info("Successfully included original image in synthesis message")
                        images_included = True
                    except Exception as e:
                        self.logger.error(f"Failed to include original image in synthesis: {str(e)}")
            
            # Create the final message
            if images_included:
                synthesis_messages.append(HumanMessage(content=message_content))
            else:
                synthesis_messages.append(HumanMessage(content=synthesis_prompt))
            
            self.logger.info("Generating synthesis response")
            synthesis_response = self.llm.invoke(synthesis_messages)
            final_answer = synthesis_response.content.strip()
            
            self.logger.info("="*50)
            self.logger.info("VQA FINAL ANSWER (from synthesis)")
            self.logger.info(f"Answer: {final_answer[:200]}..." if len(final_answer) > 200 else f"Answer: {final_answer}")
            self.logger.info("="*50)
            
            return {
                "success": True,
                "answer": final_answer,
                "iterations_used": len(self.react_history),
                "termination_reason": "synthesis_complete",
                "tool_results_count": len(tool_results),
                "included_original_image": has_original_image,
                "included_detector_visualization": bool(detector_viz_base64)
            }
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {str(e)}")
            
            # Fallback: return best tool result
            for step in reversed(self.react_history):
                if step.observation:
                    try:
                        obs_data = json.loads(step.observation)
                        if obs_data.get("success") and obs_data.get("answer"):
                            answer = obs_data["answer"]
                            self.logger.info("="*50)
                            self.logger.info("VQA FINAL ANSWER (from fallback)")
                            self.logger.info(f"Answer: {answer[:200]}..." if len(answer) > 200 else f"Answer: {answer}")
                            self.logger.info("="*50)
                            
                            return {
                                "success": True,
                                "answer": answer,
                                "iterations_used": len(self.react_history),
                                "termination_reason": "fallback_from_synthesis_error"
                            }
                    except:
                        continue
            
            self.logger.error("Synthesis failed with no fallback available")
            return {
                "success": False,
                "error": f"Synthesis error: {str(e)}",
                "iterations_used": len(self.react_history)
            }
    
    def _create_synthesis_prompt(self, task_input: Dict[str, Any], tool_results: List[Dict[str, Any]]) -> str:
        """Create synthesis prompt for final answer."""
        query = task_input.get("query", "")
        # Process the query to extract only useful information
        query = self._process_user_query(query)
        has_image = bool(task_input.get("image_path") and os.path.exists(task_input.get("image_path", "")))
        
        # Check for detector visualization
        has_detector_viz = False
        detector_info = ""
        if hasattr(self, 'current_state') and "detector_result" in self.current_state:
            detector_result = self.current_state.get("detector_result", {})
            if detector_result.get("visualization_base64"):
                has_detector_viz = True
                polyp_count = detector_result.get("count", 0)
                detector_info = f"\n\nDetector found {polyp_count} polyp(s) in the image."
                if polyp_count > 0:
                    detector_info += " The second image shows the visualization with bounding boxes around detected polyps."
        
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
        
        image_guidance = ""
        if has_image:
            image_guidance = """
I am showing you the original medical image for direct analysis. Please examine the image carefully and include visual observations in your synthesis."""
            
            if has_detector_viz:
                image_guidance += """

I am also showing you a second image with the detector's visualization containing bounding boxes around detected polyps. Please analyze both images and compare them in your assessment."""
        
        return f"""**MEDICAL CONSULTATION SYNTHESIS**

Original Query: "{query}"
Has Image: {has_image}{detector_info}{image_guidance}

Tool Execution Results:
{results_text}

**YOUR TASK:**
Analyze the tool results above and provide a comprehensive final medical consultation response.

**Requirements:**
1. Synthesize information from all successful tool executions
2. Provide clear, professional medical advice
3. If you can see the image(s), include your direct observations
4. If you can see both the original image and detector visualization, compare them and verify the detector's findings
5. Recommend next steps or follow-up if needed
6. Be concise but thorough

**Format your response as a complete medical consultation answer.**"""

    def _format_agent_result(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format VQA result."""
        if not react_result.get("success", False):
            self.logger.error("="*50)
            self.logger.error("VQA PROCESSING FAILED")
            self.logger.error(f"Error: {react_result.get('error', 'Unknown error')}")
            self.logger.error("="*50)
            
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
        
        self.logger.info("="*50)
        self.logger.info("FINAL VQA RESULT FORMATTED")
        self.logger.info(f"Success: {vqa_result['success']}")
        self.logger.info(f"Approach: {vqa_result['approach']}")
        self.logger.info(f"Used RAG: {vqa_result['used_rag']}")
        self.logger.info(f"Answer: {answer[:]}")
        self.logger.info(f"Termination reason: {react_result.get('termination_reason', 'unknown')}")
        self.logger.info(f"Iterations used: {react_result.get('iterations_used', 0)}")
        self.logger.info("="*50)
        
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
        # Store the current state for later access to detector result
        self.current_state = state
        
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
        
        # Log the image path information to help with debugging
        image_path = state.get("image_path", "")
        if image_path:
            self.logger.info(f"VQA received image path: {image_path}")
            if os.path.exists(image_path):
                self.logger.info(f"Image exists and will be used for VQA analysis")
            else:
                self.logger.warning(f"Image path provided but file does not exist: {image_path}")
        else:
            self.logger.info("No image path received, will process as text-only query")
        
        # If there are polyp detection results, verify that we maintain the image path
        if "detector_result" in state and state.get("image_path") and not state.get("is_text_only", False):
            self.logger.info(f"Multi-task workflow: using image from detector task: {state.get('image_path')}")
            # Force is_text_only to False to ensure image is used
            state["is_text_only"] = False
        
        # Get the query and process it to extract only useful information
        query = state.get("query", "")
        processed_query = self._process_user_query(query)
        
        return {
            "image_path": image_path,
            "query": processed_query,
            "medical_context": medical_context,
            "is_text_only": state.get("is_text_only", False),
            "rag_result": rag_result
        }

    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for VQA processing."""
        query = task_input.get("query", "")
        # Process the query to extract only useful information
        query = self._process_user_query(query)
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
        
        return f"""**MEDICAL AI VISUAL QUESTION ANSWERING TASK**

User Query: "{query}"

Medical Context:
{context_str}
{rag_info}

Your task as a Medical AI Assistant:
1. Analyze the medical image and provide a detailed professional assessment
2. If document information is provided, incorporate it into your medical analysis
3. Ensure your answer is comprehensive and well-supported by clinical evidence
4. Use appropriate medical terminology while maintaining clarity
5. Always identify yourself as a Medical AI Assistant specializing in healthcare

Begin with medical image analysis:"""

    def _execute_tool(self, action: str, action_input: Dict[str, Any]) -> str:
        """Execute tool with improved image handling."""
        # Process query if this is the LLaVA tool
        if action == "llava_vqa" and "query" in action_input:
            action_input["query"] = self._process_user_query(action_input["query"])
            
        # If the tool is llava_vqa, ensure image_path is included if available
        if action == "llava_vqa" and hasattr(self, 'current_task_input'):
            # Check if image_path exists in current task input but not in action input
            image_path = self.current_task_input.get("image_path")
            if image_path and os.path.exists(image_path) and "image_path" not in action_input:
                self.logger.info(f"Adding missing image_path to llava_vqa tool call: {image_path}")
                action_input["image_path"] = image_path
        
        # Call the base class implementation
        tool = next((t for t in self.tools if t.name == action), None)
        if not tool:
            return f"Error: tool '{action}' not found."
        try:
            result = tool(**action_input)
            return json.dumps(result) if isinstance(result, dict) else str(result)
        except Exception as e:
            self.logger.error(f"Tool execution error: {str(e)}")
            return f"Error executing {action}: {str(e)}"

    def _process_user_query(self, query: str) -> str:
        """
        Process user query to extract only useful medical information.
        
        This function removes unnecessary conversational elements, greetings,
        and focuses on extracting the core medical question or request.
        
        Args:
            query: The original user query
            
        Returns:
            Processed query containing only useful medical information
        """
        try:
            # Use the LLM to extract only the relevant medical information
            extraction_messages = [
                SystemMessage(content="""You are a medical query processor. Your task is to extract ONLY the 
                medically relevant information from user queries. Remove greetings, pleasantries, 
                conversational elements, and any non-medical content. Focus on symptoms, medical conditions, 
                diagnostic questions, or specific medical information requests. Return ONLY the 
                processed medical query without any explanation or additional text."""),
                HumanMessage(content=f"Original query: {query}\n\nExtract only the medically relevant information:")
            ]
            
            extraction_response = self.llm.invoke(extraction_messages)
            processed_query = extraction_response.content.strip()
            
            # Log the query transformation
            self.logger.info(f"Query transformation - Original: '{query[:50]}...', Processed: '{processed_query[:50]}...'")
            
            # If the processed query is empty or significantly shorter than the original,
            # fall back to the original to avoid losing important information
            if not processed_query or len(processed_query) < len(query) * 0.3:
                self.logger.warning("Processed query too short, falling back to original")
                return query
                
            return processed_query
            
        except Exception as e:
            self.logger.error(f"Query processing error: {str(e)}")
            # On error, return the original query to ensure functionality
            return query
