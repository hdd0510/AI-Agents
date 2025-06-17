#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FLEXIBLE DETECTOR AGENT - APPROACH (Safety Critical)
============================================================
Dynamic workflow guided by LLM reasoning
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from medical_ai_agents.agents.base_agent import BaseAgent, ThoughtType
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.detection.yolo_tools import YOLODetectionTool
from medical_ai_agents.tools.detection.util_tools import VisualizationTool
from medical_ai_agents.agents.base_agent import ReActStep

class DetectorAgent(BaseAgent):
    """
    Detector Agent - Safety Critical
    
    FLEXIBLE WORKFLOW:
    - LLM driven detection with adaptive thresholding
    - Can adjust confidence and IoU thresholds based on initial results
    - Visualization and synthesis as needed
    """
    
    def __init__(self, model_path: str, llm_model: str = "gpt-4o-mini", device: str = "cuda"):
        self.model_path = model_path
        super().__init__(name="Detector Agent", llm_model=llm_model, device=device)
        
        # configuration
        self.max_iterations = 5  # Allowing more iterations for adaptive thresholding
    
    def _register_tools(self) -> List[BaseTool]:
        """Register detection tools."""
        self.detector_tool = YOLODetectionTool(
            model_path=self.model_path,
            device=self.device
        )
        self.visualize_tool = VisualizationTool()
        return [self.detector_tool, self.visualize_tool]
    
    def _get_agent_description(self) -> str:
        """detector description."""
        return """I am a SAFETY-CRITICAL polyp detection specialist using adaptive workflow.

My flexible process:
1. Use YOLO for polyp detection with adaptive confidence thresholds
2. Adjust parameters if needed to ensure thorough detection
3. Create visualizations when appropriate
4. Synthesize findings based on all available data

I can adaptively lower thresholds if initial detection yields no results."""

    def _get_system_prompt(self) -> str:
        """system prompt with flexible workflow."""
        return f"""You are a SAFETY-CRITICAL polyp detection specialist with an adaptive workflow.

ALWAYS FOLLOW THIS EXACT FORMAT:

Thought: <your reasoning>
Action: <tool_name>
Action Input: {{
  "parameter1": value1,
  "parameter2": value2,
  ...
}}

OR when you've reached a conclusion:

Thought: <your final thoughts>
Action: Final Answer
Action Input: {{
  "answer": "Your final answer here",
  "show_visualization": true/false
}}

AVAILABLE TOOLS:

1. yolo_detection - Detect polyps with adjustable parameters
   Parameters:
   - "image_path": (string, required) Path to the image file
   - "conf_thresh": (number, optional) Confidence threshold (0.1-0.5), lower values detect more potential polyps
   - "iou_thresh": (number, optional) IoU threshold (0.1-0.5), controls overlap handling

2. visualize_detections - Create visualization of detections
   Parameters:
   - "image_path": (string, required) Path to the image file
   - "detections": (array, required) List of detection objects from yolo_detection
     NOTE: If you don't provide detections, the system will automatically use the latest detection results

CORRECT WORKFLOW STEPS (follow this sequence):

1. First, use yolo_detection to detect polyps:
   Thought: I need to detect polyps in the image
   Action: yolo_detection
   Action Input: {{
     "image_path": "/path/to/image.jpg",
     "conf_thresh": 0.25,  // Standard threshold
     "iou_thresh": 0.45    // Standard threshold
   }}

2. After detection (if any polyps found), visualize them:
   Thought: I need to visualize the detected polyps
   Action: visualize_detections
   Action Input: {{
     "image_path": "/path/to/image.jpg",
     "detections": [...]   // Objects from detection result
   }}

3. Finally, provide your assessment:
   Thought: Based on detection and visualization, I can provide my final assessment
   Action: Final Answer
   Action Input: {{
     "answer": "I detected X polyps with Y confidence...",
     "show_visualization": true  // Set to true if user might want to see visualization
   }}

ERROR HANDLING:
- If no polyps found at standard threshold (0.25), try lower conf_thresh (0.1-0.2)
- Always mention location and confidence score of polyps in final answer
- Don't repeat the same action multiple times without changing parameters

CRITICAL REMINDER: After performing visualize_detections, you should always proceed to Final Answer!"""

    def initialize(self) -> bool:
        """Initialize detector agent."""
        try:
            self.initialized = True
            self.logger.info("Detector Agent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Detector: {str(e)}")
            return False

    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detection task input."""
        task_input = {
            "image_path": state.get("image_path", ""),
            "query": state.get("query", ""),
            "medical_context": state.get("medical_context", {})
            # Không dùng _should_show_visualization nữa, để LLM quyết định
        }
        
        # Trích xuất context từ agent trước đó nếu có
        if "context_from_previous_agent" in state:
            task_input["previous_context"] = state["context_from_previous_agent"]
            
            # Log thông tin về agent trước đó
            if isinstance(state["context_from_previous_agent"], dict):
                prev_agent = state["context_from_previous_agent"].get("agent_type", "unknown")
                self.logger.info(f"Received context from previous agent: {prev_agent}")
        
        return task_input

    def _should_show_visualization(self, query: str) -> bool:
        """
        Method này không còn được sử dụng, giữ lại để tương thích với mã cũ.
        Quyết định hiển thị visualization bây giờ do LLM đưa ra.
        """
        return False  # Default value, không còn dùng đến

    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for workflow."""
        image_path = task_input.get("image_path", "")
        query = task_input.get("query", "")
        
        # Thêm thông tin từ agent trước đó nếu có
        previous_context_info = ""
        if "previous_context" in task_input:
            prev_context = task_input["previous_context"]
            if isinstance(prev_context, dict):
                previous_context_info = "\n\nINFORMATION FROM PREVIOUS AGENT:\n"
                for key, value in prev_context.items():
                    if isinstance(value, dict):
                        previous_context_info += f"- {key}:\n"
                        for sub_key, sub_value in value.items():
                            previous_context_info += f"  - {sub_key}: {sub_value}\n"
                    else:
                        previous_context_info += f"- {key}: {value}\n"
        
        formatted_input = f"""**POLYP DETECTION TASK**

Image to analyze: {image_path}
User query: "{query if query else 'Detect polyps in this endoscopy image'}"{previous_context_info}

RESPONSE FORMAT REQUIREMENTS:
You MUST use this exact format for all responses:

Thought: <your reasoning>
Action: <tool_name>
Action Input: {{
  "parameter1": value1,
  "parameter2": value2
}}

OR when finished:

Thought: <your final assessment>
Action: Final Answer
Action Input: {{
  "answer": "Your detailed medical assessment here",
  "show_visualization": true/false  
}}

STRATEGY:
1. Start with standard detection (conf_thresh=0.25, iou_thresh=0.45)
2. If no polyps found, try lower confidence thresholds (0.1-0.2)
3. Adjust IoU threshold if needed (0.3-0.5) for overlapping polyps
4. Create visualizations for verification
5. Analyze the query and decide whether to show visualization to the user
   - If user explicitly asks to see/visualize, set show_visualization=true
   - If query is just asking for polyp detection without visualization, set show_visualization=false
6. Provide detailed medical assessment

Begin your analysis now:"""

        return formatted_input

    def _run_react_loop(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flexible ReAct loop for adaptive detection workflow.
        """
        self.react_history = []
        
        # Set up tracking for the process
        detection_results = None
        visualization_results = None
        show_visualization = False  # Default, sẽ được cập nhật từ quyết định của LLM
        
        # Anti-loop detection
        last_actions = []
        repetition_count = 0
        
        # Run the standard ReAct loop with LLM control
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            
            self.logger.info(f"ReAct iteration {iteration} starting")
            
            # Get LLM to decide next step using the proper method from BaseAgent
            messages = self._create_react_messages(task_input)
            
            # Thêm thông tin về detection results và visualization để context tốt hơn
            if detection_results and detection_results.get("success"):
                count = detection_results.get("count", 0)
                info = f"\n\nIMPORTANT: {count} polyps were detected in the previous step."
                if count > 0:
                    info += f" Use this information for visualization and final assessment."
                    # Thêm chi tiết về polyp đầu tiên nếu có
                    if "objects" in detection_results and len(detection_results["objects"]) > 0:
                        first_polyp = detection_results["objects"][0]
                        info += f"\nFirst polyp confidence: {first_polyp.get('confidence', 'N/A')}"
                else:
                    info += " Consider trying lower confidence threshold."
                    
                # Thêm vào human message cuối cùng
                if isinstance(messages[-1].content, str):
                    messages[-1].content += info
                elif isinstance(messages[-1].content, list):
                    # Tìm và cập nhật phần tử text cuối cùng
                    for i in range(len(messages[-1].content) - 1, -1, -1):
                        if messages[-1].content[i].get("type") == "text":
                            messages[-1].content[i]["text"] += info
                            break
            
            response = self.llm.invoke(messages)
            
            # Ensure there's a response
            if not response or not response.content:
                self.logger.error("Empty response from LLM")
                continue
            
            self.logger.info(f"LLM response (first 100 chars): {response.content[:100]}...")
            
            # Parse the response using the method from BaseAgent
            try:
                thought, action, action_input = self._parse_llm_response(response.content)
                
                if thought is None or action is None:
                    self.logger.error("Failed to parse LLM response: missing thought or action")
                    thought = thought or "No thought provided"
                    action = action or "No action provided"
                
                self.logger.info(f"Parsed thought: {thought[:50]}...")
                self.logger.info(f"Parsed action: {action}")
                
                if action_input:
                    self.logger.info(f"Parsed action input: {json.dumps(action_input)[:100]}...")
                else:
                    self.logger.info("No action input provided")
                
            except Exception as e:
                self.logger.error(f"Error parsing LLM response: {str(e)}")
                # Try to continue with next iteration
                continue
            
            # Anti-loop detection
            if action:
                last_actions.append(action)
                if len(last_actions) >= 3 and len(set(last_actions[-3:])) == 1:
                    repetition_count += 1
                    self.logger.warning(f"Detected action repetition: {action} (count: {repetition_count})")
                    
                    # Nếu lặp quá 2 lần, chuyển sang Final Answer
                    if repetition_count >= 2:
                        self.logger.warning(f"Breaking out of action loop: {action}")
                        action = "Final Answer"
                        thought += " [System detected repeating actions and forced conclusion]"
                        action_input = {
                            "answer": f"Based on my detection, I found {detection_results.get('count', 0)} polyps in the image. [Note: This conclusion was automatically generated due to repetitive actions in the workflow.]",
                            "show_visualization": True  # Default show visualization when breaking out of loops
                        }
                else:
                    repetition_count = 0
                
            # Create and add step to history
            step = ReActStep(
                thought=thought, 
                thought_type=ThoughtType.INITIAL if iteration == 1 else ThoughtType.REASONING,
                action=action, 
                action_input=action_input or {}
            )
            
            # Check if we have a final answer
            if action and action.lower() == "final answer":
                self.logger.info("Got final answer, finishing ReAct loop")
                
                # Lấy quyết định hiển thị visualization từ LLM
                if action_input and isinstance(action_input, dict):
                    show_visualization = action_input.get("show_visualization", False)
                    self.logger.info(f"LLM decided to {'' if show_visualization else 'not '}show visualization")
                
                step.thought_type = ThoughtType.CONCLUSION
                self.react_history.append(step)
                return {
                    "success": True,
                    "answer": action_input.get("answer") if action_input else thought,
                    "detection_data": detection_results or {},
                    "visualization_data": visualization_results or {},
                    "show_visualization": show_visualization
                }
            
            # Execute the chosen tool
            if action:
                if action == "yolo_detection":
                    # Extract dynamic parameters
                    image_path = task_input.get("image_path", "")
                    conf_thresh = 0.25  # Default
                    iou_thresh = 0.45  # Default
                    if action_input:
                        image_path = action_input.get("image_path", image_path)
                        conf_thresh = action_input.get("conf_thresh", 0.25)  # Allow dynamic threshold
                        iou_thresh = action_input.get("iou_thresh", 0.45)    # Allow dynamic IoU threshold
                    
                    self.logger.info(f"Running YOLO detection on {image_path} with conf_thresh={conf_thresh}, iou_thresh={iou_thresh}")
                    
                    # Execute with dynamic parameters
                    detection_results = self._execute_step(
                        action="yolo_detection",
                        task_input={"image_path": image_path},
                        conf_thresh=conf_thresh,
                        iou_thresh=iou_thresh
                    )
                    
                    if detection_results.get("success"):
                        self.logger.info(f"Detection successful, found {detection_results.get('count', 0)} objects")
                    else:
                        self.logger.error(f"Detection failed: {detection_results.get('error', 'unknown error')}")
                    
                    # Add the observation
                    step.observation = json.dumps(detection_results, indent=2)
                    step.thought_type = ThoughtType.OBSERVATION
                    
                    # Cập nhật task_input để lần sau sử dụng
                    task_input["latest_detection_results"] = detection_results
                    
                elif action == "visualize_detections":
                    image_path = task_input.get("image_path", "")
                    detections = []
                    
                    if action_input:
                        image_path = action_input.get("image_path", image_path)
                        
                        # Handle different ways detections might be provided
                        if "detections" in action_input:
                            detections = action_input.get("detections", [])
                        elif "objects" in action_input:  # Handle case where the full detection result is passed
                            detections = action_input.get("objects", [])
                    
                    # Always use detection_results nếu không có detections được cung cấp
                    if not detections and detection_results and "objects" in detection_results:
                        self.logger.info("No detections provided, using latest detection results")
                        detections = detection_results.get("objects", [])
                    
                    self.logger.info(f"Visualizing {len(detections)} detections on {image_path}")
                    
                    visualization_results = self._execute_step(
                        action="visualize_detections",
                        task_input={"image_path": image_path},
                        detections=detections
                    )
                    
                    if visualization_results.get("success"):
                        self.logger.info("Visualization created successfully")
                        # Sau khi visualize thành công, gợi ý LLM đưa ra final answer ở lần tiếp theo
                        task_input["visualization_hint"] = "Visualization completed successfully. Consider providing your final assessment in the next step."
                    else:
                        self.logger.error(f"Visualization failed: {visualization_results.get('error', 'unknown error')}")
                    
                    # Add the observation
                    step.observation = json.dumps(visualization_results, indent=2)
                    step.thought_type = ThoughtType.OBSERVATION
                    
                    # Cập nhật task_input để lần sau sử dụng
                    task_input["latest_visualization_results"] = visualization_results
                
                else:
                    self.logger.warning(f"Unknown action: {action}")
                    step.observation = json.dumps({"success": False, "error": f"Unknown action: {action}"})
                    step.thought_type = ThoughtType.OBSERVATION
                
                # Add the step to history
                self.react_history.append(step)
                
                # Update task_input with observation for next iteration
                task_input[f"obs_{iteration}"] = step.observation
            else:
                self.logger.warning("No action provided by LLM")
        
        self.logger.warning(f"Reached max iterations ({self.max_iterations}) without conclusion")
        # If we reach max iterations without a final answer
        return {
            "success": False,
            "error": f"Reached maximum iterations ({self.max_iterations}) without conclusion",
            "detection_data": detection_results or {},
            "visualization_data": visualization_results or {},
            "show_visualization": False  # Default khi không có final answer
        }

    def _execute_step(self, action: str, task_input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute a step with dynamic parameters."""
        try:
            if action == "yolo_detection":
                image_path = task_input.get("image_path", "")
                # Use dynamic parameters with fallback defaults
                conf_thresh = kwargs.get("conf_thresh", 0.25)
                iou_thresh = kwargs.get("iou_thresh", 0.45)  # Lấy iou_thresh từ kwargs
                
                result = self.detector_tool._run(
                    image_path=image_path, 
                    conf_thresh=conf_thresh,
                    iou_thresh=iou_thresh  # Truyền tham số iou_thresh
                )
                
                # Thêm thông tin về tham số đã sử dụng vào kết quả
                if result.get("success", False) and not "parameters" in result:
                    result["conf_thresh"] = conf_thresh
                    result["iou_thresh"] = iou_thresh
                
            elif action == "visualize_detections":
                image_path = task_input.get("image_path", "")
                detections = kwargs.get("detections", [])
                result = self.visualize_tool._run(image_path=image_path, detections=detections)
                
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
            
            # Log step completion
            self.logger.info(f"Step ({action}) completed: {result.get('success', False)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Step ({action}) failed: {str(e)}")
            return {"success": False, "error": str(e)}

    def _execute_synthesis_with_visualization(self, task_input: Dict[str, Any], 
                                           detection_results: Dict[str, Any],
                                           visualization_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute final synthesis step with multimodal input:
        - Original image for context
        - Visualization image for review
        """
        try:
            import base64
            from PIL import Image
            from io import BytesIO
            
            def image_to_base64(image_path):
                with Image.open(image_path) as img:
                    buffered = BytesIO()
                    img.save(buffered, format="PNG")
                    return base64.b64encode(buffered.getvalue()).decode()

            image_path = task_input.get("image_path", "")
            query = task_input.get("query", "")
            
            # Prepare synthesis prompt
            synthesis_prompt = f"""**SYNTHESIS TASK: Review Detection Results with Visualization**

Original Query: "{query}"
Detection Results: {json.dumps(detection_results, indent=2)}
Visualization Created: {visualization_results.get('success', False)}

SYNTHESIS REQUIREMENTS:
1. Review the detection results carefully
2. Analyze the visualization I created to verify findings
3. Provide comprehensive medical assessment
4. Include confidence levels and clinical recommendations
5. Analyze if the user wants to see visualization in their query
6. Set show_visualization=true/false based on user's intention

Please provide your final medical assessment:"""

            # Convert original image to base64
            img_b64 = image_to_base64(image_path)
            
            # Create multimodal message (original image + visualization)
            messages = [
                SystemMessage(content=self._get_agent_description()),
                HumanMessage(
                    content=[
                        {"type": "text", "text": synthesis_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ]
                )
            ]
            
            # Add visualization image if available
            if visualization_results.get("success") and visualization_results.get("visualization_base64"):
                viz_b64 = visualization_results["visualization_base64"]
                messages[-1].content.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{viz_b64}"}
                })
            
            # Get synthesis from LLM
            response = self.llm.invoke(messages)
            synthesis_answer = response.content.strip()
            
            return {
                "success": True,
                "answer": synthesis_answer,
                "detection_data": detection_results,
                "visualization_data": visualization_results,
                "reviewed_visualization": True
            }
            
        except Exception as e:
            self.logger.error(f"Synthesis failed: {str(e)}")
            return {
                "success": False, 
                "error": f"Synthesis failed: {str(e)}",
                "detection_data": detection_results,
                "visualization_data": visualization_results
            }

    def _format_agent_result(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format detector result."""
        if not react_result.get("success"):
            return {
                "detector_result": {
                    "success": False,
                    "error": react_result.get("error", "detection failed"),
                },
                "context_for_next_agent": {
                    "agent_type": "detector",
                    "success": False,
                    "error": react_result.get("error", "detection failed")
                }
            }
        
        # Extract data from execution
        detection_data = react_result.get("detection_data", {})
        visualization_data = react_result.get("visualization_data", {})
        show_visualization = react_result.get("show_visualization", False)
        
        result = {
            "success": True,
            "count": detection_data.get("count", 0),
            "objects": detection_data.get("objects", []),
            "analysis": react_result.get("answer", "detection completed"),
            "visualization_available": visualization_data.get("success", False),
            "synthesis_reviewed_visualization": react_result.get("reviewed_visualization", False),
            "show_visualization": show_visualization,  # Quyết định từ LLM
            "parameters_used": {
                "conf_thresh": detection_data.get("conf_thresh", 0.25),
                "iou_thresh": detection_data.get("iou_thresh", 0.45)
            }
        }
        
        # Include visualization data if LLM quyết định hiển thị
        if show_visualization and visualization_data.get("success") and visualization_data.get("visualization_base64"):
            result["visualization_base64"] = visualization_data["visualization_base64"]
        
        # Tạo context cho agent tiếp theo
        context_for_next = {
            "agent_type": "detector",
            "success": True,
            "polyp_count": detection_data.get("count", 0),
            "detection_confidence": detection_data.get("conf_thresh", 0.25),
            "analysis_summary": react_result.get("answer", "detection completed")[:200] + "..." if len(react_result.get("answer", "")) > 200 else react_result.get("answer", ""),
        }
        
        # Nếu có phát hiện polyp, thêm chi tiết về polyp đầu tiên
        if detection_data and detection_data.get("count", 0) > 0 and "objects" in detection_data and len(detection_data["objects"]) > 0:
            first_polyp = detection_data["objects"][0]
            context_for_next["first_polyp"] = {
                "confidence": first_polyp.get("confidence", 0),
                "position": first_polyp.get("position_description", "unknown"),
                "size": {
                    "width": first_polyp.get("width", 0),
                    "height": first_polyp.get("height", 0),
                    "area": first_polyp.get("area", 0)
                }
            }
        
        return {
            "detector_result": result,
            "context_for_next_agent": context_for_next
        }

    def _create_react_messages(self, task_input: Dict[str, Any]) -> List[Any]:
        """
        Ghi đè phương thức _create_react_messages từ BaseAgent để thêm context tốt hơn.
        """
        # Bắt đầu với system prompt và task input
        msgs = [
            SystemMessage(content=self._get_system_prompt()), 
            HumanMessage(content=self._format_task_input(task_input))
        ]
        
        # Nếu có history, format và thêm vào
        if self.react_history:
            hist = []
            for s in self.react_history[-5:]:  # Chỉ dùng 5 steps gần nhất để tránh context quá dài
                hist_text = f"Thought: {s.thought}\n"
                hist_text += f"Action: {s.action}\n"
                
                if s.action_input:
                    hist_text += f"Action Input: {json.dumps(s.action_input, indent=2)}\n"
                    
                if s.observation:
                    # Format observation để dễ đọc hơn
                    try:
                        obs_json = json.loads(s.observation)
                        if s.action == "yolo_detection" and "count" in obs_json:
                            hist_text += f"Observation: Found {obs_json.get('count', 0)} polyps. "
                            if obs_json.get('count', 0) > 0 and "objects" in obs_json:
                                hist_text += f"First polyp confidence: {obs_json['objects'][0].get('confidence', 'N/A')}\n"
                            else:
                                hist_text += "Consider trying a lower confidence threshold.\n"
                        elif s.action == "visualize_detections" and obs_json.get("success", False):
                            hist_text += "Observation: Visualization created successfully. Ready for final assessment.\n"
                        else:
                            # Rút gọn observation để tránh quá dài
                            short_obs = json.dumps(obs_json)[:200] + "..." if len(json.dumps(obs_json)) > 200 else json.dumps(obs_json)
                            hist_text += f"Observation: {short_obs}\n"
                    except:
                        # Nếu không phải JSON, dùng text thông thường
                        short_obs = s.observation[:200] + "..." if s.observation and len(s.observation) > 200 else s.observation
                        hist_text += f"Observation: {short_obs}\n"
                
                hist.append(hist_text)
            
            # Thêm hint để hướng dẫn bước tiếp theo
            next_step_hint = ""
            
            # Bước cuối cùng là gì?
            if self.react_history and self.react_history[-1].action == "visualize_detections":
                next_step_hint = "\n\nFinal step: Now that you have both detection and visualization results, please provide your final medical assessment."
            elif self.react_history and self.react_history[-1].action == "yolo_detection":
                detections = 0
                try:
                    obs = json.loads(self.react_history[-1].observation) if self.react_history[-1].observation else {}
                    detections = obs.get("count", 0)
                except:
                    pass
                    
                if detections > 0:
                    next_step_hint = "\n\nNext step: Consider creating a visualization of the detected polyps before providing your final assessment."
                else:
                    next_step_hint = "\n\nNext step: You found no polyps at current threshold. Consider lowering the confidence threshold or proceeding to final assessment."
            
            # Thêm history và hint vào message
            history_msg = "\n\n".join(hist) + next_step_hint
            msgs.append(AIMessage(content=history_msg))
        
        # Thêm ảnh nếu cần - chỉ khi mới bắt đầu để tránh lặp lại ảnh mỗi vòng lặp
        if "image_path" in task_input and not self.react_history:
            import base64
            import os
            if os.path.exists(task_input['image_path']):
                try:
                    with open(task_input['image_path'], 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode()
                        image_reminder = f"Image path: {task_input['image_path']}"
                        msgs.append(HumanMessage(content=[
                            {"type": "text", "text": image_reminder},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}}
                        ]))
                        self.logger.info(f"Added image to messages: {task_input['image_path']}")
                except Exception as e:
                    self.logger.error(f"Error loading image: {str(e)}")
        
        # Thêm visualization hint nếu có
        if "visualization_hint" in task_input:
            msgs.append(HumanMessage(content=task_input["visualization_hint"]))
        
        return msgs

# ===== USAGE EXAMPLE =====
def test_detector():
    """Test the detector agent."""
    
    detector = DetectorAgent(
        model_path="medical_ai_agents/weights/detect_best.pt",
        device="cuda"
    )
    
    # Test case 1: User wants to see visualization
    test_state_1 = {
        "image_path": "test_image.jpg",
        "query": "Please detect polyps and show me the results with visualization"
    }
    
    # Test case 2: User doesn't request visualization
    test_state_2 = {
        "image_path": "test_image.jpg", 
        "query": "Are there any polyps in this image?"
    }
    
    print("=== DETECTOR TEST ===")
    
    for i, test_state in enumerate([test_state_1, test_state_2], 1):
        print(f"\nTest Case {i}: {test_state['query']}")
        result = detector.process(test_state)
        
        if result.get("detector_result"):
            det_result = result["detector_result"]
            print(f"Success: {det_result.get('success')}")
            print(f"Parameters used: {det_result.get('parameters_used')}")
            print(f"Polyps found: {det_result.get('count', 0)}")
            print(f"Visualization available: {det_result.get('visualization_available')}")

if __name__ == "__main__":
    test_detector()