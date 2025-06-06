#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TIERED DETECTOR AGENT - APPROACH (Safety Critical)
============================================================
Fixed workflow: detect → visualize → synthesis (always look at visualization)
"""

import json
import os
import logging
from typing import Dict, Any, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage

from medical_ai_agents.agents.base_agent import BaseAgent, ThoughtType
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.detection.yolo_tools import YOLODetectionTool
from medical_ai_agents.tools.detection.util_tools import VisualizationTool

class DetectorAgent(BaseAgent):
    """
    Detector Agent - Safety Critical
    
    FIXED WORKFLOW (no deviations allowed):
    1. YOLO Detection (required)
    2. Visualization Creation (always - for synthesis)
    3. Synthesis with Visualization Review (required)
    """
    
    def __init__(self, model_path: str, llm_model: str = "gpt-4o-mini", device: str = "cuda"):
        self.model_path = model_path
        super().__init__(name="Detector Agent", llm_model=llm_model, device=device)
        
        # configuration
        self.max_iterations = 3  # Exactly 3 steps: detect → visualize → synthesis
        self.required_workflow = [
            {"step": 1, "action": "yolo_detection", "required": True},
            {"step": 2, "action": "visualize_detections", "required": True},  # Always for synthesis
            {"step": 3, "action": "synthesis_with_visualization", "required": True}
        ]
    
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
        return """I am a SAFETY-CRITICAL polyp detection specialist using workflow.

My MANDATORY process (no deviations):
1. DETECT: Use YOLO to find polyps with precise confidence scores
2. VISUALIZE: Always create visualization for internal analysis 
3. SYNTHESIZE: Review both original image and visualization to provide final assessment

I ALWAYS create visualization for synthesis, but only show to user if they request it."""

    def _get_system_prompt(self) -> str:
        """system prompt with fixed workflow."""
        return f"""You are a SAFETY-CRITICAL polyp detection specialist following workflow.

MANDATORY 3-STEP PROCESS (follow exactly):

Step 1 - DETECTION (REQUIRED):
Thought: I must perform polyp detection for patient safety
Action: yolo_detection
Action Input: {{"image_path": "<path>", "conf_thresh": 0.25}}

Step 2 - VISUALIZATION (ALWAYS REQUIRED):
Thought: Creating visualization for internal analysis and synthesis
Action: visualize_detections  
Action Input: {{"image_path": "<path>", "detections": [results_from_step_1]}}

Step 3 - SYNTHESIS (REQUIRED):
Final Answer: [Review both original image and visualization, then provide comprehensive medical assessment]

CRITICAL RULES:
- NEVER skip any step
- ALWAYS create visualization (even if 0 polyps for empty visualization) 
- MUST review visualization in final synthesis
- Be precise and medically accurate
- Maximum 3 steps only

Available tools: {self.tool_descriptions}

Start with Step 1:"""

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
        return {
            "image_path": state.get("image_path", ""),
            "query": state.get("query", ""),
            "medical_context": state.get("medical_context", {}),
            "show_visualization": self._should_show_visualization(state.get("query", ""))
        }

    def _should_show_visualization(self, query: str) -> bool:
        """Analyze query intention to decide if user wants to see visualization."""
        if not query:
            return False
            
        query_lower = query.lower()
        
        # Explicit visualization requests
        show_keywords = [
            "show", "display", "visualize", "see", "view", "image", "picture",
            "highlight", "mark", "point out", "circle", "box", "outline",
            "hiện", "hiển thị", "cho xem", "khoanh", "đánh dấu", "chỉ ra"
        ]
        
        return any(keyword in query_lower for keyword in show_keywords)

    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for workflow."""
        image_path = task_input.get("image_path", "")
        query = task_input.get("query", "")
        show_viz = task_input.get("show_visualization", False)
        
        return f"""**POLYP DETECTION TASK**

Image to analyze: {image_path}
User query: "{query if query else 'Detect polyps in this endoscopy image'}"
Show visualization to user: {show_viz}

MANDATORY WORKFLOW:
Follow the exact 3-step process defined in your system prompt.
Step 1: yolo_detection
Step 2: visualize_detections (always create for synthesis)
Step 3: Final Answer with synthesis

Begin Step 1 now:"""

    def _run_react_loop(self, task_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override ReAct loop for workflow.
        Fixed 3-step process with no deviations.
        """
        self.react_history = []
        
        # STEP 1: DETECTION (Required)
        step1_result = self._execute_step(
            step_num=1,
            action="yolo_detection", 
            task_input=task_input
        )
        
        if not step1_result.get("success", False):
            return {"success": False, "error": "Step 1 (detection) failed", "step_failed": 1}
        
        # STEP 2: VISUALIZATION (Always required for synthesis)
        step2_result = self._execute_step(
            step_num=2,
            action="visualize_detections",
            task_input=task_input,
            detection_results=step1_result
        )
        
        if not step2_result.get("success", False):
            return {"success": False, "error": "Step 2 (visualization) failed", "step_failed": 2}
        
        # STEP 3: SYNTHESIS with multimodal input (original + visualization)
        synthesis_result = self._execute_synthesis_with_visualization(
            task_input=task_input,
            detection_results=step1_result,
            visualization_results=step2_result
        )
        
        return synthesis_result

    def _execute_step(self, step_num: int, action: str, task_input: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Execute a step with validation."""
        try:
            if action == "yolo_detection":
                image_path = task_input.get("image_path", "")
                result = self.detector_tool._run(image_path=image_path, conf_thresh=0.25)
                
            elif action == "visualize_detections":
                image_path = task_input.get("image_path", "")
                detection_results = kwargs.get("detection_results", {})
                detections = detection_results.get("objects", [])
                result = self.visualize_tool._run(image_path=image_path, detections=detections)
                
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
            
            # Log step completion
            self.logger.info(f"Step {step_num} ({action}) completed: {result.get('success', False)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Step {step_num} ({action}) failed: {str(e)}")
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
            show_viz = task_input.get("show_visualization", False)
            
            # Prepare synthesis prompt
            synthesis_prompt = f"""**SYNTHESIS TASK: Review Detection Results with Visualization**

Original Query: "{query}"
Detection Results: {json.dumps(detection_results, indent=2)}
Visualization Created: {visualization_results.get('success', False)}
Show visualization to user: {show_viz}

SYNTHESIS REQUIREMENTS:
1. Review the detection results carefully
2. Analyze the visualization I created to verify findings
3. Provide comprehensive medical assessment
4. Include confidence levels and clinical recommendations
5. Only mention showing visualization if user requested it

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
                    "step_failed": react_result.get("step_failed", 0),
                }
            }
        
        # Extract data from execution
        detection_data = react_result.get("detection_data", {})
        visualization_data = react_result.get("visualization_data", {})
        
        result = {
            "success": True,
            "count": detection_data.get("count", 0),
            "objects": detection_data.get("objects", []),
            "analysis": react_result.get("answer", "detection completed"),
            "visualization_available": visualization_data.get("success", False),
            "synthesis_reviewed_visualization": react_result.get("reviewed_visualization", False),
            "steps_completed": 3
        }
        
        # Include visualization data
        if visualization_data.get("success") and visualization_data.get("visualization_base64"):
            result["visualization_base64"] = visualization_data["visualization_base64"]
        
        return {"detector_result": result}

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
            print(f"Steps completed: {det_result.get('steps_completed')}")
            print(f"Reviewed visualization: {det_result.get('synthesis_reviewed_visualization')}")
            print(f"Visualization available: {det_result.get('visualization_available')}")

if __name__ == "__main__":
    test_detector()