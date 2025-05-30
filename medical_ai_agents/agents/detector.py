#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - Smart Detector Agent
---------------------------------------
Agent phát hiện polyp với intelligent tool parsing và decision making.
"""

import json
from typing import Dict, Any, List
import logging
import re
import time

from langchain.schema import HumanMessage, SystemMessage

from medical_ai_agents.agents.base_agent import BaseAgent
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.detection.yolo_tools import YOLODetectionTool
from medical_ai_agents.tools.detection.util_tools import VisualizationTool

class DetectorAgent(BaseAgent):
    """Smart Detector Agent với intelligent tool orchestration."""
    
    def __init__(self, model_path: str, llm_model: str = "gpt-4o-mini", device: str = "cuda"):
        """
        Khởi tạo Smart Detector Agent.
        
        Args:
            model_path: Đường dẫn đến YOLO model weights
            llm_model: Mô hình LLM sử dụng làm controller
            device: Device để chạy model (cuda/cpu)
        """
        self.model_path = model_path
        super().__init__(name="Smart Detector Agent", llm_model=llm_model, device=device)
        self.yolo_tool = None
        self.visualize_tool = None
    
    def _register_tools(self) -> List[BaseTool]:
        """Register tools for this agent."""
        self.yolo_tool = YOLODetectionTool(model_path=self.model_path, device=self.device)
        self.visualize_tool = VisualizationTool()
        return [self.yolo_tool, self.visualize_tool]
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines this agent's role."""
        return """Bạn là một AI chuyên gia về phát hiện polyp trong hình ảnh nội soi tiêu hóa với khả năng reasoning thông minh về tools.

**Available Tools:**
1. `yolo_detection`: Phát hiện polyp sử dụng YOLO model
   - Input: image_path (required), conf_thresh (optional, default=0.25)
   - Output: List of detected polyps với bbox, confidence, position info
   - Use case: Luôn cần để phát hiện polyp

2. `visualize_detections`: Tạo visualization với bounding boxes
   - Input: image_path (required), detections (required - từ yolo_detection output)
   - Output: Base64 encoded image với bounding boxes
   - Use case: Khi user muốn SEE/VISUALIZE results, hoặc khi phát hiện được polyp và helpful to show

**Your Intelligence:**
Bạn cần REASONING về việc sử dụng tools dựa trên:

1. **User Intent Analysis:**
   - Explicit visualization request: "show", "display", "visualize", "see image"
   - Implicit need: Câu hỏi về vị trí, số lượng polyp → helpful to visualize
   - Pure detection: Chỉ hỏi có/không có polyp

2. **Context Reasoning:**
   - Nếu phát hiện được polyp → visualization adds value
   - Nếu không có polyp → visualization ít giá trị
   - Medical context: Bệnh nhân có thể cần hiểu rõ hơn qua hình ảnh

3. **Tool Orchestration Logic:**
   ```
   ALWAYS: yolo_detection first
   THEN: IF (user wants visualization OR polyps detected AND helpful) 
         → visualize_detections với detections từ yolo
   ```

**Response Format:**
Bạn sẽ REASON trước, sau đó decide tools:

```
REASONING: [Explain your thinking about user intent and tool needs]

TOOL_PLAN:
Tool: yolo_detection
Parameters: {"image_path": "path", "conf_thresh": 0.25}

[IF visualization needed:]
Tool: visualize_detections  
Parameters: {"image_path": "path", "detections": <results_from_yolo>}
```

**Key Principles:**
- ALWAYS run yolo_detection first
- INTELLIGENTLY decide if visualization adds value
- EXPLAIN your reasoning for tool choices
- Use medical context to inform decisions"""

    def initialize(self) -> bool:
        """Khởi tạo agent và các công cụ."""
        try:
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
            "medical_context": state.get("medical_context", {}),
            "user_type": state.get("user_type", "patient")  # patient, doctor, researcher
        }
    
    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for intelligent LLM reasoning."""
        image_path = task_input.get("image_path", "")
        query = task_input.get("query", "")
        context = task_input.get("medical_context", {})
        user_type = task_input.get("user_type", "patient")
        
        context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()]) if context else "- None"
        
        return f"""**TASK: Polyp Detection Analysis**

Image to analyze: {image_path}
User query: "{query if query else 'Analyze for polyps'}"
User type: {user_type}
Medical context:
{context_str}

**YOUR MISSION:**
1. First, REASON about the user's intent and what would be most helpful
2. Decide which tools to use and in what order
3. Execute your tool plan

**REASONING GUIDELINES:**
- Consider if user wants to SEE results or just know findings
- Think about medical value of visualization
- Factor in user type (patients often benefit from visual explanations)
- Consider detected polyp count/significance

Start with your REASONING, then provide TOOL_PLAN:"""
    
    def _format_synthesis_input(self) -> str:
        """Format synthesis input for LLM prompt."""
        return """
**SYNTHESIS TASK:**
Based on tool execution results in image and user query, provide comprehensive analysis:

**Required Output Format:**
```json
{
  "detector_result": {
    "success": true/false,
    "count": number_of_polyps,
    "objects": [...detected polyps with details...],
    "analysis": "Professional medical interpretation of findings",
    "visualization_available": true/false,
    "visualization_base64": "base64_string_if_available",
  }
}
```

**Analysis Guidelines:**
- Interpret polyp count and characteristics
- Explain medical significance  
- Mention visualization if created
- Provide actionable insights for user
"""
    
    def _parse_tool_calls(self, plan: str) -> List[Dict[str, Any]]:
        """Intelligent parsing of LLM reasoning and tool decisions."""
        tool_calls = []
        
        self.logger.info(f"[Smart Detector] Parsing LLM plan:\n{plan}")
        
        # Method 1: Parse REASONING and TOOL_PLAN sections
        sections = self._extract_reasoning_and_tools(plan)
        
        if sections["tools"]:
            tool_calls = self._parse_structured_tools(sections["tools"])
            self.logger.info(f"[Smart Detector] Extracted from structured format: {len(tool_calls)} tools")
        
        # Method 2: Fallback - Look for Tool: patterns anywhere
        if not tool_calls:
            tool_calls = self._parse_tool_patterns(plan)
            self.logger.info(f"[Smart Detector] Extracted from patterns: {len(tool_calls)} tools")
        
        # Method 3: Intelligent fallback based on reasoning
        if not tool_calls:
            tool_calls = self._intelligent_fallback(plan, sections.get("reasoning", ""))
            self.logger.info(f"[Smart Detector] Generated from intelligent fallback: {len(tool_calls)} tools")
        
        # Validate and enhance tool calls
        tool_calls = self._enhance_tool_calls(tool_calls)
        
        self.logger.info(f"[Smart Detector] Final tool calls: {json.dumps(tool_calls, indent=2)}")
        return tool_calls
    
    def _extract_reasoning_and_tools(self, plan: str) -> Dict[str, str]:
        """Extract REASONING and TOOL_PLAN sections from LLM response."""
        sections = {"reasoning": "", "tools": ""}
        
        # Look for REASONING section
        reasoning_match = re.search(r'REASONING:\s*(.*?)(?=TOOL_PLAN:|Tool:|$)', plan, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            sections["reasoning"] = reasoning_match.group(1).strip()
        
        # Look for TOOL_PLAN section
        tool_plan_match = re.search(r'TOOL_PLAN:\s*(.*?)(?=REASONING:|$)', plan, re.DOTALL | re.IGNORECASE)
        if tool_plan_match:
            sections["tools"] = tool_plan_match.group(1).strip()
        else:
            # Look for tools after reasoning
            if sections["reasoning"]:
                remaining = plan[plan.find(sections["reasoning"]) + len(sections["reasoning"]):]
                sections["tools"] = remaining.strip()
        
        return sections
    
    def _parse_structured_tools(self, tools_text: str) -> List[Dict[str, Any]]:
        """Parse tools from structured TOOL_PLAN format."""
        tool_calls = []
        lines = tools_text.split('\n')
        
        current_tool = None
        current_params = {}
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Tool:'):
                # Save previous tool
                if current_tool:
                    tool_calls.append({
                        "tool_name": current_tool,
                        "params": current_params,
                        "source": "structured"
                    })
                
                # Start new tool
                current_tool = line.replace('Tool:', '').strip()
                current_params = {}
                
            elif line.startswith('Parameters:'):
                # Parse JSON parameters
                params_text = line.replace('Parameters:', '').strip()
                try:
                    if params_text.startswith('{') and params_text.endswith('}'):
                        current_params = json.loads(params_text)
                    else:
                        # Try to construct from key=value pairs
                        current_params = self._parse_key_value_params(params_text)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse parameters: {params_text}")
                    current_params = {}
        
        # Add last tool
        if current_tool:
            tool_calls.append({
                "tool_name": current_tool,
                "params": current_params,
                "source": "structured"
            })
        
        return tool_calls
    
    def _parse_tool_patterns(self, plan: str) -> List[Dict[str, Any]]:
        """Parse tools using regex patterns as fallback."""
        tool_calls = []
        
        # Pattern for Tool: ... Parameters: {...}
        tool_pattern = r'Tool:\s*(\w+).*?Parameters:\s*(\{[^}]*\})'
        matches = re.findall(tool_pattern, plan, re.DOTALL | re.IGNORECASE)
        
        for tool_name, params_str in matches:
            try:
                params = json.loads(params_str)
                tool_calls.append({
                    "tool_name": tool_name.strip(),
                    "params": params,
                    "source": "pattern"
                })
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse tool parameters: {params_str}")
        
        return tool_calls
    
    def _intelligent_fallback(self, plan: str, reasoning: str) -> List[Dict[str, Any]]:
        """Intelligent fallback based on LLM reasoning and context."""
        tool_calls = []
        
        # Always need yolo_detection
        base_tool = {
            "tool_name": "yolo_detection",
            "params": {"image_path": ""},  # Will be filled in _enhance_tool_calls
            "source": "intelligent_fallback"
        }
        tool_calls.append(base_tool)
        
        # Analyze reasoning for visualization need
        viz_indicators = [
            "visualiz", "show", "display", "see", "image", "bounding", "box",
            "helpful", "understand", "explain", "patient", "visual"
        ]
        
        reasoning_lower = (reasoning + " " + plan).lower()
        viz_score = sum(1 for indicator in viz_indicators if indicator in reasoning_lower)
        
        # Also check for explicit mentions
        explicit_viz = any(phrase in reasoning_lower for phrase in [
            "visualization", "visualize", "show image", "display result",
            "bounding box", "helpful to show"
        ])
        
        if explicit_viz or viz_score >= 3:
            viz_tool = {
                "tool_name": "visualize_detections",
                "params": {
                    "image_path": "",  # Will be filled
                    "detections": []   # Will be filled from yolo results
                },
                "source": "intelligent_fallback"
            }
            tool_calls.append(viz_tool)
            self.logger.info(f"[Smart Detector] Intelligent fallback decided visualization needed (score: {viz_score}, explicit: {explicit_viz})")
        
        return tool_calls
    
    def _enhance_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance tool calls with missing information."""
        for tool_call in tool_calls:
            params = tool_call.get("params", {})
            
            # Ensure image_path is present (will be filled from state)
            if "image_path" not in params or not params["image_path"]:
                params["image_path"] = "__STATE_IMAGE_PATH__"  # Placeholder
            
            # Set default confidence threshold for yolo
            if tool_call["tool_name"] == "yolo_detection" and "conf_thresh" not in params:
                params["conf_thresh"] = 0.25
            
            tool_call["params"] = params
        
        return tool_calls
    
    def _parse_key_value_params(self, params_text: str) -> Dict[str, Any]:
        """Parse parameters from key=value format."""
        params = {}
        # Simple parser for image_path="value" format
        matches = re.findall(r'(\w+)=(["\']?)([^"\',\s]+)\2', params_text)
        for key, quote, value in matches:
            # Try to convert to appropriate type
            if value.lower() in ['true', 'false']:
                params[key] = value.lower() == 'true'
            elif value.replace('.', '').isdigit():
                params[key] = float(value) if '.' in value else int(value)
            else:
                params[key] = value
        return params
    
    def _process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced process_state with intelligent tool orchestration."""
        try:
            # Ensure initialized
            if not self.initialized:
                success = self.initialize()
                if not success:
                    return {**state, "error": f"Failed to initialize {self.name}"}
            
            # Extract task input
            task_input = self._extract_task_input(state)
            self.logger.info(f"[Smart Detector] Processing task: {json.dumps(task_input, indent=2)}")
            
            # Get LLM reasoning and tool plan
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self._format_task_input(task_input))
            ]
            
            response = self.llm.invoke(messages)
            plan = response.content
            self.logger.info(f"[Smart Detector] LLM reasoning and plan:\n{plan}")
            
            # Parse tool calls intelligently
            tool_calls = self._parse_tool_calls(plan)
            
            # Execute tools with intelligent orchestration
            results = {}
            tool_outputs = {}
            
            for i, tool_call in enumerate(tool_calls):
                tool_name = tool_call.get("tool_name")
                params = tool_call.get("params", {})
                
                # Fill in image_path from state
                if params.get("image_path") == "__STATE_IMAGE_PATH__":
                    params["image_path"] = task_input.get("image_path", "")
                
                self.logger.info(f"[Smart Detector] Executing tool {i+1}/{len(tool_calls)}: {tool_name}")
                
                # Special handling for visualization
                if tool_name == "visualize_detections":
                    # Need detections from previous yolo_detection
                    if "yolo_detection" in tool_outputs:
                        yolo_result = tool_outputs["yolo_detection"]
                        if yolo_result.get("success", False):
                            params["detections"] = yolo_result.get("objects", [])
                            self.logger.info(f"[Smart Detector] Using {len(params['detections'])} detections for visualization")
                        else:
                            self.logger.warning("[Smart Detector] Skipping visualization - no valid detections")
                            continue
                    else:
                        self.logger.warning("[Smart Detector] Skipping visualization - no yolo_detection results")
                        continue
                
                # Execute tool
                tool_result = self.execute_tool(tool_name, **params)
                results[tool_name] = tool_result
                tool_outputs[tool_name] = tool_result
                
                self.logger.info(f"[Smart Detector] Tool {tool_name} completed: {tool_result.get('success', False)}")
            
            # Synthesize results
            # Tách kết quả visualization khỏi các kết quả khác
            viz_results = {}
            other_results = {}
            img_base64 = None
            
            for tool_name, result in results.items():
                if tool_name == "visualize_detections" and result.get("success"):
                    viz_results[tool_name] = {
                        "success": result["success"],
                        "count": result.get("count", 0)
                    }
                    # Lưu base64 image riêng để đưa vào multipart message
                    img_base64 = result.get("visualization_base64")
                else:
                    other_results[tool_name] = result
            
            synthesis_messages = [
                SystemMessage(content=self.system_prompt),
            ]
            
            # Tạo multipart message với text và image nếu có visualization
            if img_base64:
                synthesis_messages.append(
                    HumanMessage(
                        content=[
                            {"type": "text", "text": f"Tool execution results:\n{json.dumps(other_results, indent=2)}\n\n{self._format_synthesis_input()}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]
                    )
                )
            else:
                # Fallback về text-only message nếu không có visualization
                synthesis_messages.append(
                    HumanMessage(content=f"Tool execution results:\n{json.dumps(results, indent=2)}\n\n{self._format_synthesis_input()}")
                )
            
            synthesis_response = self.llm.invoke(synthesis_messages)
            agent_result = self._extract_agent_result(synthesis_response.content)
            # Add visualization info if available
            if "visualize_detections" in tool_outputs:
                viz_result = tool_outputs["visualize_detections"]
                if viz_result.get("success") and "detector_result" in agent_result:
                    agent_result["detector_result"]["visualization_base64"] = viz_result.get("visualization_base64")
                    agent_result["detector_result"]["visualization_available"] = True
            
            return {**state, **agent_result}
            
        except Exception as e:
            import traceback
            error_msg = f"Error in {self.name}: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return {**state, "error": error_msg}
    
    def _extract_agent_result(self, synthesis: str) -> Dict[str, Any]:
        """Extract agent result with enhanced JSON parsing."""
        try:
            # Multiple JSON extraction methods
            json_str = None
            
            # Method 1: Find complete JSON block
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
                
                if brace_count == 0:
                    json_str = synthesis[json_start:json_end]
            
            # Method 2: Extract from code blocks
            if not json_str:
                json_block = re.search(r'```json\s*(\{.*?\})\s*```', synthesis, re.DOTALL)
                if json_block:
                    json_str = json_block.group(1)
            
            # Parse JSON
            if json_str:
                result = json.loads(json_str)
                self.logger.info("[Smart Detector] Successfully extracted JSON result")
                return result
            
            # Fallback: Create from synthesis text
            self.logger.warning("[Smart Detector] Creating fallback result from text")
            return {
                "detector_result": {
                    "success": True,
                    "count": 0,
                    "objects": [],
                    "analysis": synthesis,
                    "reasoning": "Generated from text synthesis"
                }
            }
            
        except json.JSONDecodeError as e:
            self.logger.error(f"[Smart Detector] JSON parsing failed: {str(e)}")
            return {
                "detector_result": {
                    "success": False,
                    "error": f"JSON parsing failed: {str(e)}",
                    "analysis": synthesis
                }
            }
        except Exception as e:
            self.logger.error(f"[Smart Detector] Result extraction failed: {str(e)}")
            return {
                "detector_result": {
                    "success": False,
                    "error": str(e),
                    "analysis": synthesis
                }
            }