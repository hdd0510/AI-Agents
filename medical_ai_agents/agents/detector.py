#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - Simplified Detector Agent (Following New Pattern)
--------------------------------------------------------------------
Agent phát hiện polyp theo pattern simplified mới nhưng vẫn đầy đủ functionality.
"""

import json
import logging
import re
from typing import Dict, Any, List
import time

from langchain.schema import HumanMessage, SystemMessage

from medical_ai_agents.agents.base_agent import BaseAgent
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.detection.yolo_tools import YOLODetectionTool
from medical_ai_agents.tools.detection.util_tools import VisualizationTool

class DetectorAgent(BaseAgent):
    """Simplified Detector Agent theo pattern mới với đầy đủ functionality."""
    
    def __init__(self, model_path: str, llm_model: str = "gpt-4o-mini", device: str = "cuda"):
        """
        Khởi tạo Detector Agent theo pattern simplified.
        
        Args:
            model_path: Đường dẫn đến YOLO model weights
            llm_model: Mô hình LLM sử dụng làm controller
            device: Device để chạy model (cuda/cpu)
        """
        self.model_path = model_path
        super().__init__(name="Detector Agent", llm_model=llm_model, device=device)
        self.detector_tool = None
        self.visualize_tool = None
    
    def _register_tools(self) -> List[BaseTool]:
        """Register tools for this agent."""
        self.detector_tool = YOLODetectionTool(
            model_path=self.model_path,
            device=self.device
        )
        self.visualize_tool = VisualizationTool()
        return [self.detector_tool, self.visualize_tool]
    
    def _get_agent_description(self) -> str:
        """Get agent description for ReAct system prompt."""
        return """Bạn là chuyên gia phát hiện polyp trong hình ảnh nội soi tiêu hóa.
        
Chuyên môn của bạn:
- Phân tích hình ảnh nội soi để phát hiện polyp
- Đánh giá vị trí, kích thước và đặc điểm của polyp
- Tạo visualization với bounding boxes để minh họa kết quả
- Cung cấp nhận định y khoa chuyên sâu về các phát hiện

Bạn sử dụng công nghệ YOLO để phát hiện chính xác và có thể tạo hình ảnh minh họa khi cần thiết."""

    def _get_system_prompt(self) -> str:
        """Get system prompt theo pattern simplified."""
        return f"""Bạn là AI chuyên phát hiện polyp trong hình ảnh nội soi.
Danh sách công cụ có sẵn:
- yolo_detection: Phát hiện polyp trong hình ảnh
- visualize_detections: Tạo hình ảnh với bounding box

Luồng làm việc:
1) Nhận đường dẫn ảnh và yêu cầu
2) Sử dụng yolo_detection để phát hiện polyp  
3) Nếu cần visualization, sử dụng visualize_detections
4) Phân tích kết quả và đưa ra nhận định chuyên môn"""

    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input theo pattern simplified."""
        image_path = task_input.get("image_path", "")
        query = task_input.get("query", "") or "Detect polyps in this endoscopy image"
        medical_context = task_input.get("medical_context", {})
        
        context_lines = []
        if medical_context:
            for key, val in medical_context.items():
                context_lines.append(f"- {key}: {val}")
        context_str = "\n".join(context_lines) if context_lines else "No medical context provided"
        
        return (
            f"**Polyp Detection Task**\n\n"
            f"Image to analyze: {image_path}\n"
            f"Query: {query}\n\n"
            f"Medical context:\n{context_str}\n\n"
            f"Tool call format:\nTool: yolo_detection\n"
            f"Parameters: {{\"image_path\": \"<path>\", \"conf_thresh\": 0.25}}\n\n"
            f"If visualization needed:\nTool: visualize_detections\n"
            f"Parameters: {{\"image_path\": \"<path>\", \"detections\": <results_from_yolo>}}"
        )

    def _format_synthesis_input(self) -> str:
        """Format synthesis input theo pattern simplified."""
        return (
            f"Hãy xuất JSON đúng định dạng:\n"
            f"```json\n"
            f"{{\n"
            f"  \"detector_result\": {{\n"
            f"    \"success\": true/false,\n"
            f"    \"count\": <số_polyp>,\n"
            f"    \"objects\": [...danh_sách_polyp...],\n"
            f"    \"analysis\": \"phân tích chuyên môn về kết quả phát hiện\",\n"
            f"    \"visualization_available\": true/false,\n"
            f"    \"visualization_base64\": \"base64_string_if_available\"\n"
            f"  }}\n"
            f"}}\n"
            f"```"
        )

    def _parse_tool_calls(self, plan: str) -> List[Dict[str, Any]]:
        """Parse tool calls theo pattern simplified của ClassifierAgent mới."""
        tool_calls = []
        lines = plan.split("\n")
        current_tool = None
        current_params = {}
        
        for idx, line in enumerate(lines):
            l = line.strip()
            if l.startswith("Tool:"):
                if current_tool:
                    tool_calls.append({"tool_name": current_tool, "params": current_params})
                current_tool = l.replace("Tool:", "").strip()
                current_params = {}
            elif l.startswith("Parameters:"):
                try:
                    ptxt = l.replace("Parameters:", "").strip()
                    if ptxt.startswith("{") and ptxt.endswith("}"):
                        current_params = json.loads(ptxt)
                    elif ptxt.startswith("{"):
                        json_str = ptxt
                        for nxt in lines[idx + 1:]:
                            json_str += " " + nxt.strip()
                            if "}" in nxt:
                                break
                        current_params = json.loads(json_str)
                except Exception:
                    current_params = {}
        
        if current_tool:
            tool_calls.append({"tool_name": current_tool, "params": current_params})
        
        # Fallback parsing như ClassifierAgent mới
        if not tool_calls:
            tool_pattern = r"Tool:\s*(yolo_detection|visualize_detections)"
            param_pattern = r"Parameters:\s*(\{[^}]*\})"
            tools = re.findall(tool_pattern, plan)
            params = re.findall(param_pattern, plan, re.DOTALL)
            for i, tool in enumerate(tools):
                param_dict = {}
                if i < len(params):
                    try:
                        param_dict = json.loads(params[i])
                    except:
                        param_dict = {}
                tool_calls.append({"tool_name": tool, "params": param_dict})
        
        return tool_calls

    def _format_agent_result(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format final agent result từ ReAct execution với đầy đủ thông tin."""
        if not react_result.get("success"):
            return {
                "detector_result": {
                    "success": False,
                    "error": react_result.get("error", "Detection failed"),
                    "reasoning_steps": len(self.react_history) if hasattr(self, 'react_history') else 0,
                }
            }
        
        # Extract detection results từ ReAct history hoặc direct result
        objects = []
        count = 0
        visualization_base64 = None
        visualization_available = False
        
        # Try to get from react_history first
        if hasattr(self, 'react_history') and self.react_history:
            for step in self.react_history:
                if step.observation and "yolo_detection" in str(step.action):
                    try:
                        obs_data = json.loads(step.observation)
                        if obs_data.get("success", False):
                            objects = obs_data.get("objects", [])
                            count = obs_data.get("count", len(objects))
                            break
                    except Exception:
                        continue
                        
                # Check for visualization result
                if step.observation and "visualize_detections" in str(step.action):
                    try:
                        obs_data = json.loads(step.observation)
                        if obs_data.get("success", False):
                            visualization_base64 = obs_data.get("visualization_base64")
                            visualization_available = True
                    except Exception:
                        continue
        
        # Build analysis from answer
        analysis = react_result.get("answer", "Completed polyp detection analysis.")
        
        result = {
            "success": True,
            "count": count,
            "objects": objects,
            "analysis": analysis,
            "visualization_available": visualization_available,
        }
        
        if visualization_base64:
            result["visualization_base64"] = visualization_base64
        
        return {"detector_result": result}

    # ===== ĐẦY ĐỦ CÁC HÀM TỪ VERSION CŨ =====
    
    def initialize(self) -> bool:
        """Khởi tạo agent và các công cụ (từ version cũ)."""
        try:
            # Tools đã được khởi tạo trong _register_tools
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize detector agent: {str(e)}")
            self.initialized = False
            return False
    
    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract task-specific input from state (từ version cũ)."""
        return {
            "image_path": state.get("image_path", ""),
            "query": state.get("query", ""),
            "medical_context": state.get("medical_context", {}),
            "user_type": state.get("user_type", "patient")
        }
    
    def _process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced process_state với intelligent tool orchestration (từ version cũ)."""
        try:
            # Ensure initialized
            if not self.initialized:
                success = self.initialize()
                if not success:
                    return {**state, "error": f"Failed to initialize {self.name}"}
            
            # Extract task input
            task_input = self._extract_task_input(state)
            self.logger.info(f"[Detector] Processing task: {json.dumps(task_input, indent=2)}")
            
            # Get LLM plan
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self._format_task_input(task_input))
            ]
            
            response = self.llm.invoke(messages)
            plan = response.content
            self.logger.info(f"[Detector] LLM plan:\n{plan}")
            
            # Parse tool calls
            tool_calls = self._parse_tool_calls(plan)
            self.logger.info(f"[Detector] Parsed tool calls: {json.dumps(tool_calls, indent=2)}")
            
            # Execute tools với intelligent orchestration từ version cũ
            results = {}
            tool_outputs = {}
            
            for i, tool_call in enumerate(tool_calls):
                tool_name = tool_call.get("tool_name")
                params = tool_call.get("params", {})
                
                # Fill in image_path nếu thiếu
                if "image_path" not in params or not params["image_path"]:
                    params["image_path"] = task_input.get("image_path", "")
                
                self.logger.info(f"[Detector] Executing tool {i+1}/{len(tool_calls)}: {tool_name}")
                
                # Special handling cho visualization như version cũ
                if tool_name == "visualize_detections":
                    # Cần detections từ yolo_detection trước đó
                    if "yolo_detection" in tool_outputs:
                        yolo_result = tool_outputs["yolo_detection"]
                        if yolo_result.get("success", False):
                            params["detections"] = yolo_result.get("objects", [])
                            self.logger.info(f"[Detector] Using {len(params['detections'])} detections for visualization")
                        else:
                            self.logger.warning("[Detector] Skipping visualization - no valid detections")
                            continue
                    else:
                        self.logger.warning("[Detector] Skipping visualization - no yolo_detection results")
                        continue
                
                # Execute tool
                tool_result = self.execute_tool(tool_name, **params)
                results[tool_name] = tool_result
                tool_outputs[tool_name] = tool_result
                
                self.logger.info(f"[Detector] Tool {tool_name} completed: {tool_result.get('success', False)}")
            
            # Synthesize results với multipart message support từ version cũ
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
            
            # Tạo multipart message với text và image nếu có visualization như version cũ
            if img_base64:
                synthesis_messages.append(
                    HumanMessage(
                        content=[
                            {"type": "text", "text": f"Used yolo and visualization tools. Tool execution results:\n{json.dumps(other_results, indent=2)}\n\n{self._format_synthesis_input()}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                        ]
                    )
                )
            else:
                # Fallback về text-only message
                synthesis_messages.append(
                    HumanMessage(content=f"Used yolo tool without visualization. Tool execution results:\n{json.dumps(results, indent=2)}\n\n{self._format_synthesis_input()}")
                )

            synthesis_response = self.llm.invoke(synthesis_messages)
            agent_result = self._extract_agent_result(synthesis_response.content)
            
            # Add visualization info nếu có như version cũ
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
    
    # ===== CÁC HÀM HELPER TỪ VERSION CŨ =====
    
    def _extract_reasoning_and_tools(self, plan: str) -> Dict[str, str]:
        """Extract REASONING and TOOL_PLAN sections (từ version cũ)."""
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
        """Parse tools from structured TOOL_PLAN format (từ version cũ)."""
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
        """Parse tools using regex patterns (từ version cũ)."""
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
        """Intelligent fallback based on LLM reasoning (từ version cũ)."""
        tool_calls = []
        
        # Always need yolo_detection
        base_tool = {
            "tool_name": "yolo_detection",
            "params": {"image_path": ""},  # Will be filled later
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
            self.logger.info(f"[Detector] Intelligent fallback decided visualization needed (score: {viz_score}, explicit: {explicit_viz})")
        
        return tool_calls
    
    def _enhance_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance tool calls with missing information (từ version cũ)."""
        for tool_call in tool_calls:
            params = tool_call.get("params", {})
            
            # Ensure image_path is present
            if "image_path" not in params or not params["image_path"]:
                params["image_path"] = "__STATE_IMAGE_PATH__"  # Placeholder
            
            # Set default confidence threshold for yolo
            if tool_call["tool_name"] == "yolo_detection" and "conf_thresh" not in params:
                params["conf_thresh"] = 0.25
            
            tool_call["params"] = params
        
        return tool_calls
    
    def _parse_key_value_params(self, params_text: str) -> Dict[str, Any]:
        """Parse parameters from key=value format (từ version cũ)."""
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