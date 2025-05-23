#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - Base Agent
-----------------------------
Định nghĩa lớp cơ sở mới cho các agents trong hệ thống AI y tế.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
import logging
import os
from PIL import Image
import json

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from medical_ai_agents.tools.base_tools import BaseTool

class BaseAgent(ABC):
    """Lớp cơ sở cho tất cả các agents với LLM controller."""
    
    def __init__(self, name: str, llm_model: str = "gpt-4", device: str = "cuda"):
        """Khởi tạo Base Agent với LLM controller."""
        self.name = name
        self.device = device
        self.logger = logging.getLogger(f"agent.{self.name.lower().replace(' ', '_')}")
        self.initialized = False
        
        # Initialize LLM controller
        self.llm = ChatOpenAI(model=llm_model, temperature=0.1)
        self.tools = self._register_tools()
        self.system_prompt = self._get_system_prompt()
    
    @abstractmethod
    def _register_tools(self) -> List[BaseTool]:
        """Register tools available to this agent."""
        pass
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines this agent's role."""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """Khởi tạo agent, load model và các tài nguyên cần thiết."""
        pass
    
    def load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load hình ảnh từ đường dẫn."""
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image not found: {image_path}")
                return None
            
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            self.logger.error(f"Failed to load image: {str(e)}")
            return None
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a specific tool by name."""
        self.logger.info(f"Executing tool: {tool_name} with parameters: {json.dumps(kwargs, indent=2)}")
        
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    result = tool(**kwargs)
                    self.logger.info(f"Tool {tool_name} execution result: {json.dumps(result, indent=2)}")
                    return result
                except Exception as e:
                    self.logger.error(f"Error executing tool {tool_name}: {str(e)}")
                    return {"success": False, "error": f"Tool execution failed: {str(e)}"}
        
        return {"success": False, "error": f"Tool '{tool_name}' not found"}
    
    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Xử lý state và trả về state mới với kết quả của agent."""
        try:
            # Ensure initialized
            if not self.initialized:
                success = self.initialize()
                if not success:
                    return {**state, "error": f"Failed to initialize {self.name}"}
            
            # Process state
            return self._process_state(state)
            
        except Exception as e:
            import traceback
            error_msg = f"Error in {self.name}: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return {**state, "error": error_msg}
    
    def _process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state using LLM controller and tools."""
        # Extract relevant information from state
        task_input = self._extract_task_input(state)
        self.logger.info(f"Task input: {json.dumps(task_input, indent=2)}")
        
        # Let LLM decide which tools to use and how
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._format_task_input(task_input))
        ]
        
        # Get response from LLM
        response = self.llm.invoke(messages)
        plan = response.content
        self.logger.info(f"LLM plan: {plan}")
        
        # Parse the plan and execute tools
        tool_calls = self._parse_tool_calls(plan)
        self.logger.info(f"Parsed tool calls: {json.dumps(tool_calls, indent=2)}")
        
        results = {}
        tool_outputs = {}  # Store outputs from each tool
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool_name")
            params = tool_call.get("params", {})
            
            if tool_name:
                self.logger.info(f"Processing tool call: {tool_name}")
                self.logger.info(f"Initial parameters: {json.dumps(params, indent=2)}")
                
                # If this is visualize_detections and we have yolo_detection results
                if tool_name == "visualize_detections" and "yolo_detection" in tool_outputs:
                    yolo_result = tool_outputs["yolo_detection"]
                    self.logger.info(f"Found yolo_detection results: {json.dumps(yolo_result, indent=2)}")
                    
                    if yolo_result.get("success", False):
                        # Use the detections from yolo_detection
                        params["detections"] = yolo_result.get("objects", [])
                        self.logger.info(f"Updated parameters with detections: {json.dumps(params, indent=2)}")
                    else:
                        self.logger.warning("yolo_detection did not succeed, skipping detections")
                
                # Execute the tool
                tool_result = self.execute_tool(tool_name, **params)
                results[tool_name] = tool_result
                tool_outputs[tool_name] = tool_result
                self.logger.info(f"Tool {tool_name} completed with result: {json.dumps(tool_result, indent=2)}")
        
        # Let LLM synthesize final result
        messages.append(HumanMessage(content=f"Tool results: {json.dumps(results, indent=2)}\n\nPlease synthesize a final result."))
        synthesis_response = self.llm.invoke(messages)
        
        # Return agent result
        agent_result = self._extract_agent_result(synthesis_response.content)
        return {**state, **agent_result}
    
    @abstractmethod
    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract task-specific input from state."""
        pass
    
    @abstractmethod
    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for LLM prompt."""
        pass
    
    def _parse_tool_calls(self, plan: str) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response."""
        tool_calls = []
        
        # Simple parsing logic - extract lines with "Tool: <tool_name>" and "Parameters: <json>"
        lines = plan.split("\n")
        current_tool = None
        current_params = {}
        
        for line in lines:
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
                except Exception:
                    self.logger.warning(f"Failed to parse parameters: {line}")
        
        # Add last tool
        if current_tool:
            tool_calls.append({
                "tool_name": current_tool,
                "params": current_params
            })
        
        return tool_calls
    
    @abstractmethod
    def _extract_agent_result(self, synthesis: str) -> Dict[str, Any]:
        """Extract agent result from LLM synthesis."""
        pass
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make the agent callable for LangGraph."""
        return self.process(state)