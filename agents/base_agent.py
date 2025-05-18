#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - Base Agent
-----------------------------
Định nghĩa lớp cơ sở cho các agents trong hệ thống AI y tế.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
import os
from PIL import Image

class BaseAgent(ABC):
    """Lớp cơ sở cho tất cả các agents."""
    
    def __init__(self, name: str, device: str = "cuda"):
        """Khởi tạo Base Agent."""
        self.name = name
        self.device = device
        self.logger = logging.getLogger(f"agent.{self.name.lower().replace(' ', '_')}")
        self.model = None
        self.initialized = False
    
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
    
    @abstractmethod
    def _process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Implement in subclasses to process state and return results."""
        pass
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Make the agent callable for LangGraph."""
        return self.process(state)