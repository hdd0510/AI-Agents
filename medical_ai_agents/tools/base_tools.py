#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Tools - Base Tool
-------------------------
Tool cơ sở cho tất cả các công cụ trong hệ thống AI y tế.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel
import logging

class BaseTool(BaseModel):
    """Base class for all tools that agents can use."""
    name: str
    description: str
    
    def __init__(self, **data):
        super().__init__(**data)
        self.logger = logging.getLogger(f"tool.{self.name}")
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given parameters."""
        try:
            return self._run(**kwargs)
        except Exception as e:
            import traceback
            self.logger.error(f"Error executing {self.name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    @abstractmethod
    def _run(self, **kwargs) -> Dict[str, Any]:
        """Implement in subclasses to run the tool."""
        pass
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for the tool parameters."""
        return {}