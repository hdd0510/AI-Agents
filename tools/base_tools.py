#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Tools - Base Tool
-------------------------
Tool cơ sở cho hệ thống AI y tế.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

class Tool(ABC):
    """Base class for all tools that agents can use."""
    
    def __init__(self, name: str, description: str):
        """
        Khởi tạo base tool.
        
        Args:
            name: Tên của tool
            description: Mô tả chức năng của tool
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Chạy tool với các tham số được cung cấp.
        
        Args:
            **kwargs: Các tham số đầu vào
            
        Returns:
            Kết quả của tool dưới dạng dictionary
        """
        pass
    
    def get_spec(self) -> Dict[str, Any]:
        """Trả về đặc tả của tool để cung cấp cho agent."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self._get_parameters(),
            "returns": self._get_returns()
        }
    
    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """Định nghĩa schema của các tham số đầu vào."""
        pass
    
    @abstractmethod
    def _get_returns(self) -> Dict[str, Any]:
        """Định nghĩa schema của kết quả trả về."""
        pass