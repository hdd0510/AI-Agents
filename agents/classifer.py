#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - Classifier Agent
----------------------------------
Agent phân loại hình ảnh nội soi (kỹ thuật chụp và vị trí giải phẫu).
"""

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

from medical_ai_system.agents.base_agent import BaseAgent
from medical_ai_system.config import ClassificationResult

class ClassifierAgent(BaseAgent):
    """Agent phân loại hình ảnh nội soi."""
    
    def __init__(self, model_path: str, class_names: List[str], 
                classifier_type: str = "modality", device: str = "cuda"):
        """
        Khởi tạo Classifier Agent.
        
        Args:
            model_path: Đường dẫn đến model weights
            class_names: Danh sách tên các lớp
            classifier_type: Loại classifier ('modality' hoặc 'region')
            device: Device để chạy model (cuda/cpu)
        """
        super().__init__(name=f"{classifier_type.capitalize()} Classifier Agent", device=device)
        self.model_path = model_path
        self.class_names = class_names
        self.classifier_type = classifier_type  # 'modality' hoặc 'region'
        self.model = None
    
    def initialize(self) -> bool:
        """Load classification model."""
        try:
            from ultralytics import YOLO
            self.logger.info(f"Loading {self.classifier_type} classifier from {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to load classifier model: {str(e)}")
            self.initialized = False
            return False
    
    def _process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state to classify image."""
        if not state.get("image_path"):
            result_key = "modality_result" if self.classifier_type == "modality" else "region_result"
            return {**state, result_key: {"success": False, "error": "No image path provided"}}
        
        image_path = state["image_path"]
        image = self.load_image(image_path)
        
        if image is None:
            result_key = "modality_result" if self.classifier_type == "modality" else "region_result"
            return {**state, result_key: {"success": False, "error": "Failed to load image"}}
        
        # Run classification
        results = self.model.predict(
            source=image,
            verbose=False
        )
        
        # Process results
        top_class_name = "Unknown"
        top_confidence = 0.0
        
        for result in results:
            if hasattr(result, 'probs') and result.probs is not None:
                probs = result.probs.data.cpu().numpy()
                top_idx = probs.argmax()
                top_confidence = float(probs[top_idx])
                top_class_name = self.class_names[top_idx] if top_idx < len(self.class_names) else f"class_{top_idx}"
        
        # Create result
        classification_result: ClassificationResult = {
            "success": True,
            "class_name": top_class_name,
            "confidence": top_confidence
        }
        
        # Return updated state
        result_key = "modality_result" if self.classifier_type == "modality" else "region_result"
        return {**state, result_key: classification_result}

    def get_class_description(self) -> Dict[str, str]:
        """Get descriptions for each class."""
        if self.classifier_type == "modality":
            return {
                "WLI": "White Light Imaging - Standard visualization technique",
                "BLI": "Blue Light Imaging - Enhanced visualization of blood vessels and surface patterns",
                "FICE": "Flexible spectral Imaging Color Enhancement - Digital chromoendoscopy for mucosal assessment",
                "LCI": "Linked Color Imaging - Enhanced visualization with color contrast for lesion detection"
            }
        else:  # region
            return {
                "Hau_hong": "Pharynx - The throat region",
                "Thuc_quan": "Esophagus - Tube connecting throat to stomach",
                "Tam_vi": "Cardia - Upper stomach opening connected to esophagus",
                "Than_vi": "Body of stomach - Main part of the stomach",
                "Phinh_vi": "Fundus - Upper curved part of the stomach",
                "Hang_vi": "Antrum - Lower portion of the stomach",
                "Bo_cong_lon": "Greater curvature - Outer curved edge of the stomach",
                "Bo_cong_nho": "Lesser curvature - Inner curved edge of the stomach",
                "Hanh_ta_trang": "Duodenal bulb - First part of duodenum",
                "Ta_trang": "Duodenum - First section of small intestine"
            }   