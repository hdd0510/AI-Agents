#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - Detector Agent
--------------------------------
Agent phát hiện polyp và đối tượng trong hình ảnh nội soi.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

from medical_ai_system.agents.base_agent import BaseAgent
from medical_ai_system.config import DetectionResult

class DetectorAgent(BaseAgent):
    """Agent phát hiện polyp trong hình ảnh nội soi."""
    
    def __init__(self, model_path: str, device: str = "cuda", confidence_threshold: float = 0.25):
        """
        Khởi tạo Detector Agent.
        
        Args:
            model_path: Đường dẫn đến YOLO model weights
            device: Device để chạy model (cuda/cpu)
            confidence_threshold: Ngưỡng confidence cho detection
        """
        super().__init__(name="Detector Agent", device=device)
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
    
    def initialize(self) -> bool:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {str(e)}")
            self.initialized = False
            return False
    
    def _process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state to detect objects in image."""
        if not state.get("image_path"):
            return {**state, "detector_result": {"success": False, "error": "No image path provided"}}
        
        image_path = state["image_path"]
        image = self.load_image(image_path)
        
        if image is None:
            return {**state, "detector_result": {"success": False, "error": "Failed to load image"}}
        
        # Run detection
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=0.45,   # IoU threshold
            max_det=100,  # Maximum detections
            verbose=False
        )
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                cls_name = result.names[cls_id]
                
                # Calculate additional metrics
                x1, y1, x2, y2 = xyxy.tolist()
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                area = width * height
                
                # Determine position in image
                img_width, img_height = image.size
                position_x = "left" if center_x < img_width/3 else ("right" if center_x > 2*img_width/3 else "center")
                position_y = "top" if center_y < img_height/3 else ("bottom" if center_y > 2*img_height/3 else "middle")
                position_description = f"{position_y} {position_x}"
                
                detection = {
                    "bbox": xyxy.tolist(),
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "width": width,
                    "height": height,
                    "center": [center_x, center_y],
                    "area": area,
                    "position_description": position_description
                }
                detections.append(detection)
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        # Create result
        detector_result: DetectionResult = {
            "success": True,
            "objects": detections,
            "count": len(detections)
        }
        
        return {**state, "detector_result": detector_result}
    
    def get_detection_description(self, detections: List[Dict[str, Any]]) -> str:
        """Get textual description of detections."""
        if not detections:
            return "No polyps or abnormalities detected in the image."
        
        description = f"Detected {len(detections)} polyp(s) in the image.\n"
        
        for i, det in enumerate(detections[:3]):  # Top 3 detections
            confidence = det.get("confidence", 0) * 100
            position = det.get("position_description", "unknown location")
            size_desc = "large" if det.get("area", 0) > 5000 else ("medium" if det.get("area", 0) > 2000 else "small")
            
            description += f"Polyp {i+1}: {size_desc} size, {confidence:.1f}% confidence, located in {position}.\n"
        
        if len(detections) > 3:
            description += f"And {len(detections) - 3} more polyp(s).\n"
        
        return description