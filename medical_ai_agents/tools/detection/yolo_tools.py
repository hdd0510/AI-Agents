#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Tools - YOLO Detection Tool
------------------------------------
Tool phát hiện polyp sử dụng YOLO.
"""


from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image

from medical_ai_agents.tools.base_tools import BaseTool

class YOLODetectionTool(BaseTool):
    """Tool thực hiện detection YOLO."""
    
    def __init__(self, model_path: str, device: str = "cuda", confidence_threshold: float = 0.25, iou_threshold: float = 0.45, **kwargs):
        """Initialize the YOLO detection tool."""
        # Initialize with name and description first
        super().__init__(
            name="yolo_detection",
            description="Phát hiện polyp và đối tượng trong hình ảnh nội soi sử dụng YOLO."
        )
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self._initialize()
    
    def _initialize(self) -> bool:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            return True
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {str(e)}")
            return False
    
    def _run(self, image_path: str, conf_thresh: Optional[float] = None, iou_thresh: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """Run YOLO detection on image with dynamic parameters."""
        if self.model is None:
            return {"success": False, "error": "YOLO model not initialized"}
        
        # Load image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)}"}
        
        # Use provided thresholds or defaults
        conf = conf_thresh if conf_thresh is not None else self.confidence_threshold
        iou = iou_thresh if iou_thresh is not None else self.iou_threshold
        
        # Optional parameters
        max_det = kwargs.get('max_det', 100)
        verbose = kwargs.get('verbose', False)
        
        self.logger.info(f"Running detection with conf={conf}, iou={iou}")
        
        # Run detection
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            max_det=max_det,
            verbose=verbose
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
        
        return {
            "success": True,
            "objects": detections,
            "count": len(detections),
            "parameters": {
                "conf_thresh": conf,
                "iou_thresh": iou
            }
        }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for the tool parameters."""
        return {
            "image_path": {
                "type": "string",
                "description": "Path to the image file to analyze"
            },
            "conf_thresh": {
                "type": "number",
                "description": "Optional confidence threshold (0-1)",
                "required": False
            },
            "iou_thresh": {
                "type": "number",
                "description": "Optional IoU threshold (0-1)",
                "required": False
            }
        }