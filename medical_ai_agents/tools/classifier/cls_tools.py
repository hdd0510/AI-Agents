#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Tools - Classifier Tool
-------------------------------
Tool phân loại hình ảnh nội soi (kỹ thuật chụp hoặc vị trí giải phẫu).
"""


from typing import Dict, Any, List, Optional
from PIL import Image

from medical_ai_agents.tools.base_tools import BaseTool

class ClassifierTool(BaseTool):
    """Tool phân loại hình ảnh nội soi."""
    
    def __init__(self, model_path: str, class_names: List[str], 
                classifier_type: str = "modality", device: str = "cuda", **kwargs):
        """Initialize the image classifier tool."""
        # Set specific name and description based on classifier type
        name = f"{classifier_type}_classifier"
        description = f"Phân loại hình ảnh nội soi theo {classifier_type} (kỹ thuật chụp hoặc vị trí giải phẫu)."
        
        super().__init__(name=name, description=description)
        self.model_path = model_path
        self.class_names = class_names
        self.classifier_type = classifier_type
        self.device = device
        self.model = None
        self._initialize()
    
    def _initialize(self) -> bool:
        """Load classification model."""
        try:
            from ultralytics import YOLO
            self.logger.info(f"Loading {self.classifier_type} classifier from {self.model_path}")
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # Adjust class names order for modality classifier
            if self.classifier_type == "modality":
                # Reorder class names to match actual model output
                # Original order: ["WLI", "BLI", "FICE", "LCI"]
                # Correct order based on model output: ["BLI", "FICE", "LCI", "WLI"]
                correct_order = ["BLI", "FICE", "LCI", "WLI"]
                self.class_names = [name for name in correct_order if name in self.class_names]
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load classifier model: {str(e)}")
            return False
    
    def _run(self, image_path: str) -> Dict[str, Any]:
        """Run classification on image."""
        if self.model is None:
            return {"success": False, "error": "Classifier model not initialized"}
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"success": False, "error": f"Failed to load image: {str(e)}"}
        
        # Run classification
        results = self.model.predict(source=image, verbose=False)
        
        # Process results
        top_class_name = "Unknown"
        top_confidence = 0.0
        all_classes = {}
        
        for result in results:
            if hasattr(result, 'probs') and result.probs is not None:
                probs = result.probs.data.cpu().numpy()
                
                # Get all class probabilities
                for i, prob in enumerate(probs):
                    if i < len(self.class_names):
                        all_classes[self.class_names[i]] = float(prob)
                
                # Get top class
                top_idx = probs.argmax()
                top_confidence = float(probs[top_idx])
                top_class_name = self.class_names[top_idx] if top_idx < len(self.class_names) else f"class_{top_idx}"
        
        # Get class description
        class_descriptions = self.get_class_description()
        class_description = class_descriptions.get(top_class_name, "No description available")
        
        return {
            "success": True,
            "class_name": top_class_name,
            "confidence": top_confidence,
            "description": class_description,
            "all_classes": all_classes
        }
    
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
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for the tool parameters."""
        return {
            "image_path": {
                "type": "string",
                "description": "Path to the image file to classify"
            }
        }