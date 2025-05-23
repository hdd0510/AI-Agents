#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Tools - Visualization Tool
----------------------------------
Tool để visualize các detection với bounding box.
"""

import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from PIL import Image

from medical_ai_agents.tools.base_tools import BaseTool

class VisualizationTool(BaseTool):
    """Tool để visualize các detection với bounding box."""
    
    def __init__(self, **kwargs):
        """Initialize the visualization tool."""
        super().__init__(
            name="visualize_detections",
            description="Tạo hình ảnh visualization các polyp được phát hiện với bounding box và nhãn."
        )
    
    def _run(self, image_path: str, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Visualize detections on image."""
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            img_width, img_height = image.size
            
            # Create figure and axis
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(np.array(image))
            
            # Add bounding boxes
            for i, det in enumerate(detections):
                bbox = det.get("bbox", [0, 0, 0, 0])
                conf = det.get("confidence", 0)
                class_name = det.get("class_name", "unknown")
                
                # Create rectangle
                rect = plt.Rectangle((bbox[0], bbox[1]), 
                                     bbox[2] - bbox[0], 
                                     bbox[3] - bbox[1], 
                                     linewidth=2, 
                                     edgecolor='r', 
                                     facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                label = f"{class_name}: {conf:.2f}"
                plt.text(bbox[0], bbox[1] - 10, label, 
                         color='white', 
                         bbox=dict(facecolor='red', alpha=0.8))
            
            # Add title with detection count
            plt.title(f"Detected {len(detections)} polyp(s)")
            
            # Remove axis
            plt.axis('off')
            
            # Save figure to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            # Encode as base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close(fig)
            
            return {
                "success": True,
                "visualization_base64": img_str,
                "count": len(detections)
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for the tool parameters."""
        return {
            "image_path": {
                "type": "string",
                "description": "Path to the image file to visualize"
            },
            "detections": {
                "type": "array",
                "description": "List of detection objects with bbox and other info",
                "items": {
                    "type": "object"
                }
            }
        }