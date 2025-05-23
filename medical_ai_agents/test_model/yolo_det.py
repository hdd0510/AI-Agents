from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import torch

class YOLODetector:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        """Initialize YOLO detector
        
        Args:
            model_path (str): Path to YOLO model weights (.pt file)
            device (str): Device to run inference on (cuda:0, cpu, etc)
        """
        self.device = device
        self.model = YOLO(model_path)
        
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection on input image
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            List[Dict]: List of detections, each containing:
                - bbox: [x1,y1,x2,y2] coordinates
                - confidence: Detection confidence score
                - class_id: Class ID
                - class_name: Class name
        """
        # Run inference
        results = self.model(image, device=self.device)[0]
        
        detections = []
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get confidence and class
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = results.names[cls_id]
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": cls_name
            })
            
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw detection boxes on image
        
        Args:
            image (np.ndarray): Input image
            detections (List[Dict]): List of detections from detect()
            
        Returns:
            np.ndarray: Image with drawn detections
        """
        img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            conf = det["confidence"]
            cls_name = det["class_name"]
            
            # Draw box
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            
            # Draw label
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
        return img

# Example usage:
if __name__ == "__main__":
    # Initialize detector
    detector = YOLODetector("/mnt/dunghd/medical-ai-agents/medical_ai_agents/weights/detect_best.pt", device="cuda:0")
        # Load image
    image = cv2.imread("/mnt/dunghd/medical-ai-agents/medical_ai_agents/data/test.png")
    
    # Run detection
    detections = detector.detect(image)
    
    # Draw results
    result_image = detector.draw_detections(image, detections)
    
    # Save output
    cv2.imwrite("output.jpg", result_image)
