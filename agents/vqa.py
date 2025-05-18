#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - VQA Agent
---------------------------
Agent trả lời câu hỏi dựa trên hình ảnh nội soi.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import torch
from PIL import Image

from medical_ai_system.agents.base_agent import BaseAgent
from medical_ai_system.config import VQAResult

class VQAAgent(BaseAgent):
    """Agent trả lời câu hỏi dựa trên hình ảnh nội soi."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Khởi tạo VQA Agent.
        
        Args:
            model_path: Đường dẫn đến LLaVA model
            device: Device để chạy model (cuda/cpu)
        """
        super().__init__(name="VQA Agent", device=device)
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.conv = None
    
    def initialize(self) -> bool:
        """Load LLaVA model."""
        try:
            # Import LLaVA components
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
            from llava.conversation import conv_templates
            
            # Get model name
            model_name = os.path.basename(self.model_path.rstrip('/'))
            
            # Load model
            self.logger.info(f"Loading LLaVA model from {self.model_path}")
            self.tokenizer, self.model, self.image_processor, self.context_len = \
                load_pretrained_model(self.model_path, model_name, self.device)
            
            # Set conversation template
            self.conv = conv_templates["llava_v1"].copy()
            
            self.initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load LLaVA model: {str(e)}")
            self.initialized = False
            return False
    
    def _process_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process state to answer questions about image."""
        # Check if we have an image and a query
        if not state.get("image_path"):
            return {**state, "vqa_result": {"success": False, "error": "No image path provided"}}
        
        if not state.get("query"):
            return {**state, "vqa_result": {"success": False, "error": "No query provided"}}
        
        image_path = state["image_path"]
        query = state["query"]
        
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return {**state, "vqa_result": {"success": False, "error": "Failed to load image"}}
        
        # Preprocess image
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
        image_tensor = image_tensor.to(self.device)
        
        # Create enhanced prompt with medical context
        detector_result = state.get("detector_result", {})
        modality_result = state.get("modality_result", {})
        region_result = state.get("region_result", {})
        
        # Get detection info
        detection_info = ""
        if detector_result and detector_result.get("success", False):
            objects = detector_result.get("objects", [])
            if objects:
                detection_info = f"I detected {len(objects)} polyp(s) in the image. "
                for i, obj in enumerate(objects[:3]):  # Top 3 objects
                    detection_info += f"Polyp {i+1}: {obj.get('confidence', 0):.2f} confidence, "
                    detection_info += f"location: {obj.get('position_description', 'unknown')}. "
            else:
                detection_info = "No polyps were detected in the image. "
        
        # Get modality and region info
        modality_info = ""
        if modality_result and modality_result.get("success", False):
            modality = modality_result.get("class_name", "Unknown")
            modality_info = f"The image was taken using {modality} imaging technique. "
        
        region_info = ""
        if region_result and region_result.get("success", False):
            region = region_result.get("class_name", "Unknown")
            region_info = f"The anatomical location is {region}. "
        
        # Combine all context
        context = detection_info + modality_info + region_info
        
        # Create prompt
        prompt_template = (
            "I am a medical AI assistant specialized in analyzing endoscopy images. "
            "I'll answer your question about this medical image based on what I can observe.\n\n"
            "Image context: {context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        
        prompt = prompt_template.format(context=context, question=query)
        
        # Clear conversation history
        self.conv.clear()
        
        # Add prompt to conversation
        self.conv.append_message(self.conv.roles[0], prompt)
        self.conv.append_message(self.conv.roles[1], None)
        
        # Get prompt
        prompt = self.conv.get_prompt()
        
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=512
            )
        
        # Decode output
        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        answer = outputs.strip()
        
        # Estimate confidence based on answer patterns
        confidence = self._estimate_confidence(answer)
        
        # Create result
        vqa_result: VQAResult = {
            "success": True,
            "answer": answer,
            "confidence": confidence
        }
        
        return {**state, "vqa_result": vqa_result}
    
    def _estimate_confidence(self, answer: str) -> float:
        """Estimate confidence based on answer patterns."""
        # Simple heuristic
        low_confidence_phrases = [
            "i'm not sure", "i am not sure", "unclear", "cannot determine",
            "difficult to say", "hard to tell", "cannot see", "not visible",
            "may be", "might be", "possibly", "probably", "uncertain"
        ]
        
        answer_lower = answer.lower()
        confidence = 1.0
        
        # Reduce confidence for uncertainty phrases
        for phrase in low_confidence_phrases:
            if phrase in answer_lower:
                confidence -= 0.1
                if confidence < 0.3:
                    confidence = 0.3
                    break
        
        # Reduce confidence for very short answers
        if len(answer.split()) < 10:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))