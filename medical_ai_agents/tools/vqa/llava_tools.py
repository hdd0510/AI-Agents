#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Tools - LLaVA Tool
--------------------------
Tool sử dụng LLaVA cho VQA trên hình ảnh y tế.
"""

import os
import torch
from typing import Dict, Any, Optional
from PIL import Image

from medical_ai_agents.tools.base_tools import BaseTool

class LLaVATool(BaseTool):
    """Tool sử dụng model LLaVA để trả lời câu hỏi dựa trên hình ảnh."""
    
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """Initialize LLaVA tool."""
        name = "llava_vqa"
        description = "Sử dụng LLaVA (Large Language and Vision Assistant) để trả lời câu hỏi dựa trên hình ảnh y tế."
        super().__init__(name=name, description=description)
        
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.conv = None
        self._initialize()
    
    def _initialize(self) -> bool:
        """Load LLaVA model."""
        # Import LLaVA components
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.conversation import conv_templates
        
        # Get model name
        model_name = os.path.basename(self.model_path.rstrip('/'))
        
        # Load model
        self.logger.info(f"Loading LLaVA model from {self.model_path}")
        model_base = None
        self.tokenizer, self.model, self.image_processor, self.context_len = \
            load_pretrained_model(self.model_path, model_base, model_name, device=self.device)
        
        # Ensure model is on the correct device
        self.model = self.model.to(self.device)
        
        # Set conversation template
        self.conv = conv_templates["llava_v1"].copy()
        
        return True
        

    def _run(self, image_path: str, question: str, medical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run LLaVA on the image with the given question."""
        if self.model is None:
            return {"success": False, "error": "LLaVA model not initialized"}
        
        try:
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Preprocess image
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"]
            image_tensor = image_tensor.to(self.device)
            
            # Format context
            context_str = ""
            if medical_context:
                context_str = "Medical context:\n"
                for key, value in medical_context.items():
                    context_str += f"- {key}: {value}\n"
            
            # Create prompt
            prompt_template = (
                "I am a medical AI assistant specialized in analyzing endoscopy images. "
                "I'll answer your question about this medical image based on what I can observe.\n\n"
                "{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            )
            
            prompt = prompt_template.format(context=context_str.strip(), question=question)
            
            # ✅ FIX: Use reset instead of clear
            # Reset conversation instead of clear (which doesn't exist)
            if hasattr(self.conv, 'reset'):
                self.conv.reset()
            elif hasattr(self.conv, 'messages'):
                # Manual reset
                self.conv.messages = []
            else:
                # Create new conversation instance if needed
                from llava.conversation import conv_templates
                self.conv = conv_templates["llava_v1"].copy()
            
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
            
            return {
                "success": True,
                "answer": answer,
                "confidence": confidence
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        
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
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for the tool parameters."""
        return {
            "image_path": {
                "type": "string",
                "description": "Path to the image file to analyze"
            },
            "question": {
                "type": "string",
                "description": "Question to ask about the image"
            },
            "medical_context": {
                "type": "object",
                "description": "Optional medical context information",
                "required": False
            }
        }