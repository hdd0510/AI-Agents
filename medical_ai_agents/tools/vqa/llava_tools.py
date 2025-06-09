#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Tools - LLaVA Tool (SIMPLE FIX: Only "query" parameter)
--------------------------
Tool sử dụng LLaVA - đã được simplified với consistent parameter naming.
"""

import os
import torch
from typing import Dict, Any, Optional
from PIL import Image
import requests
from io import BytesIO
import numpy as np

from medical_ai_agents.tools.base_tools import BaseTool

class LLaVATool(BaseTool):
    """Tool sử dụng model LLaVA - simplified với consistent parameter naming."""
    
    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """Initialize LLaVA tool."""
        name = "llava_vqa"
        description = "Sử dụng LLaVA để trả lời câu hỏi y tế - hỗ trợ cả image+text và text-only queries."
        super().__init__(name=name, description=description)
        
        self.model_path = model_path
        self.device = device
        self.model_base = None
        self.load_8bit = False
        self.load_4bit = False
        self.temperature = 0.1
        self.max_new_tokens = 1024
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.context_len = None
        self.conv_mode = None
        
        self._initialize()
    
    def _initialize(self) -> bool:
        """Load LLaVA model using exact CLI logic."""
        try:
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
            from llava.mm_utils import get_model_name_from_path
            
            # Disable torch init như CLI
            disable_torch_init()
            
            # Get model name
            model_name = get_model_name_from_path(self.model_path)
            
            self.logger.info(f"Loading LLaVA model: {model_name} from {self.model_path}")
            
            # Load model với exact same parameters như CLI
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                self.model_path, 
                self.model_base, 
                model_name, 
                self.load_8bit, 
                self.load_4bit, 
                device=self.device
            )
            
            # Determine conversation mode như CLI
            if "llama-2" in model_name.lower():
                self.conv_mode = "llava_llama_2"
            elif "mistral" in model_name.lower():
                self.conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                self.conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                self.conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                self.conv_mode = "mpt"
            else:
                self.conv_mode = "llava_v0"
            
            self.logger.info(f"Using conversation mode: {self.conv_mode}")
            
            # After loading the model and before using it, ensure mm_use_im_start_end is True
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'mm_use_im_start_end'):
                if not self.model.config.mm_use_im_start_end:
                    self.model.config.mm_use_im_start_end = True
            
            return True
            
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to initialize LLaVA: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _load_image(self, image_file: str) -> Image.Image:
        """Load image exactly like CLI version."""
        self.logger.info(f"Loading image: {image_file}")
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    def _create_placeholder_image(self) -> Image.Image:
        """Create a simple placeholder image for text-only queries."""
        # Create a simple white image với medical text
        image = Image.new('RGB', (512, 512), color=(255, 255, 255))
        return image

    def _run(self, query: str, image_path: Optional[str] = None, medical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run LLaVA on image+text or text-only query."""
        if self.model is None:
            return {"success": False, "error": "LLaVA model not initialized"}
        
        if not query or not query.strip():
            return {"success": False, "error": "Query parameter is required"}
        
        try:
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import process_images, tokenizer_image_token
            
            # Enhanced image path validation and debugging
            self.logger.info(f"LLaVA tool called with query: '{query[:50]}...' and image_path: {image_path}")
            
            # Check both the explicit text_only flag and image path
            is_text_only = medical_context is not None and medical_context.get("is_text_only", False)
            
            # More detailed verification of image path
            if image_path:
                # Check if image file exists
                if os.path.exists(image_path):
                    file_size = os.path.getsize(image_path)
                    self.logger.info(f"Image file exists. Size: {file_size} bytes")
                    has_image = not is_text_only  # Use image if not explicitly text-only
                else:
                    self.logger.warning(f"Image path provided but file does not exist: {image_path}")
                    has_image = False
            else:
                self.logger.info("No image path provided")
                has_image = False
            
            # Log final decision on image usage
            self.logger.info(f"Processing query as {'image+text' if has_image else 'text-only'}")
            
            if has_image:
                # Load real image
                image = self._load_image(image_path)
                self.logger.info(f"Image loaded successfully. Size: {image.size}")
            else:
                # Create placeholder for text-only
                image = self._create_placeholder_image()
                self.logger.info("Using placeholder image for text-only query")
            
            image_size = image.size
            
            # Process images exactly like CLI
            image_tensor = process_images([image], self.image_processor, self.model.config)
            if type(image_tensor) is list:
                image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            
            # Setup conversation template
            conv = conv_templates[self.conv_mode].copy()
            if "mpt" in self.conv_mode.lower():
                roles = ('user', 'assistant')
            else:
                roles = conv.roles
            
            # Build prompt with medical context
            context_str = ""
            if medical_context:
                context_str = "Medical context:\n"
                for key, value in medical_context.items():
                    context_str += f"- {key}: {value}\n"
                context_str += "\n"
            
            # Different prompts for text-only vs image-based
            if has_image:
                # Image-based query
                inp = context_str + f"Please analyze this medical image and answer: {query}"
            else:
                # Text-only query - instruct LLaVA to use medical knowledge
                inp = context_str + f"""
You are a gastroenterology specialist with extensive medical knowledge. Please provide professional medical consultation for the following question (no image analysis needed, use your medical expertise):

Question: {query}

Please provide:
1. Professional medical assessment
2. Possible causes/explanations
3. Recommendations (examinations, lifestyle, when to seek urgent care)
4. Important notes and disclaimers

Remember to recommend direct medical consultation when appropriate and avoid definitive diagnosis through chat.
"""
            
            # Add image token for consistency (LLaVA expects it)
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            
            # Build conversation
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize using CLI method
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
            
            # Generate with exact CLI parameters
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[image_size],
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True
                )
            
            # Decode output exactly like CLI
            outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            
            # Extract only the response part (remove the original prompt)
            if conv.roles[1] + ":" in outputs:
                answer = outputs.split(conv.roles[1] + ":")[-1].strip()
            else:
                # Fallback: try to extract after the last role marker
                answer = outputs
                for role in conv.roles:
                    if role in answer:
                        parts = answer.split(role)
                        if len(parts) > 1:
                            answer = parts[-1].strip()
            
            # Clean up common artifacts
            answer = answer.replace("</s>", "").strip()
            
            return {
                "success": True,
                "answer": answer,
                "has_image": has_image,
                "query_type": "image_based" if has_image else "text_only"
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
            "query": {
                "type": "string",
                "description": "Medical question or consultation request",
                "required": True
            },
            "image_path": {
                "type": "string",
                "description": "Path to the image file (optional - if not provided, will do text-only consultation)",
                "required": False
            },
            "medical_context": {
                "type": "object",
                "description": "Optional medical context information from other agents",
                "required": False
            }
        }