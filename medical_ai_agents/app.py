#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Chatbot Launcher - CLEANED & SIMPLIFIED
===========================
Script khởi động chatbot với logic đã được simplified.
"""
# ---- PATCH Pydantic ↔ Starlette Request -------------------------------------
from starlette.requests import Request as _StarletteRequest
from pydantic_core import core_schema

def _any_schema(*_):        # chấp mọi số đối số
    return core_schema.any_schema()

_StarletteRequest.__get_pydantic_core_schema__ = classmethod(_any_schema)
# -----------------------------------------------------------------------------
import argparse
import os
import sys
import json
import time
import logging
from pathlib import Path
import re
os.environ['GRADIO_TEMP_DIR'] = '/tmp'

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class MedicalAIConfig:
    """Cấu hình cho Medical AI Chatbot."""
    
    DEFAULT_CONFIG = {
        "app": {
            "title": "Medical AI Assistant",
            "description": "Hệ thống AI hỗ trợ phân tích hình ảnh nội soi",
            "host": "0.0.0.0",
            "port": 8000,
            "share": True,
            "debug": False
        },
        "medical_ai": {
            "device": "cuda",
            "detector_model_path": "medical_ai_agents/weights/detect_best.pt",
            "vqa_model_path": "medical_ai_agents/weights/llava-med-mistral-v1.5-7b",
            "modality_classifier_path": "medical_ai_agents/weights/modal_best.pt",
            "region_classifier_path": "medical_ai_agents/weights/location_best.pt"
        },
        "memory": {
            "db_path": "medical_ai_memory.db",
            "short_term_limit": 10,
            "enable_long_term": True,
            "auto_save_important": True
        },
        "ui": {
            "theme": "soft",
            "chat_height": 500,
            "enable_stats": True,
            "enable_history": True,
            "max_file_size": "10MB"
        },
        "security": {
            "enable_user_auth": False,
            "max_sessions": 100,
            "session_timeout": 3600
        }
    }
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> dict:
        """Load cấu hình từ file hoặc tạo mặc định."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # Merge with default config
                config = self.DEFAULT_CONFIG.copy()
                self._deep_update(config, user_config)
                return config
            except Exception as e:
                print(f"Error loading config: {e}")
                print("Using default configuration...")
        
        # Save default config
        self.save_config(self.DEFAULT_CONFIG)
        return self.DEFAULT_CONFIG.copy()
    
    def save_config(self, config: dict = None):
        """Lưu cấu hình ra file."""
        config_to_save = config or self.config
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
            print(f"Config saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default=None):
        """Get config value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

def create_enhanced_chatbot():
    """Tạo chatbot với cấu hình nâng cao."""
    
    import gradio as gr
    from medical_ai_agents.memory import MedicalAIChatbot, LongShortTermMemory
    from medical_ai_agents import MedicalGraphConfig
    
    # Load config
    config = MedicalAIConfig()
    
    class EnhancedMedicalAIChatbot(MedicalAIChatbot):
        """Chatbot với các tính năng nâng cao + streaming (SIMPLIFIED)."""
        
        def __init__(self, config: MedicalAIConfig):
            self.app_config = config
            super().__init__()
        
        def _initialize_medical_ai(self):
            """Khởi tạo Medical AI với config tùy chỉnh."""
            medical_config = MedicalGraphConfig(
                device=self.app_config.get("medical_ai.device", "cuda"),
                detector_model_path=self.app_config.get("medical_ai.detector_model_path"),
                vqa_model_path=self.app_config.get("medical_ai.vqa_model_path"),
                modality_classifier_path=self.app_config.get("medical_ai.modality_classifier_path"),
                region_classifier_path=self.app_config.get("medical_ai.region_classifier_path")
            )
            
            from medical_ai_agents import MedicalAISystem
            return MedicalAISystem(medical_config)
        
        def _save_image_to_temp(self, image) -> str:
            """Save uploaded image to temporary file."""
            import os
            import tempfile
            import time
            
            # Create temp dir if doesn't exist
            temp_dir = os.path.join(os.path.dirname(__file__), '..', 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            # Generate temp file path with timestamp
            timestamp = int(time.time())
            temp_path = os.path.join(temp_dir, f"upload_{timestamp}.jpg")
            
            # Save image
            try:
                if hasattr(image, 'save'):
                    # Pillow Image object
                    image.save(temp_path)
                elif isinstance(image, str) and (image.startswith('data:image') or os.path.isfile(image)):
                    # Data URL or file path
                    if image.startswith('data:image'):
                        import base64
                        # Extract base64 data
                        header, encoded = image.split(",", 1)
                        data = base64.b64decode(encoded)
                        with open(temp_path, "wb") as f:
                            f.write(data)
                    else:
                        # Copy existing file
                        import shutil
                        shutil.copy(image, temp_path)
                else:
                    # Unknown format
                    with open(temp_path, "wb") as f:
                        f.write(image)
                
                return temp_path
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")
                # Fallback to a tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(image if isinstance(image, bytes) else str(image).encode('utf-8'))
                    return tmp.name
        
        def _save_visualization(self, base64_data: str, filename: str) -> str:
            """Save visualization image from base64 data."""
            import os
            import base64
            
            # Create visualizations dir if doesn't exist
            viz_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            
            # Save image
            viz_path = os.path.join(viz_dir, filename)
            try:
                if base64_data.startswith('data:image'):
                    # Extract base64 data
                    header, encoded = base64_data.split(",", 1)
                    data = base64.b64decode(encoded)
                else:
                    # Assume it's already base64 encoded
                    data = base64.b64decode(base64_data)
                    
                with open(viz_path, "wb") as f:
                    f.write(data)
                    
                return viz_path
            except Exception as e:
                logger.error(f"Error saving visualization: {str(e)}")
                return ""
        
        def process_message_streaming(self, message, image, history, username, session_state):
            """FIXED streaming version - properly preserve query in conversation history."""
            import uuid
            
            # 1. CRITICAL FIX: Store original message early 
            original_message = message.strip() if message else ""
            if not original_message:
                return "", history, session_state
            
            logger.info(f"[DEBUG] Processing message: '{original_message[:50]}...'")
            
            # Generate session and user IDs
            session_id = session_state.get("session_id")
            # Log the session state to debug
            logger.info(f"[SESSION] Current session state keys: {list(session_state.keys())}")
            logger.info(f"[SESSION] Current session_id from state: {session_id}")
            
            if not session_id:
                session_id = str(uuid.uuid4())
                session_state["session_id"] = session_id
                logger.info(f"Created new session ID: {session_id}")
            else:
                logger.info(f"Reusing existing session ID: {session_id}")
            
            user_id = self.generate_user_id(username)
            session_state["user_id"] = user_id
            
            # Start with user message - FIXED: preserve original message
            history.append([original_message, "🤔 Analyzing..."])  # Use original_message
            yield "", history, session_state  # Return empty string to clear input
            
            try:
                # Get contextual information
                context = self.memory.get_contextual_prompt(session_id, user_id)
                
                # IMPORTANT: Save session_id in both global state and temporary result
                # This ensures the session_id persists through the entire process
                session_state["session_id"] = session_id
                
                # Determine processing mode
                has_image = image is not None
                
                if has_image:
                    # =============  IMAGE WORKFLOW =============
                    logger.info(f"Processing image with query: '{original_message[:50]}...'")
                    
                    # Save image to temp file
                    image_path = self._save_image_to_temp(image)
                    
                    # Update status during processing
                    history[-1][1] = "🔍 Analyzing image..."
                    yield "", history, session_state
                    time.sleep(0.3)
                    
                    # 2. CRITICAL FIX: Pass original_message to analyze, not the cleared message
                    result = self.medical_ai.analyze(
                        image_path=image_path,
                        query=original_message,  # <-- Use original_message here
                        medical_context={
                            "user_context": context
                        } if context else None,
                        conversation_history=session_state.get("conversation_history", []),  # CRITICAL: Pass conversation history
                        session_id=session_id  # CRITICAL: Pass session_id explicitly
                    )
                    
                    logger.debug(f"Image analysis result: success={result.get('success', False)}")
                    
                    if result.get("success", False):
                        # Build response (existing code...)
                        response_parts = []
                        
                        # Handle polyp detection if available
                        if "polyps" in result:
                            polyp_count = len(result.get("polyps", []))
                            
                            if polyp_count > 0:
                                # Display entry 0 response first
                                if "response" in result and isinstance(result["response"], list) and len(result["response"]) > 0:
                                    response_parts.append(f"💬 **{result['response'][0]}**\n")
                                
                                response_parts.append("🔍 **Detection Results:**")
                                response_parts.append(f"- Found {polyp_count} polyp(s)")
                                
                                if result["polyps"] and "confidence" in result["polyps"][0]:
                                    confidence = result["polyps"][0]["confidence"]
                                    response_parts.append(f"- Confidence: {confidence:.1%}")
                                
                                # Handle visualization (existing code...)
                                if "detector_result" in result.get("agent_results", {}) and result["agent_results"].get("detector", {}).get("visualization_base64"):
                                    viz_base64 = result["agent_results"]["detector"]["visualization_base64"]
                                    viz_filename = f"polyp_viz_{session_state.get('session_id', 'unknown')}_{int(time.time())}.png"
                                    viz_path = self._save_visualization(viz_base64, viz_filename)
                                    
                                    session_state["last_visualization"] = viz_path
                                    session_state["has_visualization"] = True
                                    
                                    img_data_url = f"data:image/png;base64,{viz_base64}"
                                    response_parts.append("\n\n📊 **Polyp Detection Results:**")
                                    response_parts.append(f'<img src="{img_data_url}" alt="Polyp Detection Results" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;">')
                            else:
                                # Display entry 0 response first for no polyp case
                                if "response" in result and isinstance(result["response"], list) and len(result["response"]) > 0:
                                    response_parts.append(f"💬 **{result['response'][0]}**\n")
                                response_parts.append("🔍 **No polyps detected in this image.**")
                        
                        # Add final answer if available
                        if "final_answer" in result:
                            response_parts.append("\n💬 **Medical AI Assessment:**")
                            response_parts.append(result["final_answer"])
                        
                        # 3. CRITICAL FIX: Update conversation history properly
                        if "conversation_history" in result:
                            updated_history = result["conversation_history"]
                            
                            # Ensure the last entry has the correct query
                            if updated_history and len(updated_history) > 0:
                                last_entry = updated_history[-1]
                                
                                # CRITICAL FIX: If query is empty or wrong, fix it
                                if not last_entry.get("query") or last_entry.get("query") != original_message:
                                    logger.info(f"[FIX] Correcting query in history: '{last_entry.get('query', 'EMPTY')}' -> '{original_message}'")
                                    last_entry["query"] = original_message
                            
                            session_state["conversation_history"] = updated_history
                            logger.info(f"[FIXED] Updated session with conversation_history: {len(updated_history)} entries")
                        
                        # Generate streaming response
                        streaming_text = "🔬 **Medical Image Analysis**\n\n"
                        
                        # Cập nhật header trước
                        history[-1][1] = streaming_text
                        yield "", history, session_state
                        time.sleep(0.2)
                        
                        # Thêm các phần không phải final_answer
                        non_final_parts = []
                        final_answer_part = None
                        
                        for part in response_parts:
                            if part.startswith("\n💬 **Medical AI Assessment:**"):
                                final_answer_part = part
                            else:
                                non_final_parts.append(part)
                        
                        # Cập nhật phần không phải final_answer
                        if non_final_parts:
                            current_text = streaming_text + "\n".join(non_final_parts)
                            history[-1][1] = current_text
                            yield "", history, session_state
                            time.sleep(0.2)
                            streaming_text = current_text
                        
                        # Streaming cho final_answer nếu có
                        if final_answer_part:
                            # Hiển thị tiêu đề trước
                            streaming_text += "\n💬 **Medical AI Assessment:**\n"
                            history[-1][1] = streaming_text
                            yield "", history, session_state
                            time.sleep(0.2)
                            
                            # Kiểm tra xem có streaming chunks không
                            if "response_chunks" in result and result["response_chunks"]:
                                chunks = result["response_chunks"]
                                logger.info(f"Found {len(chunks)} streaming chunks for image assessment")
                                
                                for chunk in chunks:
                                    streaming_text += chunk
                                    history[-1][1] = streaming_text
                                    yield "", history, session_state
                                    time.sleep(0.05)  # Điều chỉnh tốc độ streaming
                            else:
                                # Lấy nội dung final_answer (bỏ tiêu đề)
                                final_content = final_answer_part.replace("\n💬 **Medical AI Assessment:**\n", "")
                                
                                # Stream từng câu một
                                sentences = re.split(r'(?<=[.!?])\s+', final_content)
                                for sentence in sentences:
                                    if not sentence.strip():
                                        continue
                                    streaming_text += sentence + " "
                                    history[-1][1] = streaming_text
                                    yield "", history, session_state
                                    time.sleep(0.05)
                        else:
                            # Nếu không có final_answer thì cập nhật tất cả các phần còn lại
                            streaming_text = "🔬 **Medical Image Analysis**\n\n" + "\n".join(response_parts)
                            history[-1][1] = streaming_text
                            yield "", history, session_state
                        
                    else:
                        error_msg = result.get("error", "Unknown error")
                        response_parts = [f"❌ Error analyzing the image: {error_msg}"]
                        history[-1][1] = "\n".join(response_parts)
                        yield "", history, session_state
                        
                else:
                    # =============  TEXT-ONLY WORKFLOW =============
                    logger.info(f"Processing text-only query: '{original_message[:50]}...'")
                    
                    history[-1][1] = "🧠 Consulting via LLaVA..."
                    yield "", history, session_state
                    time.sleep(0.3)
                    
                    # 4. CRITICAL FIX: Pass original_message to analyze
                    result = self.medical_ai.analyze(
                        image_path=None,
                        query=original_message,  # <-- Use original_message here too
                        medical_context={
                            "user_context": context,
                            "is_text_only": True
                        } if context else {"is_text_only": True},
                        conversation_history=session_state.get("conversation_history", []),  # CRITICAL: Pass conversation history
                        session_id=session_id  # CRITICAL: Pass session_id explicitly
                    )
                    
                    logger.debug(f"Text-only result: success={result.get('success', False)}")
                    
                    if result.get("success", False):
                        # Check VQA result
                        vqa_success = True
                        if "final_result" in result and "agent_results" in result["final_result"] and "vqa_result" in result["final_result"]["agent_results"]:
                            vqa_result = result["final_result"]["agent_results"]["vqa_result"]
                            vqa_success = vqa_result.get("success", False)
                            
                            if not vqa_success:
                                error_response = "❌ **Medical advisory system unavailable**\n\n"
                                error_response += vqa_result.get("answer", "An undefined error occurred during consultation.")
                                history[-1][1] = error_response
                                yield "", history, session_state
                                return
                        
                        # VQA succeeded
                        if vqa_success and "final_answer" in result:
                            # Chuẩn bị phần đầu của response (không phải streaming)
                            streaming_text = ""
                            
                            if context:
                                streaming_text += "💭 **Based on previous information:**\n"
                                streaming_text += (context[:200] + "..." if len(context) > 200 else context) + "\n\n"
                            
                            streaming_text += "💬 **Medical AI Response:**\n"
                            
                            # Cập nhật phần đầu trước
                            history[-1][1] = streaming_text
                            yield "", history, session_state
                            time.sleep(0.2)
                            
                            # Thực hiện streaming từ các chunks sẵn có nếu có
                            if "response_chunks" in result and result["response_chunks"]:
                                chunks = result["response_chunks"]
                                logger.info(f"Found {len(chunks)} streaming chunks to display")
                                
                                current_text = streaming_text
                                for chunk in chunks:
                                    current_text += chunk
                                    history[-1][1] = current_text
                                    yield "", history, session_state
                                    time.sleep(0.05)  # Điều chỉnh tốc độ streaming
                            else:
                                # Fallback: Stream từng câu nếu không có chunks
                                logger.info("No streaming chunks found, falling back to sentence splitting")
                                final_answer = result["final_answer"]
                                sentences = re.split(r'(?<=[.!?])\s+', final_answer)
                                
                                # Stream từng câu một
                                current_text = streaming_text
                                for sentence in sentences:
                                    if not sentence.strip():
                                        continue
                                    current_text += sentence + " "
                                    history[-1][1] = current_text
                                    yield "", history, session_state
                                    time.sleep(0.05)  # Điều chỉnh tốc độ streaming
                            
                            # Thêm footer
                            current_text += "\n\n🔬 **Processed by:** LLaVA-Med (Medical LLM)"
                            history[-1][1] = current_text
                            yield "", history, session_state
                            
                            # 5. CRITICAL FIX: Update conversation history for text-only too
                            if "conversation_history" in result:
                                updated_history = result["conversation_history"]
                                
                                # Fix query in last entry if needed
                                if updated_history and len(updated_history) > 0:
                                    last_entry = updated_history[-1]
                                    if not last_entry.get("query") or last_entry.get("query") != original_message:
                                        logger.info(f"[FIX] Correcting text-only query in history: '{last_entry.get('query', 'EMPTY')}' -> '{original_message}'")
                                        last_entry["query"] = original_message
                                
                                session_state["conversation_history"] = updated_history
                                logger.info(f"[FIXED] Updated session with text conversation_history: {len(updated_history)} entries")
                    else:
                        # Handle text-only system error
                        logger.error(f"Medical AI system failed: {result.get('error', 'Unknown error')}")
                        error_response = self._create_system_error_response(original_message)  # Use original_message
                        history[-1][1] = error_response
                        yield "", history, session_state
                
                # 6. CRITICAL FIX: Save to memory with correct query
                final_response = history[-1][1]
                
                # Extract polyp count if available
                polyp_count = 0
                if 'result' in locals() and "final_result" in result and "agent_results" in result["final_result"]:
                    agent_results = result["final_result"]["agent_results"]
                    if "detector_result" in agent_results:
                        polyp_count = agent_results["detector_result"].get("count", 0)
                
                # Create complete interaction record with FIXED query
                interaction = {
                    "query": original_message,  # <-- CRITICAL FIX: Use original_message
                    "response": final_response,
                    "has_image": has_image,
                    "analysis": result if 'result' in locals() else None,
                    "polyp_count": polyp_count,
                    "is_text_only": not has_image,
                    "timestamp": time.time(),
                    "session_id": session_id
                }
                
                logger.debug(f"[FIXED] Saving interaction to memory: query='{original_message[:30]}...', has_image={has_image}")
                self.memory.add_to_short_term(session_id, interaction)
                
                # Ensure conversation history exists in session_state
                if "conversation_history" not in session_state:
                    logger.warning(f"conversation_history missing in session_state, initializing new one")
                    session_state["conversation_history"] = []
                
                # If conversation_history was not updated by the workflow (common with image queries)
                # We manually add the current interaction to it
                ch_updated = False
                if "conversation_history" in result:
                    ch_updated = True
                    logger.info(f"conversation_history updated by workflow with {len(result['conversation_history'])} entries")
                
                # If not updated and session contains empty list or no matching entry, add it manually
                if not ch_updated:
                    curr_history = session_state.get("conversation_history", [])
                    # Check if the current query exists in the history
                    has_matching_entry = False
                    for entry in curr_history:
                        if entry.get("query") == original_message and not entry.get("is_pending", False):
                            has_matching_entry = True
                            break
                    
                    # If no matching entry, manually add this interaction to history
                    if not has_matching_entry:
                        logger.info(f"Manually adding current interaction to conversation_history: '{original_message[:30]}...'")
                        history_entry = {
                            "query": original_message,
                            "response": final_response,
                            "timestamp": time.time(),
                            "has_image": has_image,
                            "session_id": session_id
                        }
                        # Add entry to conversation history
                        curr_history.append(history_entry)
                        session_state["conversation_history"] = curr_history
                        logger.info(f"Conversation history updated manually, now has {len(curr_history)} entries")
                
                # Debug - log final state of conversation history
                conversation_history = session_state.get("conversation_history", [])
                logger.info(f"FINAL conversation_history has {len(conversation_history)} entries")
                if conversation_history:
                    for i, entry in enumerate(conversation_history[:]):  # Show last 2 entries
                        logger.info(f"FINAL HISTORY ENTRY {i}:")
                        logger.info(f"  - QUERY: {entry.get('query', 'None')[:30]}...")
                        resp = entry.get('response', 'None')
                        resp_preview = resp[:30] + "..." if resp and len(resp) > 30 else resp
                        logger.info(f"  - RESPONSE: {resp_preview}")
                
                # Save important interactions to long term
                if has_image or "polyp" in original_message.lower() or "medical" in original_message.lower():
                    logger.info(f"Saving important interaction to long-term memory for user {user_id}")
                    self.memory.save_to_long_term(user_id, session_id, interaction)
                
            except Exception as e:
                import traceback
                logger.error(f"Error processing message: {str(e)}")
                logger.error(traceback.format_exc())
                error_response = self._create_system_error_response(original_message)  # Use original_message
                history[-1][1] = error_response
                yield "", history, session_state
        def create_enhanced_interface(self):
            """Tạo giao diện với nhiều tính năng hơn."""
            
            # Custom CSS with image support
            custom_css = """
            .main-container { 
                max-width: 1200px; 
                margin: 0 auto;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .header-section {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                margin-bottom: 1.5rem;
                text-align: center;
            }
            .chat-container { 
                height: 600px; 
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
                width: 100%;
            }
            /* FIXED: Cải thiện hiển thị ảnh trong chat */
            .chat-container img {
                max-width: 80% !important;
                max-height: 400px !important;
                border-radius: 8px !important;
                margin: 10px auto !important;
                display: block !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2) !important;
                border: 1px solid rgba(0, 0, 0, 0.1) !important;
            }
            /* Ensure images in message bubble are properly styled */
            .message img {
                max-width: 100% !important;
                height: auto !important;
                border-radius: 8px !important;
                margin: 10px 0 !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            }
            .upload-container { 
                border: 2px dashed #4CAF50; 
                padding: 20px; 
                text-align: center;
                border-radius: 10px;
                background: #f8f9fa;
                margin-bottom: 1rem;
            }
            .chat-input-container {
                display: flex;
                width: 100%;
                margin: 0.5rem 0;
                gap: 0.5rem;
            }
            .tab-container {
                margin-top: 1rem;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .button-row {
                display: flex;
                gap: 0.5rem;
                margin: 0.5rem 0;
            }
            """
            
            with gr.Blocks(
                title=self.app_config.get("app.title", "Medical AI Assistant"),
                theme=gr.themes.Soft() if self.app_config.get("ui.theme") == "soft" else gr.themes.Default(),
                css=custom_css
            ) as interface:
                
                # State management
                session_state = gr.State({})
                
                # Header
                with gr.Row(elem_classes=["header-section"]):
                    gr.Markdown(f"""
                    # 🏥 {self.app_config.get("app.title", "Medical AI Assistant")}
                    ### {self.app_config.get("app.description", "AI system for endoscopy image analysis and medical consultation")}
                    
                    **🎯 Key Features:**
                    - 🧠 **LLaVA-Med Integration**: Specialized medical AI
                    - 🔍 **Accurate Analysis**: Multi-agent AI with high reliability
                    - 💬 **Intelligent Consultation**: Support for both images and text-only queries
                    - 📊 **Streaming Response**: Real-time feedback
                    """)
                
                with gr.Row():
                    # Main chat interface in a centered column
                    with gr.Column(scale=1, min_width=800):
                        # Chat container - FIXED: Enable HTML rendering for images
                        chatbot = gr.Chatbot(
                            label="💬 Conversation with AI",
                            height=self.app_config.get("ui.chat_height", 650),
                            show_copy_button=True,
                            elem_classes=["chat-container"],
                            layout="bubble",
                            render_markdown=True,
                            sanitize_html=False,  # FIXED: Allow HTML images
                            value=[["", "👋 **Hello! I am the Medical AI Assistant, ready to help you analyze medical images or answer any health-related questions. 📸 You can upload endoscopy or X-ray images for me to analyze.💬 Or you can simply ask medical questions without providing any images..\n\n💬 Or you can ask medical questions directly without needing to upload any images."]]
                        )
                        
                        # Input and buttons in a clean layout
                        with gr.Row(elem_classes=["chat-input-container"]):
                            msg_input = gr.Textbox(
                                placeholder="💭 Describe symptoms or ask questions about the image...",
                                label="Your message",
                                scale=5,
                                lines=2
                            )
                            with gr.Column(scale=1, elem_classes=["button-row"]):
                                send_btn = gr.Button("📤 Send", variant="primary", size="lg")
                                clear_btn = gr.Button("🗑️ Clear", variant="stop", size="lg")
                        
                        # Tabs for image upload and results
                        tabs = gr.Tabs(elem_classes=["tab-container"])
                        with tabs:
                            with gr.TabItem("🖼️ Upload Image"):
                                # Advanced image upload
                                image_input = gr.Image(
                                    label="Select endoscopy image, X-ray, or other medical image",
                                    type="filepath",
                                    elem_classes=["upload-container"]
                                )
                                
                                gr.Markdown("""
                                **Note:** 
                                - Supported formats: JPG, PNG, DICOM
                                - Maximum size: 10MB
                                - Ensure clear images for best results
                                - **Detection results will appear directly in chat**
                                """)
                            
                            with gr.TabItem("📊 Detection Results"):
                                result_image = gr.Image(
                                    label="Polyp detection results",
                                    type="filepath",
                                    interactive=False
                                )
                                
                                show_latest_result_btn = gr.Button("🔄 Show latest results", variant="secondary")
                
                # Hidden state for username (required for functions)
                username_input = gr.Textbox(value="Patient", visible=False)
                user_info = gr.Textbox(value="", visible=False)
                
                # Streaming format wrapper
                streaming_format = gr.Markdown(value="", elem_id="streaming_output") 
                
                def safe_process_message_streaming(message, image, history, username, user_info, state):
                    """Safe wrapper for process_message with streaming."""
                    start_time = time.time()
                    
                    # Log the current state
                    logger.info(f"[APP] Processing message from {username}: '{message[:50]}...' (if longer)")
                    
                    # Debug session state
                    logger.info(f"[APP] Current session state keys: {list(state.keys())}")
                    logger.info(f"[APP] Has session_id: {bool('session_id' in state)}")
                    
                    if "conversation_history" in state:
                        logger.info(f"[APP] Current session state has conversation_history with {len(state['conversation_history'])} entries")
                        if len(state['conversation_history']) > 0:
                            last_entry = state['conversation_history'][-1]
                            logger.info(f"[APP] Last entry: {last_entry.get('query', 'Unknown')[:30]}... - {last_entry.get('timestamp', 'No timestamp')}")
                    else:
                        logger.info("[APP] No conversation_history in session state yet")
                        
                    # Process the message
                    try:
                        # Update user info in state
                        if "medical_context" not in state:
                            state["medical_context"] = {}
                        state["medical_context"]["user_info"] = user_info
                        
                        # Initialize updated_state to avoid UnboundLocalError
                        updated_state = state.copy()
                        
                        # Use streaming version - always pass the SAME state object to ensure session consistency
                        for msg, updated_history, returned_state in self.process_message_streaming(message, image, history, username, state):
                            # Merge returned state back into our state to maintain consistency
                            if returned_state is not None:
                                for key, value in returned_state.items():
                                    state[key] = value
                                # Ensure session_id is preserved
                                if "session_id" in returned_state:
                                    logger.info(f"[APP] Preserving session_id: {returned_state['session_id']}")
                            
                            # Log session ID after each iteration
                            if "session_id" in state:
                                logger.info(f"[APP] Current session_id after iteration: {state['session_id']}")
                            
                            yield msg, updated_history, state, None
                        
                        # Check for visualization result
                        if "last_visualization" in state:
                            yield msg, updated_history, state, state.get("last_visualization")
                            
                    except Exception as e:
                        logger.error(f"Error in safe_process_message_streaming: {str(e)}", exc_info=True)
                        error_msg = f"❌ Processing error: {str(e)}"
                        if history:
                            history[-1][1] = error_msg
                        else:
                            history.append([message, error_msg])
                        yield "", history, state, None

                def show_latest_visualization(state):
                    """Show the most recent detection visualization"""
                    try:
                        if "last_visualization" in state and state["last_visualization"]:
                            return state["last_visualization"]
                        else:
                            return None
                    except Exception as e:
                        logger.error(f"Error showing visualization: {str(e)}")
                        return None
                
                # Connect all events với streaming
                send_btn.click(
                    safe_process_message_streaming,
                    inputs=[msg_input, image_input, chatbot, username_input, user_info, session_state],
                    outputs=[msg_input, chatbot, session_state, result_image]
                )
                
                msg_input.submit(
                    safe_process_message_streaming,
                    inputs=[msg_input, image_input, chatbot, username_input, user_info, session_state],
                    outputs=[msg_input, chatbot, session_state, result_image]
                )
                
                show_latest_result_btn.click(
                    show_latest_visualization,
                    inputs=[session_state],
                    outputs=[result_image]
                ).then(
                    lambda: gr.update(selected=1),  # Luôn chuyển sang tab kết quả
                    outputs=[tabs]
                )
                
                clear_btn.click(lambda: [], outputs=[chatbot])
            
            return interface
    
    # Create and return enhanced chatbot
    return EnhancedMedicalAIChatbot(config)

def main():
    """Hàm main với argument parsing."""
    parser = argparse.ArgumentParser(description="Medical AI Chatbot Launcher")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--share", action="store_true", help="Create shareable link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # Load config
    config = MedicalAIConfig(args.config)
    
    # Override config with command line args
    if args.host:
        config.config["app"]["host"] = args.host
    if args.port:
        config.config["app"]["port"] = args.port
    if args.share:
        config.config["app"]["share"] = True
    if args.debug:
        config.config["app"]["debug"] = True
    if args.device:
        config.config["medical_ai"]["device"] = args.device
    
    print("🚀 Starting Medical AI Chatbot...")
    print(f"📍 Host: {config.get('app.host')}")
    print(f"🔌 Port: {config.get('app.port')}")
    print(f"🌐 Share: {config.get('app.share')}")
    print(f"🖥️  Device: {config.get('medical_ai.device')}")
    print(f"🎬 Features: Simplified Logic + LLaVA Integration")
    
    try:
        # Create enhanced chatbot
        chatbot = create_enhanced_chatbot()
        interface = chatbot.create_enhanced_interface()
        
        # Launch interface
        interface.launch(
            server_name=config.get("app.host"),
            server_port=config.get("app.port"),
            share=config.get("app.share"),
            debug=config.get("app.debug"),
            show_error=True,
        )
        
    except Exception as e:
        print(f"❌ Error launching chatbot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()