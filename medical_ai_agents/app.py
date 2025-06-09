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
            """SIMPLIFIED streaming version - cleaned up logic."""
            import uuid
            
            # Generate session and user IDs first thing
            session_id = session_state.get("session_id")
            if not session_id:
                session_id = str(uuid.uuid4())
                session_state["session_id"] = session_id
                logger.info(f"Created new session ID: {session_id}")
            else:
                logger.info(f"Using existing session ID: {session_id}")
                
            user_id = self.generate_user_id(username)
            session_state["user_id"] = user_id
            
            # Start with user message
            history.append([message, "🤔 Analyzing..."])
            yield "", history, session_state
            
            try:
                # Get contextual information
                context = self.memory.get_contextual_prompt(session_id, user_id)
                
                # SIMPLIFIED: Determine processing mode
                has_image = image is not None
                
                if has_image:
                    # =============  IMAGE WORKFLOW (SIMPLIFIED) =============
                    logger.info(f"Processing image with query: '{message[:50]}...'")
                    
                    # Save image to temp file
                    image_path = self._save_image_to_temp(image)
                    
                    # Update status during processing
                    history[-1][1] = "🔍 Analyzing image..."
                    yield "", history, session_state
                    time.sleep(0.3)
                    
                    # Call medical AI system
                    result = self.medical_ai.analyze(
                        image_path=image_path,
                        query=message,
                        medical_context={
                            "user_context": context
                        } if context else None
                    )
                    
                    logger.debug(f"Image analysis result: success={result.get('success', False)}")
                    
                    if result.get("success", False):
                        # Only handle polyp detection if it was done
                        if "polyps" in result:
                            polyp_count = len(result.get("polyps", []))
                            
                            # Add detection info to the response message
                            if polyp_count > 0:
                                response_parts = []
                                response_parts.append("🔍 **Detection Results:**")
                                response_parts.append(f"- Found {polyp_count} polyp(s)")
                                
                                # Get confidence of first polyp
                                if result["polyps"] and "confidence" in result["polyps"][0]:
                                    confidence = result["polyps"][0]["confidence"]
                                    response_parts.append(f"- Confidence: {confidence:.1%}")
                                
                                # Check if visualization is available
                                if "detector_result" in result.get("agent_results", {}) and result["agent_results"].get("detector", {}).get("visualization_base64"):
                                    # Save visualization to temp file
                                    viz_base64 = result["agent_results"]["detector"]["visualization_base64"]
                                    
                                    # Generate a unique filename for this visualization
                                    viz_filename = f"polyp_viz_{session_state.get('session_id', 'unknown')}_{int(time.time())}.png"
                                    viz_path = self._save_visualization(viz_base64, viz_filename)
                                    
                                    # Add to session state
                                    session_state["last_visualization"] = viz_path
                                    session_state["has_visualization"] = True
                                    
                                    # Display directly in message using HTML
                                    # Create data URL
                                    img_data_url = f"data:image/png;base64,{viz_base64}"
                                    response_parts.append("\n\n📊 **Polyp Detection Results:**")
                                    response_parts.append(f'<img src="{img_data_url}" alt="Polyp Detection Results" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;">')
                            else:
                                response_parts.append("🔍 **No polyps detected in this image.**")
                                response_parts.append("Regular screening is still recommended as a preventive measure.")
                        
                        # If final answer is available, add it
                        if "final_answer" in result:
                            # Add detection results first if available
                            if "medical_context" in result:
                                context_parts = []
                                mc = result["medical_context"]
                                
                                if "imaging_type" in mc:
                                    context_parts.append(f"- Imaging technique: {mc['imaging_type']}")
                                
                                if "anatomical_region" in mc:
                                    context_parts.append(f"- Anatomical region: {mc['anatomical_region']}")
                                
                                if context_parts:
                                    response_parts.append("\n💭 **Image Analysis:**")
                                    response_parts.extend(context_parts)
                            
                            # Add LLM answer
                            response_parts.append("\n💬 **Medical AI Assessment:**")
                            response_parts.append(result["final_answer"])
                        
                        # Add medical recommendations
                        if "polyps" in result and len(result.get("polyps", [])) > 0:
                            response_parts.append("\n💡 **Medical Recommendations:**")
                            response_parts.append("- Consult with a gastroenterologist for clinical correlation")
                            response_parts.append("- Follow-up may be needed based on polyp characteristics")
                            response_parts.append("- Regular screening is important for early detection")
                        elif "vqa_answer" in result:
                            response_parts.append("\n💡 **Additional Information:**")
                            response_parts.append("- This analysis is provided for informational purposes only")
                            response_parts.append("- For specific medical advice, please consult a healthcare provider")
                        
                        # Update conversation history in session state
                        if "conversation_history" in result:
                            session_state["conversation_history"] = result["conversation_history"]
                            logger.info(f"[APP] Updated session with conversation_history: {len(result['conversation_history'])} entries")
                            if result["conversation_history"]:
                                last_entry = result["conversation_history"][-1]
                                logger.info(f"[APP] Last entry query: {last_entry.get('query', 'Unknown')[:30]}...")
                                logger.info(f"[APP] Last entry response: {last_entry.get('response', 'Unknown')[:30]}...")
                        
                        # Generate streaming response for image mode - start with a header
                        streaming_text = "🔬 **Medical Image Analysis**\n\n"
                        
                        # Add final details with poly count and tools used
                        if "final_result" in result and "agent_results" in result["final_result"] and "detector_result" in result["final_result"]["agent_results"]:
                            detector = result["final_result"]["agent_results"]["detector_result"]
                            polyp_count = detector.get("count", 0)
                            if polyp_count > 0:
                                streaming_text += f"\n\n🔍 **Detected {polyp_count} suspicious area{'s' if polyp_count > 1 else ''}**"
                                streaming_text += "\n💡 A medical professional should review these findings"
                                if "viz_image_path" in session_state:
                                    streaming_text += "\n👁️ Click 'View Results' to see visualized analysis"
                        
                        # Record polyp count in state for memory
                        polyp_count = 0
                        if "final_result" in result and "agent_results" in result["final_result"] and "detector_result" in result["final_result"]["agent_results"]:
                            polyp_count = result["final_result"]["agent_results"]["detector_result"].get("count", 0)
                        session_state["polyp_count"] = polyp_count
                        
                        streaming_text += "\n\n" + "\n".join(response_parts)
                        history[-1][1] = streaming_text
                        yield "", history, session_state
                    else:
                        error_msg = result.get("error", "Unknown error")
                        response_parts = []
                        response_parts.append(f"❌ Error analyzing the image: {error_msg}")
                        response_parts.append("Please try again or upload a different image.")
                        history[-1][1] = "\n".join(response_parts)
                        yield "", history, session_state
                else:
                    # =============  TEXT-ONLY WORKFLOW (SIMPLIFIED) =============
                    logger.info(f"Processing text-only query: '{message[:50]}...'")
                    
                    history[-1][1] = "🧠 Consulting via LLaVA..."
                    yield "", history, session_state
                    time.sleep(0.3)
                    
                    # SIMPLIFIED: Call medical AI with clean parameters
                    result = self.medical_ai.analyze(
                        image_path=None,  # No image - triggers text-only mode
                        query=message,
                        medical_context={
                            "user_context": context,
                            "is_text_only": True
                        } if context else {"is_text_only": True}
                    )
                    
                    logger.debug(f"Text-only result: success={result.get('success', False)}")
                    
                    if result.get("success", False):
                        # Stream LLaVA text-only response
                        history[-1][1] = "📝 Preparing response..."
                        yield "", history, session_state
                        time.sleep(0.3)
                        
                        # Check if VQA result is successful
                        vqa_success = True
                        if "final_result" in result and "agent_results" in result["final_result"] and "vqa_result" in result["final_result"]["agent_results"]:
                            vqa_result = result["final_result"]["agent_results"]["vqa_result"]
                            vqa_success = vqa_result.get("success", False)
                            
                            if not vqa_success:
                                # VQA/LLaVA failed - show safety error
                                error_response = "❌ **Medical advisory system unavailable**\n\n"
                                error_response += vqa_result.get("answer", "An undefined error occurred during consultation.")
                                error_response += "\n\n🔄 **Please:**\n"
                                error_response += "- Try again in a few minutes\n"
                                error_response += "- Or consult a physician directly if needed"
                                
                                history[-1][1] = error_response
                                yield "", history, session_state
                                return  # Exit early
                        
                        # VQA succeeded - process response
                        if vqa_success and "final_answer" in result:
                            streaming_text = ""
                            
                            # Add context if available
                            if context:
                                streaming_text += "💭 **Based on previous information:**\n"
                                streaming_text += (context[:200] + "..." if len(context) > 200 else context) + "\n\n"
                            
                            streaming_text += "💬 **Medical AI Response:**\n"
                            streaming_text += result["final_answer"]
                            
                            # Add processed by note
                            streaming_text += "\n\n🔬 **Processed by:** LLaVA-Med (Medical LLM)"
                            
                            history[-1][1] = streaming_text
                            yield "", history, session_state
                            
                            # Update conversation history in session state
                            if "conversation_history" in result:
                                session_state["conversation_history"] = result["conversation_history"]
                                logger.info(f"[APP] Updated session with conversation_history: {len(result['conversation_history'])} entries")
                        
                        else:
                            # Fallback for no final answer
                            streaming_text = ""
                            if context:
                                streaming_text += "💭 **Based on previous information:**\n"
                                streaming_text += (context[:200] + "..." if len(context) > 200 else context) + "\n\n"
                            
                            streaming_text += "❌ **An error occurred during response generation**\n\n"
                            streaming_text += "I apologize for the inconvenience. The Medical AI system encountered difficulty processing your query.\n\n"
                            streaming_text += "Please try again with:\n"
                            streaming_text += "- A more specific question\n"
                            streaming_text += "- Different phrasing\n"
                            streaming_text += "- Or try uploading an image for visual analysis"
                            
                            history[-1][1] = streaming_text
                            yield "", history, session_state
                    
                    else:
                        # Handle text-only system error
                        logger.error(f"Medical AI system failed: {result.get('error', 'Unknown error')}")
                        error_response = self._create_system_error_response(message)
                        history[-1][1] = error_response
                        yield "", history, session_state
                
                # Save to memory
                final_response = history[-1][1]
                
                # Safely extract polyp count if available
                polyp_count = 0
                if 'result' in locals() and "final_result" in result and "agent_results" in result["final_result"]:
                    agent_results = result["final_result"]["agent_results"]
                    if "detector_result" in agent_results:
                        polyp_count = agent_results["detector_result"].get("count", 0)
                
                # Create complete interaction record
                interaction = {
                    "query": message,
                    "response": final_response,
                    "has_image": has_image,
                    "analysis": result if 'result' in locals() else None,
                    "polyp_count": polyp_count,
                    "is_text_only": not has_image,
                    "timestamp": time.time(),
                    "session_id": session_id  # Ensure session_id is included
                }
                
                logger.debug(f"Saving interaction to memory: query='{message[:30]}...', has_image={has_image}")
                self.memory.add_to_short_term(session_id, interaction)
                
                # Save important interactions to long term
                if has_image or "polyp" in message.lower() or "medical" in message.lower():
                    logger.info(f"Saving important interaction to long-term memory for user {user_id}")
                    self.memory.save_to_long_term(user_id, session_id, interaction)
                
                # Update session state if needed
                if "is_fake_streaming_done" not in result or result["is_fake_streaming_done"]:
                    # Process final response
                    if isinstance(result, dict) and "final_answer" in result:
                        # Extract final answer
                        final_response = result["final_answer"]
                        
                        # Update conversation history in session state
                        if "conversation_history" in result:
                            session_state["conversation_history"] = result["conversation_history"]
                            logger.info(f"[APP] Updated session state with conversation_history: {len(result['conversation_history'])} entries")
                            if result["conversation_history"]:
                                last_entry = result["conversation_history"][-1]
                                logger.info(f"[APP] Last entry: {last_entry.get('query', 'Unknown')[:30]}... - {last_entry.get('response', 'Unknown')[:30]}...")
                        else:
                            # If no conversation history in result, create one from current interaction
                            if "conversation_history" not in session_state:
                                session_state["conversation_history"] = []
                            
                            # Add the current interaction to conversation history
                            user_interaction = {
                                "query": message,
                                "response": final_response,
                                "timestamp": time.time(),
                                "session_id": session_id,
                                "is_system": False  # Mark as user interaction
                            }
                            session_state["conversation_history"].append(user_interaction)
                            logger.info(f"[APP] Created new conversation history entry for query: {message[:30]}")
                        
                        # Other updates from response
                        if "session_id" in result:
                            session_state["session_id"] = result["session_id"]
                
            except Exception as e:
                import traceback
                logger.error(f"Error processing message: {str(e)}")
                logger.error(traceback.format_exc())
                error_response = self._create_system_error_response(message)
                history[-1][1] = error_response
                yield "", history, session_state
        
        def _create_system_error_response(self, message):
            """Create system error response with helpful guidance."""
            error_response = "❌ **System temporarily unavailable**\n\n"
            error_response += "I apologize for the inconvenience. The LLaVA system is currently unable to process your request.\n\n"
            error_response += "🔄 **You can try:**\n"
            error_response += "- Trying again in a few minutes\n"
            error_response += "- Rephrasing your question\n"
            error_response += "- Uploading an image for visual analysis\n\n"
            error_response += "🏥 **For urgent consultation:** Please contact a medical specialist directly."
            return error_response

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
                        
                        # Use streaming version
                        for msg, updated_history, updated_state in self.process_message_streaming(message, image, history, username, state):
                            yield msg, updated_history, updated_state, None
                        
                        # Check for visualization result
                        if "last_visualization" in updated_state:
                            yield msg, updated_history, updated_state, updated_state.get("last_visualization")
                            
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