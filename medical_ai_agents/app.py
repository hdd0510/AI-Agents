#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Assistant
----------------
Enhanced interactive chatbot with multi-modal capabilities.
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
import uuid
import logging
import argparse
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import re
import warnings
from datetime import datetime

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bỏ qua warning từ FAISS AVX2
warnings.filterwarnings("ignore", message=".*Could not load library with AVX2 support.*")
logging.getLogger("faiss.loader").setLevel(logging.ERROR)

# Thêm thư mục gốc của project vào sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the Medical AI System
from medical_ai_agents import MedicalAISystem, MedicalGraphConfig
from medical_ai_agents.memory.long_short_memory import LongShortTermMemory, MedicalAIChatbot

os.environ['GRADIO_TEMP_DIR'] = '/tmp'

# ---- PATCH Pydantic ↔ Starlette Request -------------------------------------
from starlette.requests import Request as _StarletteRequest
from pydantic_core import core_schema

def _any_schema(*_):        # chấp mọi số đối số
    return core_schema.any_schema()

_StarletteRequest.__get_pydantic_core_schema__ = classmethod(_any_schema)
# -----------------------------------------------------------------------------

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
        """Get a config value by key path."""
        keys = key_path.split(".")
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

def create_enhanced_chatbot():
    """Create an enhanced chatbot with a more visually appealing interface."""
    import gradio as gr
    
    # Get logger
    logger = logging.getLogger(__name__)
    
    # Import components
    from medical_ai_agents import MedicalAISystem, MedicalGraphConfig
    
    # Load config
    config = MedicalAIConfig("config.json")
    logger.info(f"Loaded config, device: {config.get('medical_ai.device')}")
    
    class EnhancedMedicalAIChatbot(MedicalAIChatbot):
        """Chatbot với các tính năng nâng cao + streaming (SIMPLIFIED)."""
        
        def __init__(self, config: MedicalAIConfig):
            self.app_config = config
            self.memory = LongShortTermMemory()
            self.medical_ai = self._initialize_medical_ai()

        
        def _sync_ui_history_with_conversation(self, history, conversation_history):
            """Đồng bộ hóa history UI với conversation_history để đảm bảo chúng khớp nhau."""
            if not conversation_history:
                return history
                
            # Tạo một bản sao của history để không thay đổi trực tiếp
            new_history = history.copy() if history else []
            
            # Tạo set các query đã có trong UI history
            existing_queries = set()
            for msg_pair in new_history:
                if len(msg_pair) >= 2:
                    existing_queries.add(msg_pair[0])
            
            # Thêm các tin nhắn từ conversation_history vào UI history nếu chưa có
            for entry in conversation_history:
                query = entry.get("query")
                response = entry.get("response")
                
                if query and response and query not in existing_queries:
                    # Thêm vào history UI
                    new_history.insert(0, [query, response])
                    existing_queries.add(query)
            
            return new_history

        def create_enhanced_interface(self):
            """Create an enhanced user interface for the chatbot."""
            import gradio as gr
            
            # Configuration
            theme = self.app_config.get("ui.theme", "soft")
            chat_height = self.app_config.get("ui.chat_height", 500)
            
            # Create interface with enhanced styling
            with gr.Blocks(theme=theme, title=self.app_config.get("app.title", "Medical AI Assistant")) as interface:
                # Header
                with gr.Row():
                    gr.HTML("""
                    <div style="text-align: center; margin-bottom: 5px; display:flex; align-items:center; justify-content:center; gap:10px;">
                        <h1 style="margin: 10px 0;">🩺 Medical AI Assistant</h1>
                    </div>
                    <p style="text-align: center; margin-bottom: 10px; color: #666;">
                        The AI-powered system for medical image analysis and consultation
                    </p>
                    """)
                
                # Username input (for session persistence)
                with gr.Row():
                    username = gr.Textbox(
                        label="Username (optional, for session persistence)",
                        placeholder="Enter username to maintain your chat history",
                        value="default_user"
                    )
                
                # Main chatbot interface
                with gr.Row():
                    chatbot = gr.Chatbot(
                        height=chat_height,
                        show_copy_button=True,
                        render=True
                    )
                    
                # Input components
                with gr.Row():
                    with gr.Column(scale=4):
                        with gr.Row():
                            msg = gr.Textbox(
                                show_label=False,
                                placeholder="Nhập câu hỏi về hình ảnh y tế hoặc hỏi tôi về các vấn đề y khoa...",
                                container=False,
                                scale=5
                            )
                            # Thêm nút đính kèm tài liệu
                            attach_btn = gr.Button("📎", size="sm", scale=1)
                    
                    # Vùng upload tài liệu (ẩn ban đầu)
                    doc_upload = gr.File(
                        label="📄 Đính kèm tài liệu (PDF, DOCX, TXT)",
                        file_types=["pdf", "docx", "txt"],
                        visible=False,
                        elem_id="document_upload"
                    )
                    
                    with gr.Column(scale=1, min_width=100):
                        with gr.Row():
                            image = gr.UploadButton(
                                "📷 Ảnh nội soi",
                                file_types=["image"],
                                type="filepath"
                            )
                        with gr.Row():
                            # Hiển thị trạng thái ảnh - đổi thành gr.Markdown và bỏ tham số scale
                            image_status = gr.Markdown(
                                "**Chưa có ảnh**"
                            )
                            # Nút xóa ảnh riêng biệt
                            clear_img_btn = gr.Button("❌ Xóa", size="sm", scale=1)
                    
                    with gr.Column(scale=1, min_width=100):
                        submit = gr.Button("Submit", variant="primary")
                
                # State for session management
                session_state = gr.State({})
                # Thêm state riêng để theo dõi ảnh hiện tại
                image_state = gr.State(None)
                
                # Auto-sync function to load history when interface is first loaded
                def auto_sync_history(username):
                    # Khởi tạo session_state mới
                    new_state = {}
                    
                    # Thử tải session ID từ persistent storage
                    session_id = self._load_persistent_session_id(username)
                    if session_id:
                        new_state["session_id"] = session_id
                        logger.info(f"Auto-sync: Loaded session_id {session_id} for {username}")
                        
                        # Tạo user ID từ username
                        user_id = self.generate_user_id(username)
                        new_state["user_id"] = user_id
                        
                        # CRITICAL FIX: Tải conversation_history và đảm bảo nó được lưu vào session state
                        conversation_history = self._load_conversation_history(session_id)
                        if conversation_history:
                            new_state["conversation_history"] = conversation_history
                            logger.info(f"Auto-sync: Loaded {len(conversation_history)} entries to conversation_history")
                            
                            # Tải lịch sử UI
                            ui_history = self.load_previous_session(username, session_id)
                            if not ui_history:
                                # Nếu không có UI history, tạo từ conversation_history
                                ui_history = []
                                for entry in conversation_history:
                                    query = entry.get("query")
                                    response = entry.get("response")
                                    if query and response:
                                        ui_history.append([query, response])
                                logger.info(f"Auto-sync: Created {len(ui_history)} UI history entries from conversation_history")
                            
                            # CRITICAL FIX: Ghi log để kiểm tra conversation_history đã được load
                            logger.info(f"Auto-sync: Memory setup complete. Session has conversation_history with {len(conversation_history)} entries")
                            
                            return ui_history, new_state
                    
                    # Nếu không tìm thấy session hoặc history
                    return [], new_state
                
                # Function to update image status với màu sắc tương phản rõ ràng
                def update_image_status(image_path, current_state):
                    # Lưu đường dẫn ảnh vào state
                    new_state = image_path
                    
                    if image_path:
                        file_name = os.path.basename(image_path)
                        # Dùng markdown với màu chữ đậm, dễ nhìn
                        return f"**✅ Đã tải: <span style='color: #2c7a4e; background: #e3f5ed; padding: 2px 5px; border-radius: 3px;'>{file_name}</span>**", new_state
                    return "**Chưa có ảnh**", None

                # Xử lý sự kiện xóa ảnh
                def clear_image(current_state):
                    # Reset cả image và state
                    return None, "**Chưa có ảnh**", None
                
                # Kết nối events
                image.upload(
                    fn=update_image_status,
                    inputs=[image, image_state],
                    outputs=[image_status, image_state],
                    queue=False
                )
                
                clear_img_btn.click(
                    fn=clear_image,
                    inputs=[image_state],
                    outputs=[image, image_status, image_state]
                )
                
                # Handle document upload visibility toggle
                def toggle_doc_upload(visible):
                    return gr.update(visible=not visible)
                
                attach_btn.click(
                    fn=toggle_doc_upload,
                    inputs=[doc_upload],
                    outputs=[doc_upload]
                )
                
                # Handle document upload status
                def update_doc_status(file):
                    if file:
                        return f"📄 Đã đính kèm: {file.name}"
                    return ""
                
                doc_upload.change(
                    fn=update_doc_status,
                    inputs=[doc_upload],
                    outputs=[msg]
                )
                
                # Events
                submit.click(
                    fn=self.process_message_streaming,
                    inputs=[msg, image, chatbot, username, session_state],
                    outputs=[msg, chatbot, session_state],
                    queue=True
                )
                msg.submit(
                    fn=self.process_message_streaming,
                    inputs=[msg, image, chatbot, username, session_state],
                    outputs=[msg, chatbot, session_state],
                    queue=True
                )
                
                # Auto-sync on page load
                interface.load(
                    fn=auto_sync_history,
                    inputs=[username],
                    outputs=[chatbot, session_state]
                )
                
                # Clear button
                with gr.Row():
                    # Fix: Thêm session_state vào danh sách để clear
                    clear_btn = gr.ClearButton([msg, chatbot, image, image_status, image_state], value="Clear Chat")
                    
                    # Thêm nút để đồng bộ hóa lịch sử
                    sync_history_btn = gr.Button("🔄 Sync History", variant="secondary")
                
                # Cập nhật hàm clear_handler để xử lý image_state
                def clear_handler():
                    # Xóa dữ liệu session_state, conversation_history và file lịch sử nếu có
                    try:
                        session_id = self._get_session_value(session_state, "session_id")
                        username_val = username.value if hasattr(username, 'value') else None
                        user_id = self._get_session_value(session_state, "user_id")
                        
                        # Xóa conversation_history trong session_state và tạo session mới
                        # Thay vì truy cập trực tiếp, sử dụng _update_session_state
                        new_session_id = str(uuid.uuid4())
                        session_state = self._update_session_state(session_state, {
                            "session_id": new_session_id,
                            "conversation_history": [],
                            "user_id": user_id  # Giữ nguyên user_id
                        })
                        logger.info(f"Created new session ID: {new_session_id}")
                        
                        # Xóa file history nếu có session_id cũ
                        if session_id:
                            import os
                            import shutil
                            
                            # Xóa file history UI
                            history_file = os.path.join("sessions", "history", f"{session_id}.json")
                            if os.path.exists(history_file):
                                try:
                                    os.remove(history_file)
                                    logger.info(f"Removed history file: {history_file}")
                                except Exception as e:
                                    logger.error(f"Failed to remove history file: {e}")
                                    
                            # Xóa file conversation_history
                            conv_file = os.path.join("sessions", "conversation_history", f"{session_id}.json")
                            if os.path.exists(conv_file):
                                try:
                                    os.remove(conv_file)
                                    logger.info(f"Removed conversation history file: {conv_file}")
                                except Exception as e:
                                    logger.error(f"Failed to remove conversation history file: {e}")
                            
                            # Xóa persistent session ID
                            if username_val:
                                persistent_file = os.path.join("sessions", f"{username_val}.session")
                                if os.path.exists(persistent_file):
                                    try:
                                        os.remove(persistent_file)
                                        logger.info(f"Removed persistent session file: {persistent_file}")
                                    except Exception as e:
                                        logger.error(f"Failed to remove persistent session file: {e}")
                                
                                # Xóa file persistent session trong thư mục data
                                if hasattr(self, "_get_persistent_session_path"):
                                    persistent_path = self._get_persistent_session_path(username_val)
                                    if os.path.exists(persistent_path):
                                        try:
                                            os.remove(persistent_path)
                                            logger.info(f"Removed persistent session path: {persistent_path}")
                                        except Exception as e:
                                            logger.error(f"Failed to remove persistent session path: {e}")
                            
                            # Xóa dữ liệu trong bộ nhớ
                            if hasattr(self, "memory"):
                                try:
                                    # Xóa short-term memory
                                    if hasattr(self.memory, "clear_short_term") and session_id:
                                        self.memory.clear_short_term(session_id)
                                        logger.info(f"Cleared short-term memory for session: {session_id}")
                                    
                                    # Xóa long-term memory cho user này
                                    if hasattr(self.memory, "clear_long_term") and user_id:
                                        self.memory.clear_long_term(user_id, session_id)
                                        logger.info(f"Cleared long-term memory for user: {user_id}")
                                except Exception as e:
                                    logger.error(f"Error clearing memory: {e}")
                            
                    except Exception as e:
                        logger.error(f"Error clearing chat data: {e}")
                        # Tạo session mới ngay cả khi có lỗi
                        session_state = self._update_session_state(session_state, {
                            "session_id": str(uuid.uuid4()),
                            "conversation_history": []
                        })
                    
                    # Thêm debug log
                    logger.info("Clear chat triggered - reset completed, all data cleared")
                    
                    # Trả về empty session và UI elements
                    return session_state, "", [], None, "**Chưa có ảnh**", None
                
                # Kết nối nút clear với hàm xử lý
                clear_btn.click(
                    fn=clear_handler,
                    inputs=[],
                    outputs=[session_state, msg, chatbot, image, image_status, image_state]
                )
                
                # CRITICAL FIX: Thêm log để giúp debug conversation_history
                def debug_conversation_history(session_state):
                    """Debug conversation history state"""
                    try:
                        if not session_state:
                            logger.info(f"DEBUG: session_state is empty or None")
                            return "No session state."
                            
                        session_id = session_state.get("session_id", "No session ID")
                        conversation_history = session_state.get("conversation_history", [])
                        
                        logger.info(f"DEBUG: Session {session_id} has conversation_history with {len(conversation_history)} entries")
                        
                        if conversation_history and len(conversation_history) > 0:
                            last_entry = conversation_history[-1]
                            logger.info(f"DEBUG: Last entry query: {last_entry.get('query', 'None')[:30]}...")
                            
                        return f"Session {session_id} has {len(conversation_history)} conversation entries."
                    except Exception as e:
                        logger.error(f"Error in debug_conversation_history: {str(e)}")
                        return "Error debugging conversation history."
                
                # Xử lý sự kiện đồng bộ hóa lịch sử - FIXED
                def sync_history_handler(history, username, session_state):
                    # Lấy session_id từ session_state hoặc từ persistent storage
                    session_id = self._get_session_value(session_state, "session_id")
                    if not session_id:
                        session_id = self._load_persistent_session_id(username)
                        if session_id:
                            session_state = self._update_session_state(session_state, {
                                "session_id": session_id
                            })
                    
                    # Nếu không có session_id, không thể đồng bộ
                    if not session_id:
                        return history, session_state
                    
                    # CRITICAL FIX: Tải conversation_history từ file và đảm bảo nó được lưu vào session_state
                    conversation_history = self._load_conversation_history(session_id)
                    if not conversation_history:
                        logger.warning(f"No conversation history found for session {session_id}")
                        return history, session_state
                    
                    # CRITICAL FIX: Cập nhật session_state với conversation_history mới
                    session_state = self._update_session_state(session_state, {
                        "conversation_history": conversation_history
                    })
                    logger.info(f"FIXED: Loaded and updated conversation_history with {len(conversation_history)} entries")
                    
                    # CRITICAL FIX: Hiển thị thông tin bổ sung để debug
                    if conversation_history and len(conversation_history) > 0:
                        last_entry = conversation_history[-1]
                        logger.info(f"Last conversation entry query: {last_entry.get('query', 'None')[:30]}...")
                        resp = last_entry.get('response', 'None')
                        resp_preview = resp[:30] + "..." if resp and len(resp) > 30 else resp
                        logger.info(f"Last conversation entry response: {resp_preview}")
                    
                    # Đồng bộ hóa history UI với conversation_history
                    synced_history = self._sync_ui_history_with_conversation(history, conversation_history)
                    
                    # Log kết quả
                    logger.info(f"Synced UI history: {len(history)} -> {len(synced_history)} messages")
                    
                    return synced_history, session_state
                
                # Kết nối nút đồng bộ với hàm xử lý
                sync_history_btn.click(
                    fn=sync_history_handler,
                    inputs=[chatbot, username, session_state],
                    outputs=[chatbot, session_state]
                )
                    
                # Footer
                with gr.Row():
                    gr.HTML("""
                    <div style="text-align:center; margin-top:10px; padding: 5px; color: #666">
                        <p>Medical AI Assistant | Created by Medical AI Team | Version 1.0.0</p>
                    </div>
                    """)
                    
            return interface
            
        def _load_persistent_session_id(self, username: str) -> str:
            """Tải session ID đã lưu trữ nếu có."""
            session_file = self._get_persistent_session_path(username)
            try:
                if os.path.exists(session_file):
                    with open(session_file, 'r') as f:
                        saved_session_id = f.read().strip()
                        if saved_session_id:
                            logger.info(f"Loaded persistent session ID for {username}: {saved_session_id}")
                            return saved_session_id
            except Exception as e:
                logger.error(f"Error loading persistent session ID: {str(e)}")
            return None
            
        def _save_persistent_session_id(self, username: str, session_id: str) -> bool:
            """Lưu session ID cho lần sử dụng tiếp theo."""
            session_file = self._get_persistent_session_path(username)
            try:
                with open(session_file, 'w') as f:
                    f.write(session_id)
                logger.info(f"Saved persistent session ID for {username}: {session_id}")
                return True
            except Exception as e:
                logger.error(f"Error saving persistent session ID: {str(e)}")
                return False

        def _initialize_medical_ai(self):
            """Khởi tạo Medical AI system."""
            device = self.app_config.get("medical_ai.device", "cpu")
            logger.info(f"Initializing Medical AI with device: {device}")
            
            # Chỉ truyền tham số device để tương thích với cả 2 phiên bản của MedicalGraphConfig
            return MedicalAISystem(MedicalGraphConfig(device=device))
        
        def _save_image_to_temp(self, image) -> str:
            """Lưu ảnh vào thư mục tạm."""
            import tempfile
            import os
            from PIL import Image
            import io
            
            if not image:
                logger.error("No image provided")
                return None
                
            try:
                # Kiểm tra xem image có phải đã là đường dẫn file không
                if isinstance(image, str) and os.path.isfile(image):
                    return image
                
                # Xử lý cho trường hợp image là numpy array (từ Gradio)
                if hasattr(image, 'shape') and len(getattr(image, 'shape', [])) == 3:
                    # Đây là numpy array
                    img = Image.fromarray(image)
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        img.save(tmp.name, format='JPEG')
                        return tmp.name
                
                # Xử lý cho trường hợp image là bytes
                if isinstance(image, bytes):
                    try:
                        # Thử mở như một ảnh
                        img = Image.open(io.BytesIO(image))
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                            img.save(tmp.name, format='JPEG')
                            return tmp.name
                    except Exception as e:
                        logger.warning(f"Could not process image bytes: {str(e)}")
                
                # Fallback: Lưu trực tiếp
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    tmp.write(image if isinstance(image, bytes) else str(image).encode('utf-8'))
                    return tmp.name
            except Exception as e:
                logger.error(f"Error saving image to temp: {str(e)}")
                return None
        
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
            import os
            
            # 1. CRITICAL FIX: Store original message early 
            original_message = message.strip() if message else ""
            if not original_message:
                return "", history, session_state
            
            logger.info(f"[DEBUG] Processing message: '{original_message[:50]}...'")
            
            # Generate session ID với persistent storage
            # session_id = session_state.get("session_id")  # ❌ Lỗi: 'State' object has no attribute 'get'
            
            # Sửa lại cách truy cập session_state
            if isinstance(session_state, dict):
                session_id = session_state.get("session_id")
            else:
                # Nếu session_state là đối tượng Gradio State, không phải dict
                try:
                    session_id = session_state["session_id"] if "session_id" in session_state else None
                except (TypeError, KeyError):
                    session_id = None
                    
            logger.info(f"[SESSION] Current session state: {type(session_state)}")
            try:
                logger.info(f"[SESSION] Current session state keys: {list(session_state.keys() if hasattr(session_state, 'keys') else [])}")
            except:
                logger.info(f"[SESSION] Session state doesn't support keys()")
            logger.info(f"[SESSION] Current session_id from state: {session_id}")
            
            # Flag để kiểm tra nếu session được phục hồi
            is_restored_session = False
            
            if not session_id:
                # Thử tải session ID từ persistent storage
                session_id = self._load_persistent_session_id(username)
                if session_id:
                    is_restored_session = True
            
            # Nếu vẫn chưa có session ID, tạo mới
            if not session_id:
                session_id = str(uuid.uuid4())
                logger.info(f"Created new session ID: {session_id}")
                
                # Lưu session ID mới vào persistent storage
                self._save_persistent_session_id(username, session_id)
            else:
                logger.info(f"Reusing existing session ID: {session_id}")
            
            # Tạo user ID từ username
            user_id = self.generate_user_id(username)
            
            # Cập nhật session_state - Sửa lỗi với State object
            # session_state["session_id"] = session_id
            # session_state["user_id"] = user_id
            
            # Thay vì gán trực tiếp, sử dụng phương thức cập nhật an toàn
            session_state = self._update_session_state(session_state, {
                "session_id": session_id,
                "user_id": user_id
            })
            
            # FIX: Luôn load lịch sử nếu là session được phục hồi, không chỉ khi history trống
            if is_restored_session:
                logger.info(f"Loading history from restored session: {session_id}")
                old_history = self.load_previous_session(username, session_id)
                
                # Sửa lại cách truy cập session_state để kiểm tra conversation_history
                has_conversation_history = False
                if isinstance(session_state, dict):
                    has_conversation_history = "conversation_history" in session_state and session_state["conversation_history"]
                else:
                    try:
                        has_conversation_history = "conversation_history" in session_state and session_state["conversation_history"]
                    except (TypeError, KeyError):
                        has_conversation_history = False
                
                if not has_conversation_history:
                    conversation_history = self._load_conversation_history(session_id)
                    if conversation_history:
                        logger.info(f"[FIX] Manually loaded {len(conversation_history)} entries to conversation_history")
                        # session_state["conversation_history"] = conversation_history  # Lỗi với State object
                        session_state = self._update_session_state(session_state, {
                            "conversation_history": conversation_history
                        })
                    else:
                        # Khởi tạo mới nếu không tìm thấy
                        logger.info(f"[FIX] Initializing new conversation_history for session {session_id}")
                        # session_state["conversation_history"] = []  # Lỗi với State object
                        session_state = self._update_session_state(session_state, {
                            "conversation_history": []
                        })
                
                # Khôi phục phần xử lý old_history
                if old_history:
                    # FIX: Chỉ thay thế history nếu nó trống hoặc không có tin nhắn nào
                    if not history or len(history) == 0:
                        history = old_history
                    else:
                        # FIX: Nếu đã có history, kiểm tra xem có trùng lặp không
                        # và chỉ thêm các tin nhắn không trùng lặp
                        existing_messages = set()
                        for msg_pair in history:
                            if len(msg_pair) >= 2:
                                existing_messages.add(msg_pair[0])
                        
                        # Thêm các tin nhắn cũ không trùng lặp
                        for old_msg_pair in old_history:
                            if len(old_msg_pair) >= 2 and old_msg_pair[0] not in existing_messages:
                                history.insert(0, old_msg_pair)  # Thêm vào đầu để giữ thứ tự thời gian
                                existing_messages.add(old_msg_pair[0])
                    
                    yield "", history, session_state
                    logger.info(f"Loaded {len(old_history)} messages from old session")
            
            # Start with user message - FIXED: preserve original message
            history.append([original_message, "🤔 Analyzing..."])  # Use original_message
            yield "", history, session_state  # Return empty string to clear input
            
            try:
                # Get contextual information
                context = self.memory.get_contextual_prompt(session_id, user_id)
                
                # IMPORTANT: Save session_id in both global state and temporary result
                # This ensures the session_id persists through the entire process
                session_state["session_id"] = session_id
                
                # FIX: Đảm bảo conversation_history được khởi tạo đúng cách
                has_conversation_history = False
                if isinstance(session_state, dict):
                    has_conversation_history = "conversation_history" in session_state and session_state["conversation_history"]
                else:
                    try:
                        has_conversation_history = "conversation_history" in session_state and session_state["conversation_history"]
                    except (TypeError, KeyError):
                        has_conversation_history = False
                    
                if not has_conversation_history:
                    # Thử load lại từ file nếu chưa có
                    conversation_history = self._load_conversation_history(session_id)
                    if conversation_history:
                        logger.info(f"[FIX] Manually loaded {len(conversation_history)} entries to conversation_history")
                        # session_state["conversation_history"] = conversation_history  # Lỗi với State object
                        session_state = self._update_session_state(session_state, {
                            "conversation_history": conversation_history
                        })
                    else:
                        # Khởi tạo mới nếu không tìm thấy
                        logger.info(f"[FIX] Initializing new conversation_history for session {session_id}")
                        # session_state["conversation_history"] = []  # Lỗi với State object
                        session_state = self._update_session_state(session_state, {
                            "conversation_history": []
                        })
                
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
                        conversation_history=self._get_session_value(session_state, "conversation_history", []),  # CRITICAL: Pass conversation history
                        session_id=session_id  # CRITICAL: Pass session_id explicitly
                    )
                    
                    logger.debug(f"Image analysis result: success={result.get('success', False)}")
                    
                    if result.get("success", False):
                        # Build response (existing code...)
                        response_parts = []
                        
                        # Handle polyp detection if available
                        if "agent_results" in result and "detector" in result.get("agent_results", {}):
                            detector = result["agent_results"]["detector"]
                            
                            if detector.get("success", False) and detector.get("count", 0) > 0:
                                # Display entry 0 response first but REMOVE Medical AI Assessment prefix if exists
                                if "response" in result and isinstance(result["response"], list) and len(result["response"]) > 0:
                                    response_text = result['response'][0]
                                    # Kiểm tra và loại bỏ tiêu đề Medical AI Assessment nếu có
                                    if response_text.startswith("Medical AI Assessment:"):
                                        response_text = response_text.replace("Medical AI Assessment:", "").strip()
                                    response_parts.append(f"💬 **{response_text}**\n")
                                
                                # Lưu thông tin visualization (nhưng chưa hiển thị)
                                if detector.get("visualization_base64"):
                                    viz_base64 = detector["visualization_base64"]
                                    viz_filename = f"polyp_viz_{session_state.get('session_id', 'unknown')}_{int(time.time())}.png"
                                    viz_path = self._save_visualization(viz_base64, viz_filename)
                                    
                                    # FIXED: Lưu cả path và base64 data
                                    # session_state["last_visualization"] = viz_path
                                    # session_state["last_visualization_base64"] = viz_base64
                                    # session_state["has_visualization"] = True
                                    # session_state["pending_viz"] = True  # Flag để hiển thị sau final_answer
                                    session_state = self._update_session_state(session_state, {
                                        "last_visualization": viz_path,
                                        "last_visualization_base64": viz_base64,
                                        "has_visualization": True,
                                        "pending_viz": True
                                    })
                            else:
                                # No polyps detected case
                                # Display entry 0 response first for no polyp case but REMOVE Medical AI Assessment prefix if exists
                                if "response" in result and isinstance(result["response"], list) and len(result["response"]) > 0:
                                    response_text = result['response'][0]
                                    # Kiểm tra và loại bỏ tiêu đề Medical AI Assessment nếu có
                                    if response_text.startswith("Medical AI Assessment:"):
                                        response_text = response_text.replace("Medical AI Assessment:", "").strip()
                                    response_parts.append(f"💬 **{response_text}**\n")
                                response_parts.append("🔍 **No polyps detected in this image.**")
                        # Giữ lại điều kiện cũ nhưng ghi log để hiểu sự khác biệt
                        elif "polyps" in result:
                            polyp_count = len(result.get("polyps", []))
                        
                        # FIXED: Check if Medical AI Assessment is already added
                        has_assessment_header = any("Medical AI Assessment" in part for part in response_parts)
                        
                        # Add final answer if available
                        if "final_answer" in result:
                            # REMOVED: Không hiển thị tiêu đề Medical AI Assessment nữa
                            # if not has_assessment_header:
                            #     response_parts.append("\n💬 **Medical AI Assessment:**")
                            response_parts.append(result["final_answer"])
                            
                            # Hiển thị visualization sau final_answer nếu có
                            if session_state.get("pending_viz") and session_state.get("last_visualization_base64"):
                                viz_base64 = session_state["last_visualization_base64"]
                                img_data_url = f"data:image/png;base64,{viz_base64}"
                                response_parts.append("\n\n📊 **Detector Results:**")
                                
                                # Add synthesized analysis before the image
                                if "final_answer" in result and not result["final_answer"] in response_parts:
                                    # Get a short version of the synthesized result if it's long
                                    synth_result = result["final_answer"]
                                    if len(synth_result) > 150:
                                        sentences = synth_result.split('.')[:2]
                                        short_result = '.'.join(sentences) + '.'
                                        response_parts.append(f"{short_result}\n")
                                
                                # Add the visualization image
                                response_parts.append(f'<img src="{img_data_url}" alt="Polyp Detection Results" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;">')
                                session_state = self._update_session_state(session_state, {
                                    "pending_viz": False
                                })
                        
                        # CRITICAL FIX: Update conversation history properly
                        if "conversation_history" in result:
                            updated_history = result["conversation_history"]
                            
                            # Ensure the last entry has the correct query
                            if updated_history and len(updated_history) > 0:
                                last_entry = updated_history[-1]
                                
                                # Fix query in last entry if needed
                                if not last_entry.get("query") or last_entry.get("query") != original_message:
                                    logger.info(f"[FIX] Correcting query in history: '{last_entry.get('query', 'EMPTY')}' -> '{original_message}'")
                                    last_entry["query"] = original_message
                            
                            # session_state["conversation_history"] = updated_history  # Lỗi với State object
                            session_state = self._update_session_state(session_state, {
                                "conversation_history": updated_history
                            })
                            logger.info(f"[FIXED] Updated session with conversation_history: {len(updated_history)} entries")
                            
                            # FIX: Lưu conversation_history vào file để phục hồi sau này
                            self._save_conversation_history(session_id, updated_history)
                        
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
                            # REMOVED: Không hiển thị tiêu đề Medical AI Assessment nữa
                            # Kiểm tra xem Medical AI Assessment đã được thêm chưa
                            # medical_ai_header_exists = "Medical AI Assessment" in streaming_text
                            
                            # Hiển thị tiêu đề trước nếu chưa tồn tại
                            # if not medical_ai_header_exists:
                            #     streaming_text += "\n💬 **Medical AI Assessment:**\n"
                            #     history[-1][1] = streaming_text
                            #     yield "", history, session_state
                            #     time.sleep(0.2)
                            
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
                                logger.info(f"dont use chunk: {final_answer_part}")
                                final_content = final_answer_part.replace("\n💬 **Medical AI Assessment:**\n", "").replace("\n💬 **Medical AI Assessment:**", "")
                                
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
                        conversation_history=self._get_session_value(session_state, "conversation_history", []),  # CRITICAL: Pass conversation history
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
                                # Loại bỏ dòng "Based on previous information:"
                                # streaming_text += "💭 **Based on previous information:**\n"
                                # streaming_text += (context[:200] + "..." if len(context) > 200 else context) + "\n\n"
                                pass  # Bỏ qua phần hiển thị thông tin ngữ cảnh trước đó
                            
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
                            
                            # Loại bỏ footer "Processed by: LLaVA-Med (Medical LLM)"
                            # current_text += "\n\n🔬 **Processed by:** LLaVA-Med (Medical LLM)"
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
                                
                                # session_state["conversation_history"] = updated_history  # Lỗi với State object
                                session_state = self._update_session_state(session_state, {
                                    "conversation_history": updated_history
                                })
                                logger.info(f"[FIXED] Updated session with text conversation_history: {len(updated_history)} entries")
                                
                                # FIX: Lưu conversation_history vào file cho cả text-only workflow
                                self._save_conversation_history(session_id, updated_history)
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
                if not self._get_session_value(session_state, "conversation_history"):
                    logger.warning(f"conversation_history missing in session_state, initializing new one")
                    # session_state["conversation_history"] = []  # Lỗi với State object
                    session_state = self._update_session_state(session_state, {
                        "conversation_history": []
                    })
                
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
                        # logger.info(f"Manually adding current interaction to conversation_history: '{original_message[:30]}...'")
                        history_entry = {
                            "query": original_message,
                            "response": final_response,
                            "timestamp": time.time(),
                            "has_image": has_image,
                            "session_id": session_id
                        }
                        # Add entry to conversation history
                        curr_history.append(history_entry)
                        # session_state["conversation_history"] = curr_history  # Lỗi với State object
                        session_state = self._update_session_state(session_state, {
                            "conversation_history": curr_history
                        })
                        # logger.info(f"Conversation history updated manually, now has {len(curr_history)} entries")
                
                # Debug - log final state of conversation history
                conversation_history = session_state.get("conversation_history", [])
                # logger.info(f"FINAL conversation_history has {len(conversation_history)} entries")
                # if conversation_history:
                #     for i, entry in enumerate(conversation_history[:]):  # Show last 2 entries
                #         logger.info(f"FINAL HISTORY ENTRY {i}:")
                #         logger.info(f"  - QUERY: {entry.get('query', 'None')[:30]}...")
                #         resp = entry.get('response', 'None')
                #         resp_preview = resp[:30] + "..." if resp and len(resp) > 30 else resp
                #         logger.info(f"  - RESPONSE: {resp_preview}")
                
                # Save important interactions to long term
                if has_image or "polyp" in original_message.lower() or "medical" in original_message.lower():
                    logger.info(f"Saving important interaction to long-term memory for user {user_id}")
                    self.memory.save_to_long_term(user_id, session_id, interaction)
                
                # Save session history sau khi có response
                if username and session_id and history:
                    self.save_session_history(username, session_id, history)
                
                # Make sure result is in markdown 
                return "", history, session_state
                
            except Exception as e:
                import traceback
                logger.error(f"Error processing message: {str(e)}")
                logger.error(traceback.format_exc())
                error_response = self._create_system_error_response(original_message)  # Use original_message
                history[-1][1] = error_response
                yield "", history, session_state
        
        def collect_response_parts(self, parts):
            """Ghép các phần của response thành một chuỗi đầy đủ."""
            if not parts:
                return ""
                
            return "\n".join(parts)
        
        def _create_system_error_response(self, query):
            """Create an error response for system failures."""
            error_message = [
                "❌ **System Error**",
                "",
                "I apologize, but I encountered a technical issue while processing your request.",
                "",
                "Please try again with:",
                "- A clearer description",
                "- A different image (if you uploaded one)",
                "- Break complex questions into simpler ones",
                "",
                "If the problem persists, please contact technical support."
            ]
            return "\n".join(error_message)
            
        def save_session_history(self, username, session_id, history):
            """Save chat history to disk"""
            if not username or not session_id or not history:
                logger.warning(f"Missing data for save_session_history: username={bool(username)}, session_id={bool(session_id)}, history={len(history) if history else 0}")
                return False
                
            # Ensure history directory exists
            history_dir = os.path.join("sessions", "history")
            os.makedirs(history_dir, exist_ok=True)
            
            history_file = os.path.join(history_dir, f"{session_id}.json")
            
            # Log the history we're about to save
            logger.info(f"[SAVE] Saving {len(history)} messages to {history_file}")
            
            try:
                with open(history_file, "w") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
                    
                logger.info(f"[SAVE] Successfully saved {len(history)} messages for session {session_id}")
                return True
            except Exception as e:
                logger.error(f"Error saving session history: {str(e)}")
                return False
                
        def load_previous_session(self, username, session_id):
            """Load chat history from a previous session"""
            if not username or not session_id:
                logger.warning(f"Missing username or session_id in load_previous_session")
                return []
                
            # Check for session messages
            history_dir = os.path.join("sessions", "history")
            os.makedirs(history_dir, exist_ok=True)
            
            history_file = os.path.join(history_dir, f"{session_id}.json")
            logger.info(f"[LOAD] Checking for history file: {history_file}")
            
            if not os.path.exists(history_file):
                logger.info(f"No history file found for session {session_id}")
                # Fallback to memory system if available
                try:
                    user_id = self.generate_user_id(username)
                    if hasattr(self, "memory") and hasattr(self.memory, "load_previous_session"):
                        memory_history = self.memory.load_previous_session(user_id, session_id)
                        if memory_history:
                            logger.info(f"Found {len(memory_history)} messages in memory system")
                            return memory_history
                except Exception as e:
                    logger.error(f"Error checking memory system: {e}")
                return []
                
            try:
                with open(history_file, "r") as f:
                    logger.info(f"[LOAD] Reading history file for session {session_id}")
                    history = json.load(f)
                    # FIX: Đảm bảo rằng history đọc được là hợp lệ
                    if not isinstance(history, list):
                        logger.error(f"Invalid history format in {history_file}, expected list but got {type(history)}")
                        return []
                    
                    # FIX: Lọc các tin nhắn không hợp lệ
                    valid_history = []
                    for entry in history:
                        if isinstance(entry, list) and len(entry) >= 2:
                            valid_history.append(entry)
                        else:
                            logger.warning(f"Skipping invalid history entry: {entry}")
                    
                    logger.info(f"[LOAD] Loaded {len(valid_history)} valid messages from session {session_id}")
                    return valid_history
            except Exception as e:
                logger.error(f"Error loading session history: {str(e)}")
                return []
        
        def get_user_sessions(self, username: str) -> List[Dict[str, Any]]:
            """Lấy danh sách các phiên của người dùng."""
            if not username:
                return []
                
            # Lấy user ID từ username
            user_id = self.generate_user_id(username)
            
            # Danh sách kết quả session
            sessions = {}
            
            # Thử lấy session từ bộ nhớ dài hạn (nếu có)
            try:
                if hasattr(self, "memory") and hasattr(self.memory, "get_user_sessions"):
                    memory_sessions = self.memory.get_user_sessions(user_id)
                    # Thêm các session từ bộ nhớ dài hạn
                    for session in memory_sessions:
                        session_id = session.get("session_id")
                        if session_id:
                            sessions[session_id] = session
            except Exception as e:
                logger.error(f"Error getting sessions from memory: {e}")
            
            # Thử lấy persistent session ID (nếu có)
            persistent_session = self._load_persistent_session_id(username)
            
            # Kiểm tra session history cho persistent session
            history_dir = os.path.join("sessions", "history")
            os.makedirs(history_dir, exist_ok=True)
            
            # Thêm các session từ thư mục history
            try:
                for file_name in os.listdir(history_dir):
                    if file_name.endswith(".json"):
                        session_id = file_name.replace(".json", "")
                        if session_id not in sessions:
                            history_file = os.path.join(history_dir, file_name)
                            try:
                                with open(history_file, "r") as f:
                                    history = json.load(f)
                                    msg_count = len(history)
                                    
                                session = {
                                    "session_id": session_id,
                                    "user_id": user_id, 
                                    "timestamp": os.path.getmtime(history_file),
                                    "messages": msg_count,
                                    "display_name": f"Session {session_id[:6]}... ({msg_count} messages)"
                                }
                                sessions[session_id] = session
                            except Exception as e:
                                logger.error(f"Error reading history file {file_name}: {e}")
            except Exception as e:
                logger.error(f"Error listing history directory: {e}")
            
            # Thêm persistent session nếu có và chưa được thêm
            if persistent_session and persistent_session not in sessions:
                # Tìm session history file
                history_file = os.path.join(history_dir, f"{persistent_session}.json")
                
                if os.path.exists(history_file):
                    # Tạo session metadata từ file
                    try:
                        with open(history_file, "r") as f:
                            history = json.load(f)
                            msg_count = len(history)
                            
                        session = {
                            "session_id": persistent_session,
                            "user_id": user_id,
                            "timestamp": time.time(),
                            "messages": msg_count,
                            "display_name": f"Session {persistent_session[:6]}... ({msg_count} messages)",
                            "is_current": True
                        }
                        sessions[persistent_session] = session
                    except Exception as e:
                        logger.error(f"Error loading persistent session history: {e}")
            
            # Chuyển đổi dict thành list và sắp xếp theo thời gian
            result = list(sessions.values())
            result.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            return result
            
        def _load_persistent_session_id(self, username):
            """Load persistent session ID from disk"""
            if not username:
                return None
                
            session_file = os.path.join("sessions", f"{username}.session")
            
            if not os.path.exists(session_file):
                return None
                
            try:
                with open(session_file, "r") as f:
                    data = json.load(f)
                    logger.info(f"Loaded persistent session ID: {data.get('session_id')}")
                    return data.get("session_id")
            except Exception as e:
                logger.error(f"Error loading session ID: {e}")
                return None
                
        def _save_persistent_session_id(self, username, session_id):
            """Save persistent session ID to disk"""
            if not username or not session_id:
                return False
                
            # Ensure sessions directory exists
            os.makedirs("sessions", exist_ok=True)
            
            session_file = os.path.join("sessions", f"{username}.session")
            
            try:
                # Save both session ID and creation timestamp
                data = {
                    "session_id": session_id,
                    "created_at": datetime.now().isoformat(),
                    "username": username
                }
                
                with open(session_file, "w") as f:
                    json.dump(data, f)
                    
                logger.info(f"Saved persistent session ID: {session_id}")
                return True
            except Exception as e:
                logger.error(f"Error saving session ID: {e}")
                return False

        def _load_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
            """Load conversation history từ file lưu trữ."""
            if not session_id:
                return []
                
            # Đường dẫn file lưu trữ conversation history
            history_dir = os.path.join("sessions", "conversation_history")
            os.makedirs(history_dir, exist_ok=True)
            
            history_file = os.path.join(history_dir, f"{session_id}.json")
            # logger.info(f"[LOAD] Checking for conversation history file: {history_file}")
            
            if not os.path.exists(history_file):
                # logger.info(f"No conversation history file found for session {session_id}")
                return []
                
            try:
                with open(history_file, "r") as f:
                    # logger.info(f"[LOAD] Reading conversation history file for session {session_id}")
                    conversation_history = json.load(f)
                    
                    # Validate format
                    if not isinstance(conversation_history, list):
                        logger.error(f"Invalid conversation history format, expected list but got {type(conversation_history)}")
                        return []
                        
                    # logger.info(f"[LOAD] Loaded {len(conversation_history)} entries from conversation history")
                    return conversation_history
            except Exception as e:
                logger.error(f"Error loading conversation history: {str(e)}")
                return []

        def _save_conversation_history(self, session_id: str, conversation_history: List[Dict[str, Any]]) -> bool:
            """Save conversation history to file."""
            if not session_id or not conversation_history:
                return False
                
            # Ensure directory exists
            history_dir = os.path.join("sessions", "conversation_history")
            os.makedirs(history_dir, exist_ok=True)
            
            history_file = os.path.join(history_dir, f"{session_id}.json")
            
            try:
                with open(history_file, "w") as f:
                    json.dump(conversation_history, f, ensure_ascii=False, indent=2)
                    
                # logger.info(f"[SAVE] Saved {len(conversation_history)} entries to conversation history")
                return True
            except Exception as e:
                logger.error(f"Error saving conversation history: {str(e)}")
                return False

        # Thêm helper method để truy cập an toàn vào session_state
        def _get_session_value(self, session_state, key, default=None):
            """Truy cập an toàn giá trị từ session_state."""
            if isinstance(session_state, dict):
                return session_state.get(key, default)
            else:
                try:
                    return session_state[key] if key in session_state else default
                except (TypeError, KeyError, AttributeError):
                    return default

        # Thêm helper method để cập nhật an toàn session_state
        def _update_session_state(self, session_state, updates):
            """Cập nhật an toàn session_state."""
            if isinstance(session_state, dict):
                # Nếu là dict bình thường, cập nhật trực tiếp
                session_state.update(updates)
                return session_state
            else:
                # Nếu là đối tượng State của Gradio hoặc loại khác
                try:
                    # Tạo một dict mới từ đối tượng state hiện tại
                    new_state = {}
                    try:
                        # Cố gắng lưu các giá trị hiện có vào dict mới
                        for key in session_state:
                            new_state[key] = session_state[key]
                    except:
                        logger.warning("Unable to iterate through session_state, creating new state")
                    
                    # Cập nhật từ dict mới
                    new_state.update(updates)
                    return new_state
                except Exception as e:
                    # Fallback nếu có lỗi
                    logger.error(f"Error updating session state: {e}")
                    return updates  # Trả về updates như một state mới

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
    parser.add_argument("--install-faiss-avx2", action="store_true", help="Install FAISS with AVX2 support and exit")
    
    args = parser.parse_args()
    
    # Install FAISS with AVX2 support if requested
    if args.install_faiss_avx2:
        print("🔄 Installing FAISS with AVX2 support...")
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "faiss-gpu" if args.device=="cuda" else "faiss-cpu"], check=True)
            print("✅ FAISS installation completed. Please restart the application.")
        except Exception as e:
            print(f"❌ FAISS installation failed: {e}")
        return
    
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