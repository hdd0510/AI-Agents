#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Chatbot Launcher
===========================
Script khởi động chatbot với nhiều tùy chọn cấu hình.
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
from pathlib import Path
os.environ['GRADIO_TEMP_DIR'] = '/tmp'

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
            "port": 7860,
            "share": True,
            "debug": False
        },
        "medical_ai": {
            "device": "cuda",
            "use_reflection": True,
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
        """Chatbot với các tính năng nâng cao."""
        
        def __init__(self, config: MedicalAIConfig):
            self.app_config = config
            super().__init__()
        
        def _initialize_medical_ai(self):
            """Khởi tạo Medical AI với config tùy chỉnh."""
            medical_config = MedicalGraphConfig(
                device=self.app_config.get("medical_ai.device", "cuda"),
                use_reflection=self.app_config.get("medical_ai.use_reflection", True),
                detector_model_path=self.app_config.get("medical_ai.detector_model_path"),
                vqa_model_path=self.app_config.get("medical_ai.vqa_model_path"),
                modality_classifier_path=self.app_config.get("medical_ai.modality_classifier_path"),
                region_classifier_path=self.app_config.get("medical_ai.region_classifier_path")
            )
            
            from medical_ai_agents import MedicalAISystem
            return MedicalAISystem(medical_config)
        
        def create_enhanced_interface(self):
            """Tạo giao diện với nhiều tính năng hơn."""
            
            # Custom CSS
            custom_css = """
            .main-container { 
                max-width: 1400px; 
                margin: 0 auto;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .header-section {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                text-align: center;
            }
            .chat-container { 
                height: 600px; 
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .upload-container { 
                border: 2px dashed #4CAF50; 
                padding: 20px; 
                text-align: center;
                border-radius: 10px;
                background: #f8f9fa;
            }
            .stats-panel {
                background: #e3f2fd;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
            .quick-actions {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }
            .memory-info {
                background: #fff3e0;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #ff9800;
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
                    ### {self.app_config.get("app.description", "Hệ thống AI hỗ trợ phân tích hình ảnh nội soi")}
                    
                    **🎯 Tính năng nổi bật:**
                    - 🧠 **Trí nhớ thông minh**: Ghi nhớ lịch sử và cá nhân hóa trải nghiệm
                    - 🔍 **Phân tích chính xác**: Sử dụng AI đa agent với độ tin cậy cao
                    - 💬 **Tương tác tự nhiên**: Chat thông minh với khả năng hiểu ngữ cảnh
                    - 📊 **Thống kê chi tiết**: Theo dõi tiến trình sức khỏe qua thời gian
                    """)
                
                with gr.Row():
                    with gr.Column(scale=3):
                        # Main chat interface
                        chatbot = gr.Chatbot(
                            label="💬 Cuộc trò chuyện với AI",
                            height=self.app_config.get("ui.chat_height", 500),
                            show_copy_button=True,
                            elem_classes=["chat-container"],
                        )
                        
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="💭 Hãy mô tả triệu chứng hoặc đặt câu hỏi về hình ảnh...",
                                label="Tin nhắn của bạn",
                                scale=5,
                                lines=2
                            )
                            send_btn = gr.Button("📤 Gửi", variant="primary", scale=1)
                        
                        # Advanced image upload
                        with gr.Accordion("🖼️ Tải lên hình ảnh y tế", open=True):
                            image_input = gr.Image(
                                label="Chọn hình ảnh nội soi, X-quang hoặc hình ảnh y tế khác",
                                type="filepath",
                                elem_classes=["upload-container"]
                            )
                            
                            gr.Markdown("""
                            **Lưu ý:** 
                            - Hỗ trợ định dạng: JPG, PNG, DICOM
                            - Kích thước tối đa: 10MB
                            - Đảm bảo hình ảnh rõ nét cho kết quả tốt nhất
                            """)
                    
                    with gr.Column(scale=1):
                        # User panel
                        with gr.Accordion("👤 Thông tin cá nhân", open=True):
                            username_input = gr.Textbox(
                                label="Tên của bạn",
                                placeholder="Nhập tên để cá nhân hóa...",
                                value="Bệnh nhân"
                            )
                            
                            user_info = gr.Textbox(
                                label="Thông tin bổ sung",
                                placeholder="Tuổi, tiền sử bệnh, thuốc đang dùng...",
                                lines=3
                            )
                        
                        # Stats panel
                        with gr.Accordion("📊 Thống kê của bạn", open=False):
                            stats_display = gr.Markdown(
                                "📈 Chưa có dữ liệu thống kê",
                                elem_classes=["stats-panel"]
                            )
                            stats_btn = gr.Button("🔄 Cập nhật thống kê", variant="secondary")
                        
                        # Quick actions
                        with gr.Accordion("⚡ Thao tác nhanh", open=True):
                            with gr.Column(elem_classes=["quick-actions"]):
                                quick_polyp_btn = gr.Button("🔍 Kiểm tra polyp", size="sm")
                                quick_general_btn = gr.Button("🩺 Tư vấn tổng quát", size="sm")
                                history_btn = gr.Button("📜 Xem lịch sử", size="sm")
                                export_btn = gr.Button("💾 Xuất báo cáo", size="sm")
                                clear_btn = gr.Button("🗑️ Xóa cuộc trò chuyện", size="sm", variant="stop")
                        
                        # Memory and AI info
                        with gr.Accordion("🧠 Trí nhớ AI", open=False):
                            gr.Markdown("""
                            <div class="memory-info">
                            <h4>🤖 Khả năng AI:</h4>
                            <ul>
                                <li><strong>Ngắn hạn:</strong> Nhớ 10 tương tác gần nhất</li>
                                <li><strong>Dài hạn:</strong> Lưu kết quả quan trọng</li>
                                <li><strong>Học tập:</strong> Cải thiện theo thời gian</li>
                                <li><strong>Bảo mật:</strong> Dữ liệu được mã hóa</li>
                            </ul>
                            </div>
                            """)
                            
                            memory_status = gr.Textbox(
                                label="Trạng thái bộ nhớ",
                                value="✅ Đang hoạt động bình thường",
                                interactive=False
                            )
                
                # Event handlers với error handling
                def safe_process_message(message, image, history, username, user_info, state):
                    try:
                        # Add user info to medical context
                        if user_info.strip():
                            if "medical_context" not in state:
                                state["medical_context"] = {}
                            state["medical_context"]["user_info"] = user_info
                        
                        return self.process_message(message, image, history, username, state)
                    except Exception as e:
                        error_msg = f"❌ Lỗi xử lý: {str(e)}"
                        history.append([message, error_msg])
                        return "", history, state
                
                def safe_update_stats(username):
                    try:
                        return self.get_user_stats(username)
                    except Exception as e:
                        return f"❌ Lỗi lấy thống kê: {str(e)}"
                
                def export_chat_history(chat_history, username):
                    """Xuất lịch sử chat ra file."""
                    if not chat_history:
                        return "❌ Không có dữ liệu để xuất"
                    
                    try:
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"medical_chat_{username}_{timestamp}.txt"
                        
                        with open(filename, 'w', encoding='utf-8') as f:
                            f.write(f"=== Medical AI Chat History ===\n")
                            f.write(f"User: {username}\n")
                            f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                            f.write("="*50 + "\n\n")
                            
                            for i, (user_msg, ai_msg) in enumerate(chat_history, 1):
                                f.write(f"[{i}] User: {user_msg}\n")
                                f.write(f"[{i}] AI: {ai_msg}\n")
                                f.write("-"*30 + "\n")
                        
                        return f"✅ Đã xuất lịch sử ra file: {filename}"
                    except Exception as e:
                        return f"❌ Lỗi xuất file: {str(e)}"
                
                # Connect all events
                send_btn.click(
                    safe_process_message,
                    inputs=[msg_input, image_input, chatbot, username_input, user_info, session_state],
                    outputs=[msg_input, chatbot, session_state]
                )
                
                msg_input.submit(
                    safe_process_message,
                    inputs=[msg_input, image_input, chatbot, username_input, user_info, session_state],
                    outputs=[msg_input, chatbot, session_state]
                )
                
                stats_btn.click(
                    safe_update_stats,
                    inputs=[username_input],
                    outputs=[stats_display]
                )
                
                clear_btn.click(lambda: [], outputs=[chatbot])
                
                quick_polyp_btn.click(
                    lambda: "Hãy phân tích hình ảnh này và cho tôi biết có phát hiện polyp không? Nếu có, xin mô tả chi tiết.",
                    outputs=[msg_input]
                )
                
                quick_general_btn.click(
                    lambda: "Tôi cần tư vấn y tế tổng quát dựa trên hình ảnh. Xin đánh giá tình trạng sức khỏe.",
                    outputs=[msg_input]
                )
                
                export_btn.click(
                    export_chat_history,
                    inputs=[chatbot, username_input],
                    outputs=[gr.Textbox(visible=False)]  # Show result in console
                )
            
            return interface
    
    # Create and return enhanced chatbot
    return EnhancedMedicalAIChatbot(config)

def main():
    """Hàm main với argument parsing."""
    parser = argparse.ArgumentParser(description="Medical AI Chatbot Launcher")
    parser.add_argument("--config", default="config.json", help="Path to config file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
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