#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Chatbot Launcher - FIXED với Image Display + Streaming
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
import time
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
            "port": 8000,
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
        """Chatbot với các tính năng nâng cao + streaming."""
        
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
        
        def process_message_streaming(self, message, image, history, username, session_state):
            """STREAMING version của process_message."""
            import uuid
            
            # Generate session and user IDs
            session_id = session_state.get("session_id", str(uuid.uuid4()))
            user_id = self.generate_user_id(username)
            
            session_state["session_id"] = session_id
            session_state["user_id"] = user_id
            
            # Start with user message
            history.append([message, "🤔 Đang phân tích..."])
            yield "", history, session_state
            
            try:
                response_parts = []
                analysis_result = None
                
                # Get contextual information
                context = self.memory.get_contextual_prompt(session_id, user_id)
                
                # Process image if provided
                if image is not None:
                    # Streaming progress
                    history[-1][1] = "🔍 Đang phân tích hình ảnh..."
                    yield "", history, session_state
                    time.sleep(0.5)
                    
                    # Analyze image with Medical AI
                    history[-1][1] = "⚙️ Chạy AI detection..."
                    yield "", history, session_state
                    
                    result = self.medical_ai.analyze(
                        image_path=image,
                        query=message,
                        medical_context={"user_context": context} if context else None
                    )
                    
                    analysis_result = result
                    
                    if result.get("success", False):
                        # Stream response parts
                        history[-1][1] = "📝 Đang tạo báo cáo..."
                        yield "", history, session_state
                        time.sleep(0.3)
                        
                        # Create comprehensive response
                        if "final_answer" in result:
                            response_parts.append("🔍 **Kết quả phân tích hình ảnh:**")
                            
                            # Stream the final answer word by word (tùy chọn)
                            final_answer = result["final_answer"]
                            streaming_text = "🔍 **Kết quả phân tích hình ảnh:**\n\n"
                            
                            words = final_answer.split()
                            for i, word in enumerate(words):
                                streaming_text += word + " "
                                if i % 5 == 0:  # Update every 5 words
                                    history[-1][1] = streaming_text + "..."
                                    yield "", history, session_state
                                    time.sleep(0.1)
                            
                            response_parts = [streaming_text.rstrip()]
                        
                        # Add detection details
                        if "agent_results" in result and "detector_result" in result["agent_results"]:
                            detector = result["agent_results"]["detector_result"]
                            if detector.get("success") and detector.get("count", 0) > 0:
                                detection_info = f"\n\n📊 **Chi tiết phát hiện:**\n"
                                detection_info += f"- Số lượng polyp: {detector['count']}\n"
                                detection_info += f"- Độ tin cậy: {detector['objects'][0]['confidence']:.2%}\n"
                                
                                response_parts.append(detection_info)
                                
                                # Stream detection info
                                current_response = "\n".join(response_parts)
                                history[-1][1] = current_response
                                yield "", history, session_state
                                time.sleep(0.5)
                                
                                # FIXED: Hiển thị ảnh visualization trong chat
                                if detector.get("visualization_base64") and detector.get("visualization_available"):
                                    # Lưu base64 vào session_state để sử dụng sau
                                    session_state["last_visualization"] = detector.get("visualization_base64")
                                    
                                    # Tạo data URL từ base64
                                    img_data_url = f"data:image/png;base64,{detector.get('visualization_base64')}"
                                    
                                    # FIXED: Sử dụng HTML img tag thay vì markdown
                                    viz_html = f'\n\n📊 **Kết quả phát hiện polyp:**\n<img src="{img_data_url}" alt="Kết quả phát hiện polyp" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'
                                    
                                    response_parts.append(viz_html)
                                    
                                    # Stream với ảnh
                                    current_response = "\n".join(response_parts)
                                    history[-1][1] = current_response
                                    yield "", history, session_state
                                    time.sleep(0.5)
                                    
                                    # Lưu thông tin ảnh vào session_state
                                    session_state["has_image_result"] = True
                                    session_state["last_result_image_data"] = img_data_url
                        
                        # Add medical recommendations
                        recommendations = "\n\n💡 **Khuyến nghị:**\n"
                        if result.get("polyp_count", 0) > 0:
                            recommendations += "- Nên tham khảo ý kiến bác sĩ chuyên khoa\n"
                            recommendations += "- Theo dõi định kỳ theo lịch hẹn"
                        else:
                            recommendations += "- Duy trì lối sống lành mạnh\n"
                            recommendations += "- Kiểm tra định kỳ theo khuyến nghị"
                        
                        response_parts.append(recommendations)
                        
                        # Final response
                        final_response = "\n".join(response_parts)
                        history[-1][1] = final_response
                        yield "", history, session_state
                        
                    else:
                        error_response = "❌ Có lỗi trong quá trình phân tích hình ảnh.\n"
                        error_response += f"Chi tiết lỗi: {result.get('error', 'Unknown error')}"
                        history[-1][1] = error_response
                        yield "", history, session_state
                
                else:
                    # Text-only conversation - stream response
                    text_response = "💬 **Trả lời:**\n\n"
                    
                    if context:
                        text_response += "💭 **Dựa trên thông tin trước đó:**\n"
                        text_response += (context[:200] + "..." if len(context) > 200 else context) + "\n\n"
                    
                    text_response += "Tôi có thể giúp bạn phân tích hình ảnh nội soi và trả lời các câu hỏi y tế. "
                    text_response += "Vui lòng tải lên hình ảnh để tôi có thể hỗ trợ tốt hơn."
                    
                    # Stream text response
                    words = text_response.split()
                    streaming_text = ""
                    for i, word in enumerate(words):
                        streaming_text += word + " "
                        if i % 3 == 0:  # Update every 3 words
                            history[-1][1] = streaming_text
                            yield "", history, session_state
                            time.sleep(0.05)
                    
                    history[-1][1] = streaming_text.strip()
                    yield "", history, session_state
                
                # Save to memory
                final_response = history[-1][1]
                interaction = {
                    "query": message,
                    "response": final_response,
                    "has_image": image is not None,
                    "analysis": analysis_result,
                    "polyp_count": analysis_result.get("polyp_count", 0) if analysis_result else 0
                }
                
                self.memory.add_to_short_term(session_id, interaction)
                
                # Save important interactions to long term
                if image is not None or "polyp" in message.lower():
                    self.memory.save_to_long_term(user_id, session_id, interaction)
                
            except Exception as e:
                error_response = f"❌ Xin lỗi, có lỗi xảy ra: {str(e)}"
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
                    ### {self.app_config.get("app.description", "Hệ thống AI hỗ trợ phân tích hình ảnh nội soi")}
                    
                    **🎯 Tính năng nổi bật:**
                    - 🧠 **Trí nhớ thông minh**: Ghi nhớ lịch sử và cá nhân hóa trải nghiệm
                    - 🔍 **Phân tích chính xác**: Sử dụng AI đa agent với độ tin cậy cao
                    - 💬 **Tương tác tự nhiên**: Chat thông minh với khả năng hiểu ngữ cảnh
                    - 📊 **Streaming response**: Phản hồi real-time thay vì chờ đợi
                    """)
                
                with gr.Row():
                    # Main chat interface in a centered column
                    with gr.Column(scale=1, min_width=800):
                        # Chat container - FIXED: Enable HTML rendering for images
                        chatbot = gr.Chatbot(
                            label="💬 Cuộc trò chuyện với AI",
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
                                placeholder="💭 Hãy mô tả triệu chứng hoặc đặt câu hỏi về hình ảnh...",
                                label="Tin nhắc của bạn",
                                scale=5,
                                lines=2
                            )
                            with gr.Column(scale=1, elem_classes=["button-row"]):
                                send_btn = gr.Button("📤 Gửi", variant="primary", size="lg")
                                clear_btn = gr.Button("🗑️ Xóa", variant="stop", size="lg")
                        
                        # Tabs for image upload and results
                        tabs = gr.Tabs(elem_classes=["tab-container"])
                        with tabs:
                            with gr.TabItem("🖼️ Tải ảnh"):
                                # Advanced image upload
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
                                - **Ảnh kết quả sẽ hiển thị trực tiếp trong chat**
                                """)
                            
                            with gr.TabItem("📊 Kết quả phát hiện"):
                                result_image = gr.Image(
                                    label="Kết quả phát hiện polyp",
                                    type="filepath",
                                    interactive=False
                                )
                                
                                show_latest_result_btn = gr.Button("🔄 Hiển thị kết quả mới nhất", variant="secondary")
                
                # Hidden state for username (required for functions)
                username_input = gr.Textbox(value="Bệnh nhân", visible=False)
                user_info = gr.Textbox(value="", visible=False)
                
                # FIXED: Event handlers với streaming support
                def safe_process_message_streaming(message, image, history, username, user_info, state):
                    """Wrapper cho streaming function."""
                    try:
                        # Add user info to medical context
                        if user_info.strip():
                            if "medical_context" not in state:
                                state["medical_context"] = {}
                            state["medical_context"]["user_info"] = user_info
                        
                        # Use streaming version
                        for msg, hist, updated_state in self.process_message_streaming(message, image, history, username, state):
                            yield msg, hist, updated_state, None
                        
                        # Check for visualization result
                        if "last_result_image_data" in updated_state:
                            # Convert data URL back to filepath if needed
                            # For now, just yield the state
                            yield msg, hist, updated_state, None
                            
                    except Exception as e:
                        error_msg = f"❌ Lỗi xử lý: {str(e)}"
                        if history:
                            history[-1][1] = error_msg
                        else:
                            history.append([message, error_msg])
                        yield "", history, state, None

                def show_latest_visualization(state):
                    """Hiển thị kết quả phát hiện mới nhất từ session state"""
                    try:
                        if "viz_image_path" in state and state["viz_image_path"]:
                            return state["viz_image_path"]
                        else:
                            return None
                    except Exception as e:
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
    print(f"🎬 Features: Image Display + Streaming Response")
    
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