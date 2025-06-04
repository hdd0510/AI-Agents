#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Chat Interface with Long Short Term Memory - FIXED IMAGE DISPLAY
===================================================
Giao diện chatbot Gradio với khả năng nhớ ngắn hạn và dài hạn cho hệ thống AI y tế.
"""

import gradio as gr
import json
import sqlite3
import hashlib
import uuid
import datetime
from typing import Dict, List, Any, Optional, Tuple
import os
import tempfile
from pathlib import Path
import logging
import base64

# Import medical AI system
from medical_ai_agents import MedicalAISystem, MedicalGraphConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LongShortTermMemory:
    """Hệ thống memory với khả năng lưu trữ ngắn hạn và dài hạn."""
    
    def __init__(self, db_path: str = "medical_ai_memory.db"):
        self.db_path = db_path
        self.short_term_memory = {}  # Session-based memory
        self.init_database()
    
    def init_database(self):
        """Khởi tạo database cho long term memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bảng lưu conversations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                timestamp TEXT,
                query TEXT,
                response TEXT,
                image_analysis TEXT,
                polyp_count INTEGER,
                session_id TEXT
            )
        ''')
        
        # Bảng lưu user profiles
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                medical_history TEXT,
                preferences TEXT,
                last_visit TEXT,
                total_scans INTEGER DEFAULT 0
            )
        ''')
        
        # Bảng lưu medical patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS medical_patterns (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                pattern_type TEXT,
                pattern_data TEXT,
                frequency INTEGER DEFAULT 1,
                last_seen TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_session_memory(self, session_id: str) -> Dict[str, Any]:
        """Lấy short term memory cho session."""
        if session_id not in self.short_term_memory:
            self.short_term_memory[session_id] = {
                "conversation_history": [],
                "current_context": {},
                "user_info": {},
                "session_start": datetime.datetime.now().isoformat()
            }
        return self.short_term_memory[session_id]
    
    def add_to_short_term(self, session_id: str, interaction: Dict[str, Any]):
        """Thêm tương tác vào short term memory."""
        memory = self.get_session_memory(session_id)
        memory["conversation_history"].append({
            "timestamp": datetime.datetime.now().isoformat(),
            "interaction": interaction
        })
        
        # Keep only last 10 interactions in short term
        if len(memory["conversation_history"]) > 10:
            memory["conversation_history"] = memory["conversation_history"][-10:]
    
    def save_to_long_term(self, user_id: str, session_id: str, interaction: Dict[str, Any]):
        """Lưu tương tác quan trọng vào long term memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        interaction_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO conversations 
            (id, user_id, timestamp, query, response, image_analysis, polyp_count, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            interaction_id,
            user_id,
            datetime.datetime.now().isoformat(),
            interaction.get("query", ""),
            interaction.get("response", ""),
            json.dumps(interaction.get("analysis", {})),
            interaction.get("polyp_count", 0),
            session_id
        ))
        
        conn.commit()
        conn.close()
    
    def get_user_history(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Lấy lịch sử tương tác của user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, query, response, polyp_count 
            FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "timestamp": row[0],
                "query": row[1], 
                "response": row[2],
                "polyp_count": row[3]
            }
            for row in results
        ]
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]):
        """Cập nhật profile người dùng."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO user_profiles 
            (user_id, name, medical_history, preferences, last_visit, total_scans)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id,
            profile_data.get("name", ""),
            json.dumps(profile_data.get("medical_history", {})),
            json.dumps(profile_data.get("preferences", {})),
            datetime.datetime.now().isoformat(),
            profile_data.get("total_scans", 0)
        ))
        
        conn.commit()
        conn.close()
    
    def get_contextual_prompt(self, session_id: str, user_id: str) -> str:
        """Tạo prompt với context từ memory."""
        memory = self.get_session_memory(session_id)
        history = self.get_user_history(user_id, 3)
        
        context_parts = []
        
        # Add recent conversation context
        if memory["conversation_history"]:
            context_parts.append("Recent conversation context:")
            for conv in memory["conversation_history"][-3:]:
                interaction = conv["interaction"]
                if interaction.get("query"):
                    context_parts.append(f"- User asked: {interaction['query']}")
                if interaction.get("polyp_count", 0) > 0:
                    context_parts.append(f"- Found {interaction['polyp_count']} polyps")
        
        # Add user history
        if history:
            context_parts.append("User's medical scan history:")
            for h in history:
                if h["polyp_count"] > 0:
                    context_parts.append(f"- Previous scan found {h['polyp_count']} polyps")
        
        return "\n".join(context_parts) if context_parts else ""

class MedicalAIChatbot:
    """Medical AI Chatbot với Gradio interface."""
    
    def __init__(self):
        self.memory = LongShortTermMemory()
        self.medical_ai = self._initialize_medical_ai()
        self.active_sessions = {}
    
    def _initialize_medical_ai(self) -> MedicalAISystem:
        """Khởi tạo Medical AI System."""
        config = MedicalGraphConfig(
            device="cuda" if os.environ.get("CUDA_AVAILABLE") else "cpu",
        )
        return MedicalAISystem(config)
    
    def generate_user_id(self, username: str) -> str:
        """Tạo user ID từ username."""
        return hashlib.md5(username.encode()).hexdigest()
    
    def process_message(
        self, 
        message: str, 
        image: Optional[str], 
        chat_history: List[List[str]], 
        username: str,
        session_state: Dict[str, Any]
    ) -> Tuple[str, List[List[str]], Dict[str, Any]]:
        """Xử lý tin nhắn từ user."""
        
        # Generate session and user IDs
        session_id = session_state.get("session_id", str(uuid.uuid4()))
        user_id = self.generate_user_id(username)
        
        session_state["session_id"] = session_id
        session_state["user_id"] = user_id
        
        try:
            response_parts = []
            analysis_result = None
            
            # Get contextual information
            context = self.memory.get_contextual_prompt(session_id, user_id)
            
            # Process image if provided
            if image is not None:
                logger.info(f"Processing image analysis for user {username}")
                
                # Analyze image with Medical AI
                result = self.medical_ai.analyze(
                    image_path=image,
                    query=message,
                    medical_context={"user_context": context} if context else None
                )
                
                analysis_result = result
                
                if result.get("success", False):
                    # Create comprehensive response
                    if "final_answer" in result:
                        response_parts.append("🔍 **Kết quả phân tích hình ảnh:**")
                        response_parts.append(result["final_answer"])
                    
                    # Add detection details
                    if "agent_results" in result and "detector_result" in result["agent_results"]:
                        detector = result["agent_results"]["detector_result"]
                        if detector.get("success") and detector.get("count", 0) > 0:
                            response_parts.append(f"\n📊 **Chi tiết phát hiện:**")
                            response_parts.append(f"- Số lượng polyp: {detector['count']}")
                            response_parts.append(f"- Độ tin cậy: {detector['objects'][0]['confidence']:.2%}")
                            
                            # FIXED: Hiển thị ảnh visualization trong chat
                            if detector.get("visualization_base64") and detector.get("visualization_available"):
                                # Lưu base64 vào session_state để sử dụng sau
                                session_state["last_visualization"] = detector.get("visualization_base64")
                                
                                # Tạo data URL từ base64
                                img_data_url = f"data:image/png;base64,{detector.get('visualization_base64')}"
                                
                                # FIXED: Sử dụng HTML img tag thay vì markdown
                                response_parts.append(f"\n\n📊 **Kết quả phát hiện polyp:**")
                                response_parts.append(f'<img src="{img_data_url}" alt="Kết quả phát hiện polyp" style="max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0;">')
                                
                                # Lưu thông tin ảnh vào session_state
                                session_state["has_image_result"] = True
                                session_state["last_result_image_data"] = img_data_url
                    
                    # Add medical recommendations
                    response_parts.append("\n💡 **Khuyến nghị:**")
                    if result.get("polyp_count", 0) > 0:
                        response_parts.append("- Nên tham khảo ý kiến bác sĩ chuyên khoa")
                        response_parts.append("- Theo dõi định kỳ theo lịch hẹn")
                    else:
                        response_parts.append("- Duy trì lối sống lành mạnh")
                        response_parts.append("- Kiểm tra định kỳ theo khuyến nghị")
                else:
                    response_parts.append("❌ Có lỗi trong quá trình phân tích hình ảnh.")
                    response_parts.append(f"Chi tiết lỗi: {result.get('error', 'Unknown error')}")
            
            else:
                # Text-only conversation
                if context:
                    response_parts.append("💭 **Dựa trên thông tin trước đó:**")
                    response_parts.append(context[:200] + "..." if len(context) > 200 else context)
                
                response_parts.append("💬 **Trả lời:**")
                response_parts.append("Tôi có thể giúp bạn phân tích hình ảnh nội soi và trả lời các câu hỏi y tế. ")
                response_parts.append("Vui lòng tải lên hình ảnh để tôi có thể hỗ trợ tốt hơn.")
            
            final_response = "\n".join(response_parts)
            
            # Save to memory
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
            
            # Update chat history
            chat_history.append([message, final_response])
            
            return "", chat_history, session_state
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            error_response = f"❌ Xin lỗi, có lỗi xảy ra: {str(e)}"
            chat_history.append([message, error_response])
            return "", chat_history, session_state
    
    def get_user_stats(self, username: str) -> str:
        """Lấy thống kê của user."""
        user_id = self.generate_user_id(username)
        history = self.memory.get_user_history(user_id, 10)
        
        if not history:
            return "📊 **Thống kê của bạn:**\n- Chưa có lịch sử sử dụng"
        
        total_scans = len(history)
        total_polyps = sum(h["polyp_count"] for h in history)
        
        stats = [
            "📊 **Thống kê của bạn:**",
            f"- Tổng số lần kiểm tra: {total_scans}",
            f"- Tổng số polyp phát hiện: {total_polyps}",
            f"- Lần kiểm tra gần nhất: {history[0]['timestamp'][:10] if history else 'N/A'}"
        ]
        
        return "\n".join(stats)
    
    def create_interface(self) -> gr.Blocks:
        """Tạo giao diện Gradio với hiển thị ảnh đã được fix."""
        
        with gr.Blocks(
            title="Medical AI Assistant", 
            theme=gr.themes.Soft(),
            css="""
            .main-container { max-width: 1200px; margin: 0 auto; }
            .chat-container { height: 600px; }
            .upload-container { border: 2px dashed #ccc; padding: 20px; text-align: center; }
            /* Đảm bảo ảnh hiển thị đúng trong chat */
            .message img {
                max-width: 100%;
                height: auto;
                border-radius: 8px;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            """
        ) as interface:
            
            # State management
            session_state = gr.State({})
            
            gr.Markdown("""
            # 🏥 Medical AI Assistant
            ### Hệ thống AI hỗ trợ phân tích hình ảnh nội soi và tư vấn y tế
            
            **Hướng dẫn sử dụng:**
            1. Nhập tên của bạn để hệ thống có thể ghi nhớ
            2. Tải lên hình ảnh nội soi (nếu có)
            3. Đặt câu hỏi hoặc yêu cầu phân tích
            4. Xem kết quả và khuyến nghị từ AI
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface - FIXED: Enable HTML rendering
                    chatbot = gr.Chatbot(
                        label="💬 Cuộc trò chuyện với AI",
                        height=500,
                        show_copy_button=True,
                        elem_classes=["chat-container"],
                        layout="bubble",
                        render_markdown=True,  # Enable HTML rendering
                        sanitize_html=False,   # Allow HTML images
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="💭 Hãy mô tả triệu chứng hoặc đặt câu hỏi về hình ảnh...",
                            label="Tin nhắc của bạn",
                            scale=4,
                            lines=2
                        )
                        send_btn = gr.Button("📤 Gửi", variant="primary", scale=1)
                    
                    # Image upload
                    image_input = gr.Image(
                        label="🖼️ Tải lên hình ảnh nội soi",
                        type="filepath",
                        elem_classes=["upload-container"]
                    )
                
                with gr.Column(scale=1):
                    # User info panel
                    gr.Markdown("### 👤 Thông tin người dùng")
                    username_input = gr.Textbox(
                        label="Tên của bạn",
                        placeholder="Nhập tên để hệ thống ghi nhớ...",
                        value="Guest"
                    )
                    
                    stats_display = gr.Markdown("📊 Chưa có thống kê")
                    
                    stats_btn = gr.Button("Xem thống kê của tôi", variant="secondary")
                    
                    # Quick actions
                    gr.Markdown("### ⚡ Thao tác nhanh")
                    
                    with gr.Column():
                        quick_analysis_btn = gr.Button("🔍 Phân tích nhanh", size="sm")
                        history_btn = gr.Button("📜 Xem lịch sử", size="sm") 
                        clear_btn = gr.Button("🗑️ Xóa cuộc trò chuyện", size="sm")
                    
                    # Memory info
                    gr.Markdown("""
                    ### 🧠 Trí nhớ AI
                    - **Ngắn hạn**: Nhớ 10 tin nhắn gần nhất
                    - **Dài hạn**: Lưu trữ kết quả quan trọng
                    - **Cá nhân hóa**: Ghi nhớ lịch sử của bạn
                    """)
            
            # Event handlers
            def process_and_respond(message, image, history, username, state):
                return self.process_message(message, image, history, username, state)
            
            def update_stats(username):
                return self.get_user_stats(username)
            
            def clear_chat():
                return []
            
            def quick_analysis_prompt():
                return "Hãy phân tích hình ảnh này và cho tôi biết có phát hiện polyp không?"
            
            # Connect events
            send_btn.click(
                process_and_respond,
                inputs=[msg_input, image_input, chatbot, username_input, session_state],
                outputs=[msg_input, chatbot, session_state]
            )
            
            msg_input.submit(
                process_and_respond,
                inputs=[msg_input, image_input, chatbot, username_input, session_state],
                outputs=[msg_input, chatbot, session_state]
            )
            
            stats_btn.click(
                update_stats,
                inputs=[username_input],
                outputs=[stats_display]
            )
            
            clear_btn.click(
                clear_chat,
                outputs=[chatbot]
            )
            
            quick_analysis_btn.click(
                quick_analysis_prompt,
                outputs=[msg_input]
            )
        
        return interface

def launch_chatbot():
    """Khởi động chatbot."""
    chatbot = MedicalAIChatbot()
    interface = chatbot.create_interface()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=True,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    launch_chatbot()