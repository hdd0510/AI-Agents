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
import time

# Import medical AI system
from medical_ai_agents import MedicalAISystem, MedicalGraphConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LongShortTermMemory:
    """Quản lý short term và long term memory cho Medical AI."""
    
    def __init__(self, storage_path: str = "data/memory"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Short-term memory storage in RAM
        self.short_term_memory = {}
        self._load_short_term_memory()
        
        # Set up logger
        self.logger = logging.getLogger("memory.manager")
        
    def _load_short_term_memory(self):
        """Load short-term memory from disk if available."""
        short_term_path = os.path.join(self.storage_path, "short_term.json")
        if os.path.exists(short_term_path):
            try:
                with open(short_term_path, 'r', encoding='utf-8') as f:
                    self.short_term_memory = json.load(f)
                    # Filter out expired sessions (older than 24 hours)
                    now = time.time()
                    self.short_term_memory = {
                        session_id: data 
                        for session_id, data in self.short_term_memory.items()
                        if now - data.get("created_at", 0) < 86400  # 24 hours
                    }
                    logging.info(f"Loaded {len(self.short_term_memory)} sessions from short-term memory")
            except Exception as e:
                logging.error(f"Error loading short-term memory: {str(e)}")
                self.short_term_memory = {}
    
    def _save_short_term_memory(self):
        """Save short-term memory to disk."""
        short_term_path = os.path.join(self.storage_path, "short_term.json")
        try:
            with open(short_term_path, 'w', encoding='utf-8') as f:
                json.dump(self.short_term_memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Error saving short-term memory: {str(e)}")
    
    def get_session_memory(self, session_id: str) -> Dict[str, Any]:
        """Get or create session memory."""
        if session_id not in self.short_term_memory:
            self.short_term_memory[session_id] = {
                "created_at": time.time(),
                "interactions": []
            }
        return self.short_term_memory[session_id]
    
    def add_to_short_term(self, session_id: str, interaction: dict) -> None:
        """Add an interaction to short-term memory."""
        logger = logging.getLogger("memory.short_term")
        
        # Ensure the interaction has the required fields
        if "query" not in interaction or not interaction["query"]:
            logger.warning("Tried to save interaction without a query!")
            if "analysis" in interaction and interaction["analysis"] and "raw_query" in interaction["analysis"]:
                # Try to recover query from analysis
                interaction["query"] = interaction["analysis"]["raw_query"]
                logger.info(f"Recovered query from analysis: {interaction['query'][:30]}...")
            
        # Create or get session data
        if session_id not in self.short_term_memory:
            self.short_term_memory[session_id] = {
                "created_at": time.time(),
                "interactions": []
            }
        
        # Add timestamp if not present
        if "timestamp" not in interaction:
            interaction["timestamp"] = time.time()
            
        # Add interaction to memory
        self.short_term_memory[session_id]["interactions"].append(interaction)
        logger.info(f"Added interaction to short-term memory for session {session_id}: {interaction.get('query', 'Unknown')[:30]}...")
        
        # Persist to storage
        self._save_short_term_memory()
    
    def save_to_long_term(self, user_id: str, session_id: str, interaction: Dict[str, Any]) -> None:
        """Save important interaction to long-term memory."""
        logger = logging.getLogger("memory.long_term")
        
        # Ensure storage directory exists
        user_dir = os.path.join(self.storage_path, "users", user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Create a unique ID for this interaction
        interaction_id = f"{int(time.time())}_{session_id}"
        
        # Add metadata
        interaction_to_save = interaction.copy()
        interaction_to_save.update({
            "interaction_id": interaction_id,
            "user_id": user_id,
            "session_id": session_id,
            "saved_at": time.time()
        })
        
        # Save to user's history
        history_file = os.path.join(user_dir, "history.jsonl")
        try:
            with open(history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(interaction_to_save, ensure_ascii=False) + '\n')
            logger.info(f"Saved interaction {interaction_id} to long-term memory for user {user_id}")
        except Exception as e:
            logger.error(f"Error saving to long-term memory: {str(e)}")
    
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
    
    def get_contextual_prompt(self, session_id: str, user_id: str = None) -> str:
        """Get contextual information for prompting."""
        context_parts = []
        
        # Get recent interactions from short-term memory
        if session_id in self.short_term_memory:
            session_data = self.short_term_memory[session_id]
            recent_interactions = session_data.get("interactions", [])[-3:]  # Last 3 interactions
            
            if recent_interactions:
                context_parts.append("Based on our recent conversation:")
                
                for interaction in recent_interactions:
                    query = interaction.get("query", "")
                    response = interaction.get("response", "")
                    
                    # Truncate long responses
                    if len(response) > 150:
                        response = response[:147] + "..."
                    
                    if query:
                        context_parts.append(f"You asked: {query}")
                    if response:
                        context_parts.append(f"I responded: {response}")
            
        # Get medical context if user_id is provided
        if user_id:
            user_dir = os.path.join(self.storage_path, "users", user_id)
            history_file = os.path.join(user_dir, "history.jsonl")
            
            if os.path.exists(history_file):
                try:
                    # Read last 5 entries from history file
                    recent_history = []
                    with open(history_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            recent_history.append(json.loads(line))
                            if len(recent_history) >= 5:
                                recent_history = recent_history[-5:]  # Keep only last 5
                    
                    # Extract medical findings
                    findings = []
                    for entry in recent_history:
                        # Check for polyp findings
                        polyp_count = entry.get("polyp_count", 0)
                        if polyp_count > 0:
                            findings.append(f"You previously had {polyp_count} polyp(s) detected.")
                    
                    if findings:
                        context_parts.append("\nMedical context:")
                        context_parts.extend(findings)
                        
                except Exception as e:
                    logging.error(f"Error reading long-term memory: {str(e)}")
        
        # Combine context
        return "\n".join(context_parts) if context_parts else ""

    def clear_session(self, session_id: str) -> None:
        """Clear short-term memory for a session."""
        if session_id in self.short_term_memory:
            del self.short_term_memory[session_id]
            self._save_short_term_memory()
            logging.info(f"Cleared short-term memory for session {session_id}")

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session to use with LLM context."""
        if session_id not in self.short_term_memory:
            return []
        
        # Extract conversation pairs from interactions
        conversations = []
        for interaction in self.short_term_memory[session_id].get("interactions", []):
            if "query" in interaction and interaction.get("query"):
                conversations.append({
                    "query": interaction["query"],
                    "response": interaction.get("response", ""),
                    "timestamp": interaction.get("timestamp", time.time()),
                    "session_id": session_id
                })
        
        return conversations

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
            
            # Get session memory
            session_memory = self.memory.get_session_memory(session_id)
            
            # Get conversation history from memory or initialize
            conversation_history = session_memory.get("conversation_history", [])
            
            # Debug conversation history state
            logger.info(f"[MEMORY] Processing message, current conversation history has {len(conversation_history)} entries")
            if len(conversation_history) > 0:
                logger.info(f"[MEMORY] Last history entry: {conversation_history[-1].get('query', 'NO_QUERY')[:30]}...")
            
            # Initialize system message if this is a new conversation
            if not conversation_history:
                logger.info("[MEMORY] Initializing new conversation history")
                # Add a welcome message as the first system message
                conversation_history.append({
                    "query": "",  # Empty query because this isn't a user question
                    "response": "Hello! I'm your Medical AI Assistant specializing in healthcare consultation. How can I assist you with your medical questions today?",
                    "timestamp": time.time(),
                    "is_system": True,  # Clearly mark as system message
                    "type": "init"  # Add a type to identify this as initialization
                })
                session_memory["conversation_history"] = conversation_history
                logger.info(f"[MEMORY] Added system welcome message, history now has {len(conversation_history)} entries")
            
            # Get contextual information
            context = self.memory.get_contextual_prompt(session_id, user_id)
            
            # Process image if provided
            if image is not None:
                logger.info(f"Processing image analysis for user {username}")
                
                # Analyze image with Medical AI
                result = self.medical_ai.analyze(
                    image_path=image,
                    query=message,
                    medical_context={"user_context": context} if context else None,
                    conversation_history=conversation_history
                )
                
                analysis_result = result
                
                # Debug conversation history before and after
                logger.info(f"[IMAGE] Memory conversation history BEFORE: {len(conversation_history)} entries")
                if conversation_history:
                    for i, entry in enumerate(conversation_history[-2:]):  # Show last 2 entries
                        logger.info(f"[IMAGE] BEFORE - ENTRY {i}:")
                        logger.info(f"  - TYPE: {'SYSTEM' if entry.get('is_system', False) else 'USER'}")
                        logger.info(f"  - QUERY: {entry.get('query', 'None')[:30]}...")
                        resp = entry.get('response', 'None')
                        resp_preview = resp[:30] + "..." if resp and len(resp) > 30 else resp
                        logger.info(f"  - RESPONSE: {resp_preview}")
                
                # Update conversation history from the system response
                if "conversation_history" in result:
                    # Clean up only pending entries that match the current query
                    clean_history = []
                    for entry in conversation_history:
                        if entry.get("is_pending", False) and entry.get("query") == message:
                            logger.info(f"Removing pending entry for query: {message[:30]}...")
                        else:
                            clean_history.append(entry)
                    
                    # Get the newest entry from the result
                    new_entries = []
                    if result["conversation_history"]:
                        latest_entry = result["conversation_history"][-1]
                        if latest_entry.get("query") == message:
                            new_entries.append(latest_entry)
                            logger.info(f"Adding new complete entry for query: {message[:30]}...")
                    
                    # Create final merged history
                    conversation_history = clean_history + new_entries
                    session_memory["conversation_history"] = conversation_history
                    
                    # Debug updated conversation history
                    logger.info(f"[IMAGE] Memory conversation history AFTER: {len(conversation_history)} entries")
                    if conversation_history:
                        for i, entry in enumerate(conversation_history[-1:]):  # Show last entry
                            logger.info(f"[IMAGE] AFTER - ENTRY {i}:")
                            logger.info(f"  - QUERY: {entry.get('query', 'None')[:30]}...")
                            resp = entry.get('response', 'None')
                            resp_preview = resp[:30] + "..." if resp and len(resp) > 30 else resp
                            logger.info(f"  - RESPONSE: {resp_preview}")
                
                if result.get("success", False):
                    # Chỉ hiển thị final_answer nếu có
                    if "final_answer" in result:
                        response_parts.append("🔍 **Kết quả phân tích hình ảnh:**")
                        response_parts.append(result["final_answer"])
                    else:
                        response_parts.append("❌ Không có kết quả tổng hợp từ hệ thống. Vui lòng thử lại hoặc liên hệ hỗ trợ.")
                    
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
                
                # Process text-only query with Medical AI
                result = self.medical_ai.analyze(
                    query=message,
                    medical_context={"user_context": context} if context else None,
                    conversation_history=conversation_history
                )
                
                analysis_result = result
                
                # Debug conversation history before and after
                logger.info(f"Memory conversation history BEFORE: {len(conversation_history)} entries")
                if conversation_history:
                    for i, entry in enumerate(conversation_history[-2:]):  # Show last 2 entries
                        logger.info(f"BEFORE - ENTRY {i}:")
                        logger.info(f"  - TYPE: {'SYSTEM' if entry.get('is_system', False) else 'USER'}")
                        logger.info(f"  - QUERY: {entry.get('query', 'None')[:30]}...")
                        resp = entry.get('response', 'None')
                        resp_preview = resp[:30] + "..." if resp and len(resp) > 30 else resp
                        logger.info(f"  - RESPONSE: {resp_preview}")
                
                # Update conversation history from the system response
                if "conversation_history" in result:
                    # Clean up only pending entries that match the current query
                    clean_history = []
                    for entry in conversation_history:
                        if entry.get("is_pending", False) and entry.get("query") == message:
                            logger.info(f"Removing pending entry for query: {message[:30]}...")
                        else:
                            clean_history.append(entry)
                    
                    # Get the newest entry from the result
                    new_entries = []
                    if result["conversation_history"]:
                        latest_entry = result["conversation_history"][-1]
                        if latest_entry.get("query") == message:
                            new_entries.append(latest_entry)
                            logger.info(f"Adding new complete entry for query: {message[:30]}...")
                    
                    # Create final merged history
                    conversation_history = clean_history + new_entries
                    session_memory["conversation_history"] = conversation_history
                    
                    # Debug updated conversation history
                    logger.info(f"Memory conversation history AFTER: {len(conversation_history)} entries")
                    if conversation_history:
                        for i, entry in enumerate(conversation_history[-1:]):  # Show last entry
                            logger.info(f"AFTER - ENTRY {i}:")
                            logger.info(f"  - QUERY: {entry.get('query', 'None')[:30]}...")
                            resp = entry.get('response', 'None')
                            resp_preview = resp[:30] + "..." if resp and len(resp) > 30 else resp
                            logger.info(f"  - RESPONSE: {resp_preview}")
                
                if result.get("success", False) and "final_answer" in result:
                    response_parts.append("💬 **Response:**")
                    response_parts.append(result["final_answer"])
                else:
                    response_parts.append("💬 **Response:**")
                    response_parts.append("I can help you analyze endoscopy images and answer medical questions.")
                    response_parts.append("Please upload an image so I can provide better assistance.")
            
            final_response = "\n".join(response_parts)
            
            # Save to memory - we're now using the updated conversation history from the AI system
            # instead of manually tracking the conversation
            
            # Save important interactions to long term
            if image is not None or "polyp" in message.lower():
                interaction = {
                    "query": message,
                    "response": final_response,
                    "has_image": image is not None,
                    "analysis": analysis_result,
                    "polyp_count": analysis_result.get("polyp_count", 0) if analysis_result else 0
                }
                self.memory.save_to_long_term(user_id, session_id, interaction)
            
            # Update chat history
            chat_history.append([message, final_response])
            
            return "", chat_history, session_state
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            error_response = f"❌ Sorry, an error occurred: {str(e)}"
            chat_history.append([message, error_response])
            return "", chat_history, session_state
    
    def get_user_stats(self, username: str) -> str:
        """Get user statistics."""
        user_id = self.generate_user_id(username)
        history = self.memory.get_user_history(user_id, 10)
        
        if not history:
            return "📊 **Your Statistics:**\n- No usage history available"
        
        total_scans = len(history)
        total_polyps = sum(h["polyp_count"] for h in history)
        
        stats = [
            "📊 **Your Statistics:**",
            f"- Total scans: {total_scans}",
            f"- Total polyps detected: {total_polyps}",
            f"- Last scan: {history[0]['timestamp'][:10] if history else 'N/A'}"
        ]
        
        return "\n".join(stats)
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface with fixed image display."""
        
        with gr.Blocks(
            title="Medical AI Assistant", 
            theme=gr.themes.Soft(),
            css="""
            .main-container { max-width: 1200px; margin: 0 auto; }
            .chat-container { height: 600px; }
            .upload-container { border: 2px dashed #ccc; padding: 20px; text-align: center; }
            /* Ensure images display correctly in chat */
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
            ### AI system for endoscopy image analysis and medical consultation
            
            **Usage Guide:**
            1. Enter your name so the system can remember you
            2. Upload an endoscopy image (if available)
            3. Ask questions or request analysis
            4. Review results and recommendations from the AI
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Chat interface - FIXED: Enable HTML rendering
                    chatbot = gr.Chatbot(
                        label="💬 Conversation with AI",
                        height=500,
                        show_copy_button=True,
                        elem_classes=["chat-container"],
                        layout="bubble",
                        render_markdown=True,  # Enable HTML rendering
                        sanitize_html=False,   # Allow HTML images
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="💭 Describe symptoms or ask questions about the image...",
                            label="Your message",
                            scale=4,
                            lines=2
                        )
                        send_btn = gr.Button("📤 Send", variant="primary", scale=1)
                    
                    # Image upload
                    image_input = gr.Image(
                        label="🖼️ Upload endoscopy image",
                        type="filepath",
                        elem_classes=["upload-container"]
                    )
                
                with gr.Column(scale=1):
                    # User info panel
                    gr.Markdown("### 👤 User Information")
                    username_input = gr.Textbox(
                        label="Your name",
                        placeholder="Enter your name for the system to remember...",
                        value="Guest"
                    )
                    
                    stats_display = gr.Markdown("📊 No statistics available")
                    
                    stats_btn = gr.Button("View my statistics", variant="secondary")
                    
                    # Quick actions
                    gr.Markdown("### ⚡ Quick Actions")
                    
                    with gr.Column():
                        quick_analysis_btn = gr.Button("🔍 Quick Analysis", size="sm")
                        history_btn = gr.Button("📜 View History", size="sm") 
                        clear_btn = gr.Button("🗑️ Clear Conversation", size="sm")
                    
                    # Memory info
                    gr.Markdown("""
                    ### 🧠 AI Memory
                    - **Short-term**: Remembers last 10 messages
                    - **Long-term**: Stores important results
                    - **Personalization**: Remembers your history
                    """)
            
            # Event handlers
            def process_and_respond(message, image, history, username, state):
                return self.process_message(message, image, history, username, state)
            
            def update_stats(username):
                return self.get_user_stats(username)
            
            def clear_chat():
                return []
            
            def quick_analysis_prompt():
                return "Please analyze this image and tell me if there are any polyps detected?"
            
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