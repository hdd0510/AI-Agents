#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Assistant
----------------
Enhanced interactive chatbot with multi-modal capabilities.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

        def _initialize_medical_ai(self) -> MedicalAISystem:
            """Khởi tạo hệ thống Medical AI với cấu hình đã cập nhật."""
            
            # Tạo cấu hình cho hệ thống
            medical_config = MedicalGraphConfig(
                device=self.app_config.get("medical_ai.device", "cuda"),
                detector_model_path=self.app_config.get("medical_ai.detector_model_path", "medical_ai_agents/weights/detect_best.pt"),
                vqa_model_path=self.app_config.get("medical_ai.vqa_model_path", "medical_ai_agents/weights/llava-med-mistral-v1.5-7b"),
                modality_classifier_path=self.app_config.get("medical_ai.modality_classifier_path", "medical_ai_agents/weights/modal_best.pt"),
                region_classifier_path=self.app_config.get("medical_ai.region_classifier_path", "medical_ai_agents/weights/location_best.pt"),
                
                # Thêm cấu hình RAG
                rag_storage_path=self.app_config.get("medical_ai.rag_storage_path", "./rag_storage"),
                
                # Thông số LLM
                llm_model=self.app_config.get("medical_ai.llm_model", "gpt-4o-mini"),
                llm_temperature=self.app_config.get("medical_ai.llm_temperature", 0.2),
                
                # Checkpointing (tùy chọn)
                checkpoint_dir=self.app_config.get("medical_ai.checkpoint_dir", "sessions")
            )
            
            # Khởi tạo hệ thống
            system = MedicalAISystem(config=medical_config)
            
            # Log để xác nhận
            logger = logging.getLogger(__name__)
            logger.info("Initialized Medical AI System with updated pipeline router")
            logger.info(f"RAG storage path: {medical_config.rag_storage_path}")
            
            return system

        # ---------------------------------------------------------------------
        # 8) ***XỬ LÝ UPLOAD TÀI LIỆU (giữ nguyên logic cũ)***
        # ---------------------------------------------------------------------
        def process_document_upload(self, files, session_state):
            """
            Nhận danh sách 'files' (Gradio File component), thêm vào hệ thống RAG,
            cập nhật session_state và trả về session_state đã cập nhật.
            """
            import logging, uuid, os
            logger = logging.getLogger(__name__)
            logger.info(f"Processing {len(files) if files else 0} uploaded documents")

            if not files:                           # Không có file → giữ nguyên state
                return session_state

            # --- Lấy (hoặc tạo) session_id ------------------------------------
            session_id = self._get_session_value(session_state, "session_id")
            if not session_id:
                session_id = str(uuid.uuid4())
                session_state = self._update_session_state(session_state, {
                    "session_id": session_id
                })
                logger.info(f"Created new session ID for document upload: {session_id}")

            # --- Lưu đường dẫn file ------------------------------------------
            file_paths = []
            for file in files:
                try:
                    file_path = file.name if hasattr(file, "name") else str(file)
                    logger.info(f"Document ready: {os.path.basename(file_path)}")
                    file_paths.append(file_path)
                except Exception as e:
                    logger.error(f"Error processing file: {e}")

            session_state = self._update_session_state(session_state, {
                "uploaded_documents": file_paths
            })

            # --- Đưa tài liệu vào RAG ----------------------------------------
            try:
                if not hasattr(self, "rag_agent"):
                    from medical_ai_agents.agents.rag import RAGAgent
                    self.rag_agent = RAGAgent()

                # KIỂM TRA: Xem vector database đã tồn tại và có tài liệu chưa
                has_existing_docs = self.rag_agent.has_documents()
                logger.info(f"Checking existing documents in vector database: {has_existing_docs}")

                # EMBEDDING: Xử lý các tài liệu mới
                result = self.rag_agent.add_documents(file_paths)

                if result.get("success"):
                    processed_files = result.get("processed_files", [])
                    processed_count = len(processed_files)
                    
                    logger.info(f"Successfully processed {processed_count} documents")
                    
                    # Thêm thông tin embedding vào state
                    session_state = self._update_session_state(session_state, {
                        "document_processing_result": result,
                        "embedding_result": {
                            "success": True,
                            "documents_processed": processed_files,
                            "message": f"Successfully processed {processed_count} documents",
                            "timestamp": time.time(),
                            "has_existing_documents": True
                        }
                    })
                    
                    # Đảm bảo document_qa được thêm vào required_tasks nếu chưa có
                    required_tasks = self._get_session_value(session_state, "required_tasks", [])
                    if "document_qa" not in required_tasks:
                        required_tasks.append("document_qa")
                        session_state = self._update_session_state(session_state, {
                            "required_tasks": required_tasks
                        })
                        
                    # Thêm vào execution_order nếu chưa có
                    execution_order = self._get_session_value(session_state, "execution_order", [])
                    if "document_qa" not in execution_order:
                        execution_order.append("document_qa")
                        session_state = self._update_session_state(session_state, {
                            "execution_order": execution_order
                        })
                        
                    # Đặt document_qa làm current_task để đảm bảo nó được ưu tiên
                    session_state = self._update_session_state(session_state, {
                        "current_task": "document_qa"
                    })
                    
                    logger.info("Document embedding completed successfully, RAG is ready")
                else:
                    logger.error(f"Document processing failed: {result.get('error', 'unknown')}")
                    session_state = self._update_session_state(session_state, {
                        "document_processing_result": result,
                        "embedding_result": {
                            "success": False,
                            "error": result.get("error", "Failed to embed documents"),
                            "message": "Failed to embed documents"
                        }
                    })
            except Exception as e:
                logger.error(f"Error in document processing: {e}")
                session_state = self._update_session_state(session_state, {
                    "document_processing_error": str(e),
                    "embedding_result": {
                        "success": False,
                        "error": f"Error in document processing: {e}",
                        "message": f"Error during document embedding: {str(e)}"
                    }
                })

            return session_state


        
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
                # Chỉ xử lý các phần tử là dictionary
                if not isinstance(entry, dict):
                    continue
                    
                query = entry.get("query")
                response = entry.get("response")
                
                if query and response and query not in existing_queries:
                    # Thêm vào history UI
                    new_history.insert(0, [query, response])
                    existing_queries.add(query)
            
            return new_history

        def _save_image_to_temp(self, image) -> str:
            """Lưu ảnh vào thư mục tạm."""
            import tempfile
            import os
            from PIL import Image
            import io
            import numpy as np
            
            if not image:
                logger.error("No image provided")
                return None
                
            try:
                # Kiểm tra xem image có phải đã là đường dẫn file không
                if isinstance(image, str) and os.path.isfile(image):
                    return image
                
                # Handle PIL Image objects directly from Gradio
                if hasattr(image, "__class__") and "PIL" in str(image.__class__):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                        image.save(tmp.name, format='JPEG')
                        return tmp.name
                
                # Xử lý cho trường hợp image là numpy array (từ Gradio)
                if isinstance(image, np.ndarray) or (hasattr(image, 'shape') and len(getattr(image, 'shape', [])) == 3):
                    # Đây là numpy array
                    img = Image.fromarray(image.astype(np.uint8) if hasattr(image, 'astype') else image)
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
                
                # Handle case where image has a name attribute (file-like object)
                if hasattr(image, "name") and os.path.isfile(image.name):
                    return image.name
                
                # Fallback: Lưu trực tiếp
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                    if hasattr(image, "save"):
                        image.save(tmp.name)
                    else:
                        tmp.write(image if isinstance(image, bytes) else str(image).encode('utf-8'))
                    return tmp.name
            except Exception as e:
                logger.error(f"Error saving image to temp: {str(e)}")
                return None
        
        def _save_visualization_to_file(self, base64_data: str, session_id: str) -> str:
            """Lưu dữ liệu base64 vào file ảnh và trả về đường dẫn tương đối."""
            import os
            import base64
            import time
            
            # Đảm bảo base64_data không chứa phần header của data URL
            if base64_data and "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]
                
            # Tạo thư mục nếu chưa tồn tại
            viz_dir = os.path.join("visualizations", session_id)
            os.makedirs(viz_dir, exist_ok=True)
            
            # Tạo tên file duy nhất
            filename = f"detect_{int(time.time())}.png"
            file_path = os.path.join(viz_dir, filename)
            
            try:
                # Giải mã và lưu file
                img_data = base64.b64decode(base64_data)
                with open(file_path, "wb") as f:
                    f.write(img_data)
                
                # Log để debug
                logger.info(f"Saved visualization to file: {file_path}")
                logger.info(f"File exists: {os.path.exists(file_path)}")
                logger.info(f"File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
                
                # Trả về đường dẫn tương đối
                return file_path
            except Exception as e:
                logger.error(f"Error saving visualization to file: {e}")
                return ""
                
        def _ensure_history_format(self, history):
            """Đảm bảo history luôn ở định dạng list of dictionaries."""
            if not history or len(history) == 0:
                return []
                
            # Nếu đã là list of dicts thì return ngay
            if isinstance(history[0], dict):
                return history
                
            # Nếu là list of lists (UI history), chuyển đổi sang format dictionary
            if isinstance(history[0], list):
                formatted_history = []
                for item in history:
                    if len(item) >= 2:
                        formatted_history.append({
                            "query": item[0],
                            "response": item[1],
                            "has_image": False,  # Mặc định
                            "timestamp": time.time()
                        })
                return formatted_history
                
            # Trường hợp không xác định, trả về list rỗng an toàn
            return []

        def process_message_streaming(self, message, image, history, username, session_state):
            """
            Streaming handler: nhận `message` (str), `image` (PIL/np/file-path hoặc None),
            cập nhật `history` (UI) và `session_state`, trả về ba giá trị (msg_out, history, state)
            dưới dạng generator cho Gradio.
            """
            import time, uuid, os, re, logging
            logger = logging.getLogger(__name__)

            # ----- 0. Tiền xử lý --------------------------------------------------------
            query = (message or "").strip()
            if not query:
                return "", history, session_state

            # ----- 1. Quản lý session ---------------------------------------------------
            sid = self._get_session_value(session_state, "session_id")
            if not sid:
                sid = self._load_persistent_session_id(username) or str(uuid.uuid4())
                session_state = self._update_session_state(session_state, {"session_id": sid})
                self._save_persistent_session_id(username, sid)

            uid = self.generate_user_id(username)
            session_state = self._update_session_state(session_state, {"user_id": uid})

            # ----- 2. Khôi phục lịch sử nếu cần ----------------------------------------
            if not self._get_session_value(session_state, "conversation_history"):
                conv_hist = self._load_conversation_history(sid)
                session_state = self._update_session_state(session_state, {
                    "conversation_history": conv_hist or []
                })

            # ----- 3. Thêm placeholder vào UI ------------------------------------------
            history.append([query, "⏳ Đang xử lý..."])
            yield "", history, session_state

            # ----- 4. Lấy ngữ cảnh bộ nhớ ngắn hạn -------------------------------------
            context_prompt = self.memory.get_contextual_prompt(sid, uid)

            # ----- 5. Phân nhánh ảnh / text --------------------------------------------
            try:
                # Đảm bảo conversation_history có định dạng đúng trước khi truyền vào analyze
                conv_history = self._ensure_history_format(
                    self._get_session_value(session_state, "conversation_history", [])
                )
                
                result = self.medical_ai.analyze(
                    image_path=self._save_image_to_temp(image) if image is not None else None,
                    query=query,
                    medical_context={"user_context": context_prompt} if context_prompt else None,
                    conversation_history=conv_history,
                    session_id=sid
                )
            except Exception as e:
                logger.error(e)
                history[-1][1] = "❌ Lỗi hệ thống, vui lòng thử lại."
                yield "", history, session_state
                return

            # ----- 6. Xử lý kết quả -----------------------------------------------------
            if not result.get("success", False):
                history[-1][1] = f"❌ {result.get('error', 'Unknown error')}"
                yield "", history, session_state
                return

            # Xử lý visualization nếu có
            has_visualization = False
            visualization_html = ""
            visualization_file_path = ""
            
            # Escape square brackets in response to prevent markdown rendering issues
            if "final_answer" in result:
                result["final_answer"] = result["final_answer"].replace("[", "\\[").replace("]", "\\]")
            
            if "response_chunks" in result and result["response_chunks"]:
                # Escape square brackets in each chunk
                result["response_chunks"] = [chunk.replace("[", "\\[").replace("]", "\\]") for chunk in result["response_chunks"]]
            
            if "agent_results" in result and "detection" in result["agent_results"]:
                detector = result["agent_results"]["detection"]
                logger.info(f"Direct detector keys: {list(detector.keys())}")
                logger.info(f"Has visualization_base64: {detector.get('visualization_base64') is not None}")
                
                # Check if we should show visualization based on the show_visualization flag
                show_visualization = detector.get("show_visualization", False)
                logger.info(f"Show visualization flag: {show_visualization}")
                
                if detector.get("visualization_base64") and show_visualization:
                    viz_base64 = detector["visualization_base64"]
                    has_visualization = True
                    logger.info(f"Found visualization in direct detector (len: {len(viz_base64) if viz_base64 else 0})")
                    
                    # Đảm bảo viz_base64 không chứa phần header của data URL
                    if viz_base64 and "," in viz_base64:
                        viz_base64 = viz_base64.split(",", 1)[1]
                    
                    # Lưu vào file thay vì embed trực tiếp vào response
                    file_path = self._save_visualization_to_file(viz_base64, sid)
                    visualization_file_path = file_path
                    
                    # Sử dụng định dạng Markdown cho hình ảnh
                    visualization_html = f"\n\n### 📊 Kết quả phát hiện:\n\n"
                    
                    # Dùng đường dẫn file thay vì base64 inline
                    if file_path and os.path.exists(file_path):
                        # Đảm bảo đường dẫn file là tương đối từ gốc để Gradio có thể xử lý
                        # Bỏ "visualizations/" ở đầu đường dẫn nếu có, vì thư mục này đã được khai báo trong allowed_paths
                        if file_path.startswith("visualizations/"):
                            display_path = file_path
                        else:
                            display_path = file_path
                        
                        # Log cho debug
                        logger.info(f"Visualization file path: {file_path}")
                        logger.info(f"Display path: {display_path}")
                        
                        # Sử dụng thẻ HTML img thay vì markdown - đảm bảo không có xung đột với markdown
                        visualization_html += f'<div style="text-align: left; margin: 20px 0;"><img src="file/{display_path}" alt="Detection Result" style="max-width: 90%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); display: inline-block;"></div>'
                    else:
                        visualization_html += "*Không thể hiển thị hình ảnh kết quả*"
                    
                    # Lưu vào session_state để có thể dùng lại sau này
                    session_state = self._update_session_state(session_state, {
                        "last_visualization_file": file_path
                    })
                elif detector.get("objects") and len(detector.get("objects", [])) > 0:
                    # If we have detection objects but visualization is not shown, add a note
                    logger.info(f"Found {len(detector.get('objects', []))} objects but visualization is not shown")
            
            # Kiểm tra visualization trong session_state (đã lưu từ trước)
            elif session_state.get("last_visualization_file"):
                file_path = session_state["last_visualization_file"]
                has_visualization = True
                visualization_file_path = file_path
                
                # Sử dụng định dạng Markdown cho hình ảnh
                visualization_html = f"\n\n### 📊 Kết quả phát hiện:\n\n"
                
                # Dùng đường dẫn file
                if file_path and os.path.exists(file_path):
                    # Đảm bảo đường dẫn file là tương đối từ gốc để Gradio có thể xử lý
                    # Bỏ "visualizations/" ở đầu đường dẫn nếu có, vì thư mục này đã được khai báo trong allowed_paths
                    if file_path.startswith("visualizations/"):
                        display_path = file_path
                    else:
                        display_path = file_path
                    
                    # Log cho debug
                    logger.info(f"Visualization file path: {file_path}")
                    logger.info(f"Display path: {display_path}")
                    
                    # Sử dụng thẻ HTML img thay vì markdown - đảm bảo không có xung đột với markdown
                    visualization_html += f'<div style="text-align: left; margin: 20px 0;"><img src="file/{display_path}" alt="Detection Result" style="max-width: 90%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); display: inline-block;"></div>'
                else:
                    visualization_html += "*Không thể hiển thị hình ảnh kết quả*"
            
            logger.info(f"Has visualization: {has_visualization}")
            
            # Streaming chunk nếu có
            if result.get("response_chunks"):
                answer = ""
                for chunk in result["response_chunks"]:
                    answer += chunk
                    history[-1][1] = answer
                    yield "", history, session_state
                    time.sleep(0.05)
                
                # Thêm visualization vào cuối nếu có
                if has_visualization:
                    logger.info("Adding visualization to streaming response")
                    history[-1][1] = answer + "\n\n" + visualization_html
                    yield "", history, session_state
            else:
                response_text = result.get("final_answer", "✅ Hoàn tất.")
                
                # Thêm visualization vào cuối nếu có
                if has_visualization:
                    logger.info("Adding visualization to standard response")
                    response_text += "\n\n" + visualization_html
                
                history[-1][1] = response_text
                yield "", history, session_state

            # ----- 7. Cập nhật conversation_history & bộ nhớ ---------------------------
            # Tạo phiên bản response không có base64 để lưu vào history
            response_for_history = history[-1][1]
            
            # Thay thế tham chiếu ảnh trong markdown với placeholder để không lưu base64
            if has_visualization and visualization_file_path:
                # Tạo một phiên bản response không chứa base64 nhưng vẫn giữ thông tin về visualization
                response_pattern = f"![Detection Result](/.*?)"
                response_replacement = f"![Detection Result](visualization:{os.path.basename(visualization_file_path)})"
                response_for_history = re.sub(response_pattern, response_replacement, response_for_history)
            
            interaction = {
                "query": query,
                "response": response_for_history,
                "has_image": image is not None,
                "has_visualization": has_visualization,
                "visualization_file": visualization_file_path if has_visualization else "",
                "timestamp": time.time()
            }
            conv_hist = self._get_session_value(session_state, "conversation_history", [])
            conv_hist.append(interaction)
            session_state = self._update_session_state(session_state, {
                "conversation_history": conv_hist
            })
            self._save_conversation_history(sid, conv_hist)
            self.memory.add_to_short_term(sid, interaction)

            # Lưu long-term nếu quan trọng
            if image is not None or any(k in query.lower() for k in ("polyp", "medical")):
                self.memory.save_to_long_term(uid, sid, interaction)

            # ----- 8. Lưu UI history ra file KHÔNG dùng nữa vì đã lưu conversation history ---
            # self._save_conversation_history(sid, history)

            yield "", history, session_state    

        def create_enhanced_interface(self):
            """
            Classic-layout UI: chat (scale=7) bên trái, upload ảnh + tài liệu (scale=3) bên phải.
            Mọi logic xử lý gốc (streaming, session, memory, clear, sync…) được giữ nguyên.
            """
            import gradio as gr, os, uuid, logging
            logger = logging.getLogger(__name__)

            # ---------- Cấu hình ----------
            theme        = self.app_config.get("ui.theme", "soft")
            chat_height  = self.app_config.get("ui.chat_height", 500)

            with gr.Blocks(title=self.app_config.get("app.title", "Medical AI Assistant"),
                        theme=theme,
                        css="""
                        .medical-chatbot {
                            font-family: 'Arial', sans-serif;
                        }
                        .medical-chatbot .message {
                            padding: 10px;
                        }
                        .medical-chatbot .message-wrap {
                            overflow-wrap: break-word;
                            word-break: break-word;
                        }
                        .gradio-container img {
                            max-height: 100%;
                            object-fit: contain;
                        }
                        /* Điều chỉnh kích thước ảnh trong khung upload */
                        .image-container img, .upload-preview img {
                            max-width: 70% !important;
                            max-height: 70% !important;
                            margin: 0 auto !important;
                            display: block !important;
                        }
                        /* Class tùy chỉnh cho ảnh nhỏ hơn */
                        .small-image-preview img {
                            max-width: 60% !important;
                            max-height: 60% !important;
                            transform: scale(0.8);
                            transform-origin: center;
                        }
                        /* Hiển thị ảnh trong chat */
                        .chatbot-container img {
                            border-radius: 8px;
                            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                            margin: 8px 0;
                            max-width: 90%;
                            display: block;
                            margin-left: 0 !important; /* Căn lề trái */
                            margin-right: auto !important;
                        }
                        /* Đảm bảo ảnh trong visualization hiển thị đúng */
                        .visualization-image {
                            width: 90%;
                            margin: 10px auto;
                            display: block;
                            border-radius: 8px;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.15);
                            margin-left: 0 !important; /* Căn lề trái */
                            margin-right: auto !important;
                        }
                        /* Căn giữa ảnh trong khung upload */
                        .medium-image-preview {
                            display: flex !important;
                            justify-content: center !important;
                            align-items: center !important;
                        }
                        .medium-image-preview > div {
                            display: flex !important;
                            justify-content: center !important;
                            width: 100% !important;
                        }
                        .medium-image-preview img {
                            margin: 0 auto !important;
                            display: block !important;
                            object-fit: contain !important;
                        }
                        /* Căn lề trái cho tất cả các hình ảnh trong chatbot */
                        .chatbot-container > div > div > div img {
                            margin-left: 0 !important;
                            margin-right: auto !important;
                            text-align: left !important;
                        }
                        /* Đảm bảo div chứa hình ảnh căn trái */
                        .chatbot-container div[style*="text-align"] {
                            text-align: left !important;
                        }
                        """
                        ) as interface:

                # ---- Header --------------------------------------------------------
                gr.Markdown("# 🩺 Medical AI Assistant")
                gr.Markdown("Interactive medical image analysis and consultation")

                # ---- Username ẩn (để lưu session) ----------------------------------
                username = gr.Textbox(value="default_user", visible=False)

                # ---- Khối chính ----------------------------------------------------
                with gr.Row():
                    # === Cột trái: Chatbot + nhập liệu =================================
                    with gr.Column(scale=7):
                        chatbot = gr.Chatbot(height=chat_height,
                                            show_copy_button=True,
                                            avatar_images=(None, None),
                                            bubble_full_width=False,
                                            line_breaks=True,
                                            render_markdown=True,
                                            sanitize_html=False,  # Allow HTML content
                                            elem_classes="medical-chatbot chatbot-container",
                                            show_label=False)

                        with gr.Row():
                            msg = gr.Textbox(
                                show_label=False,
                                placeholder="Nhập câu hỏi y khoa hoặc hỏi về ảnh nội soi...",
                                container=False,
                                lines=1,
                                max_lines=5
                            )
                            submit_btn = gr.Button("💬 Gửi", variant="primary", scale=0)

                    # === Cột phải: Ảnh + toolbar ======================================
                    with gr.Column(scale=3):
                        image_upload = gr.Image(
                            label="🖼️ Medical Image (optional)",
                            type="pil",
                            height=200,             # Giữ nguyên hoặc tăng lên nếu muốn
                            show_download_button=False,
                            show_label=True,
                            container=True,
                            elem_id="image_upload_box",   # ← gắn ID riêng để CSS dễ "bắt"
                        )
                        
                        gr.HTML("""
                        <style>
                        /* --- Căn giữa hoàn toàn khung upload --- */
                        #image_upload_box .upload-container,
                        #image_upload_box .upload-preview {
                            position: relative;
                            display: flex !important;
                            justify-content: center !important;
                            align-items: center !important;
                            width: 100% !important;
                            height: 100% !important;
                            overflow: hidden;
                            padding-top: 40px;  
                        }

                        /* --- Giới hạn kích thước ảnh và giữ aspect ratio --- */
                        #image_upload_box .upload-preview img {
                            max-width: 100% !important;
                            max-height: 100% !important;
                            object-fit: contain !important;   /* luôn nằm gọn trong khung */
                            margin: 0 auto !important;
                            display: block !important;
                        }
                        </style>
                        """)
                        
                        image_status = gr.Markdown("**No Image**")

                        doc_file = gr.File(
                            label="📄 Đính kèm tài liệu (PDF/DOCX/TXT)",
                            file_types=["pdf", "docx", "txt"],
                            file_count="multiple",
                            height=50
                        )
                        

                        # Nút xoá toàn bộ chat + sync
                        clear_btn = gr.Button("🗑️ Delete Chat History")
                        sync_history_btn = gr.Button("🔄 Sync History")

                # ---- State ẩn ------------------------------------------------------
                session_state = gr.State({})   # chứa session_id, conversation_history, ...
                image_state   = gr.State(None) # lưu đường dẫn ảnh tạm thời

                # ---------------------------------------------------------------------
                # 1) ***TẢI LỊCH SỬ TỰ ĐỘNG KHI MỞ GIAO DIỆN***
                # ---------------------------------------------------------------------
                def auto_sync_history(username):
                    new_state = {}
                    session_id = self._load_persistent_session_id(username)
                    if session_id:
                        new_state["session_id"] = session_id
                        new_state["user_id"]    = self.generate_user_id(username)

                        conv_hist = self._load_conversation_history(session_id)
                        if conv_hist:
                            # Đảm bảo conv_hist đúng định dạng
                            conv_hist = self._ensure_history_format(conv_hist)
                            new_state["conversation_history"] = conv_hist
                            # dựng UI history từ conv_hist nếu chưa có
                            ui_history = self.load_previous_session(username, session_id) or \
                                        [[e["query"], e["response"]] for e in conv_hist
                                        if e.get("query") and e.get("response")]
                            return ui_history, new_state
                    return [], new_state

                # ---------------------------------------------------------------------
                # 2) ***XỬ LÝ TRẠNG THÁI ẢNH***
                # ---------------------------------------------------------------------
                def update_image_status(img, _state):
                    if img is None:
                        return (
                            "**Chưa có ảnh**",
                            None                         # image_state
                        )

                    # Lấy tên file (nếu có)
                    file_name = (
                        os.path.basename(img) if isinstance(img, str) and os.path.exists(img)
                        else os.path.basename(img.name)  if hasattr(img, "name") and os.path.exists(img.name)
                        else "image"
                    )
                    # Trả về trạng thái mới
                    return (
                        f"**✅ Đã tải: {file_name}**",
                        img                          # image_state
                    )

                # 3.3  Callback xoá ảnh
                def clear_image(_state):
                    return (
                        None,                             # reset image_upload
                        "**Chưa có ảnh**",                # reset status
                        None,                             # reset image_state
                    )

                image_upload.change(
                    fn=update_image_status,                   # callback
                    inputs=[image_upload, image_state],
                    outputs=[image_status, image_state],  # không cập nhật lại image_upload
                    queue=False
                )

                # ---------------------------------------------------------------------
                # 3) ***UPLOAD TÀI LIỆU***
                # ---------------------------------------------------------------------
                def update_doc_status(file):
                    if not file:
                        return ""
                    elif isinstance(file, list):
                        if len(file) == 0:
                            return ""
                        elif len(file) == 1:
                            return f"📄 Đã đính kèm: {file[0].name}" if hasattr(file[0], 'name') else f"📄 Đã đính kèm: {file[0]}"
                        else:
                            return f"📄 Đã đính kèm: {len(file)} tài liệu"
                    else:
                        return f"📄 Đã đính kèm: {file.name}" if hasattr(file, 'name') else f"📄 Đã đính kèm: {file}"

                doc_file.change(update_doc_status, inputs=[doc_file], outputs=[msg])
                doc_file.change(self.process_document_upload,
                                inputs=[doc_file, session_state],
                                outputs=[session_state])

                # ---------------------------------------------------------------------
                # 4) ***GỬI TIN NHẮN***
                # ---------------------------------------------------------------------
                # Kích hoạt khi nhấn nút Gửi
                submit_btn.click(self.process_message_streaming,
                               inputs=[msg, image_upload, chatbot, username, session_state],
                               outputs=[msg, chatbot, session_state],
                               queue=True)
                
                # Kích hoạt khi nhấn phím Enter trong textbox
                msg.submit(self.process_message_streaming,
                          inputs=[msg, image_upload, chatbot, username, session_state],
                          outputs=[msg, chatbot, session_state],
                          queue=True)

                # ---------------------------------------------------------------------
                # 5) ***SYNC HISTORY KHI LOAD***
                # ---------------------------------------------------------------------
                interface.load(auto_sync_history,
                            inputs=[username],
                            outputs=[chatbot, session_state])

                # ---------------------------------------------------------------------
                # 6) ***XÓA TOÀN BỘ CUỘC TRÒ CHUYỆN***
                # ---------------------------------------------------------------------
                def clear_handler():
                    try:
                        sid  = self._get_session_value(session_state, "session_id")
                        uid  = self._get_session_value(session_state, "user_id")
                        usr  = username.value
                        new_sid = str(uuid.uuid4())

                        # reset session_state
                        st = self._update_session_state(session_state, {
                            "session_id": new_sid,
                            "conversation_history": [],
                            "user_id": uid
                        })

                        # xoá file lịch sử trên đĩa (nếu có)
                        for p in [f"sessions/history/{sid}.json",
                                f"sessions/conversation_history/{sid}.json",
                                f"sessions/{usr}.session"]:
                            if p and os.path.exists(p):
                                os.remove(p)

                        # clear memory
                        if hasattr(self.memory, "clear_short_term") and sid:
                            self.memory.clear_short_term(sid)
                        if hasattr(self.memory, "clear_long_term") and uid:
                            self.memory.clear_long_term(uid, sid)

                    except Exception as e:
                        logger.error(f"Clear error: {e}")
                        st = self._update_session_state(session_state, {
                            "session_id": str(uuid.uuid4()),
                            "conversation_history": []
                        })
                    # trả về giá trị reset
                    return st, "", [], None, "**Chưa có ảnh**", None

                clear_btn.click(clear_handler,
                                inputs=[],
                                outputs=[session_state, msg, chatbot,
                                        image_upload, image_status, image_state])

                # ---------------------------------------------------------------------
                # 7) ***ĐỒNG BỘ LẠI LỊCH SỬ***
                # ---------------------------------------------------------------------
                def sync_history_handler(history, user, st):
                    sid = self._get_session_value(st, "session_id") or \
                        self._load_persistent_session_id(user)
                    if not sid:
                        return history, st
                    conv = self._load_conversation_history(sid)
                    if not conv:
                        return history, st
                        
                    # Đảm bảo conv đúng định dạng
                    conv = self._ensure_history_format(conv)
                    st = self._update_session_state(st, {"conversation_history": conv})
                    synced = self._sync_ui_history_with_conversation(history, conv)
                    return synced, st

                sync_history_btn.click(sync_history_handler,
                                    inputs=[chatbot, username, session_state],
                                    outputs=[chatbot, session_state])

                # ---- Footer ---------------------------------------------------------
                gr.Markdown("<div style='text-align:center; color:#888;'>Medical AI Assistant — Version 1.0</div>")

            return interface

            
        def _load_persistent_session_id(self, username):
            """Load persistent session ID from disk"""
            if not username:
                return None
                
            session_file = self._get_persistent_session_path(username)
            
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

        def _get_persistent_session_path(self, username):
            """Get path to persistent session file."""
            if not username:
                return None
                
            # Ensure sessions directory exists
            os.makedirs("sessions", exist_ok=True)
            
            return os.path.join("sessions", f"{username}.session")

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
    
    # Tạo thư mục visualizations để lưu ảnh
    import os
    from pathlib import Path
    visualizations_dir = "visualizations"
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Tạo file test để đảm bảo thư mục hoạt động
    test_file_path = os.path.join(visualizations_dir, "test.txt")
    with open(test_file_path, "w") as f:
        f.write("Test file to verify visualizations directory is accessible")
    print(f"✅ Created test file at {test_file_path}")
    print(f"✅ File exists: {os.path.exists(test_file_path)}")
    
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
        
        # Get absolute path to visualizations directory
        viz_abs_path = os.path.abspath(visualizations_dir)
        print(f"📁 Visualizations directory: {viz_abs_path}")
        
        # Launch interface
        interface.launch(
            server_name=config.get("app.host"),
            server_port=config.get("app.port"),
            share=config.get("app.share"),
            debug=config.get("app.debug"),
            show_error=True,
            allowed_paths=[viz_abs_path]  # Sử dụng đường dẫn tuyệt đối
        )
        
    except Exception as e:
        print(f"❌ Error launching chatbot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()