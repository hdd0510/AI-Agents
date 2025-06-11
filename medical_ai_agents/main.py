"""
Medical AI System - Main Entry Point (MODIFIED for multi-task support)
-----------------------------------
điểm vào chính với multi-task execution support.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List
import uuid
import time

from medical_ai_agents.config import MedicalGraphConfig
from medical_ai_agents.graph.pipeline import create_medical_ai_graph

class EnhancedMedicalAISystem:
    """
    Medical AI System với multi-task execution support.
    """
    
    def __init__(self, config: MedicalGraphConfig = None):
        """Initialize the Medical AI System."""
        self.config = config or MedicalGraphConfig()
        self.graph = create_medical_ai_graph(self.config)
        self.logger = logging.getLogger("enhanced-medical-ai-system")
        self.logger.info(f"Initialized Medical AI System: {self.config.name}")
    
    def analyze(self, 
               image_path: Optional[str] = None,
               query: Optional[str] = None, 
               medical_context: Optional[Dict[str, Any]] = None,
               conversation_history: Optional[List[Dict[str, Any]]] = None,
               session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main analysis method that dispatches to appropriate workflow.
        """
        # Ensure we have a valid conversation history
        if conversation_history is None:
            conversation_history = []
        
        # self.logger.info(f"Initial conversation history length: {len(conversation_history)}")
        
        # If no session_id provided, try to extract from conversation history
        if session_id is None and conversation_history:
            for entry in conversation_history:
                if "session_id" in entry:
                    session_id = entry["session_id"]
                    # self.logger.info(f"Extracted session_id from conversation history: {session_id}")
                    break
        
        # If still no session_id, generate one
        if session_id is None:
            session_id = str(uuid.uuid4())
            self.logger.info(f"Generated new session ID: {session_id}")
        
        # Debug conversation history
        # self.logger.info(f"Conversation history contains {len(conversation_history)} entries")
        
        # Initialize history with system welcome message if empty
        history = conversation_history
        if not history:
            # self.logger.info("Initializing new conversation with system welcome message")
            history = [{
                "query": "",  # Empty query because this isn't a user question
                "response": "Hello! I'm your Medical AI Assistant specializing in healthcare consultation. How can I assist you with your medical questions today?",
                "timestamp": time.time(),
                "is_system": True,  # Clearly mark as system message
                "type": "init",  # Add a type to identify this as initialization
                "session_id": session_id  # Add session_id to the initial system message
            }]
            # self.logger.info(f"System welcome message added to history - New history length: {len(history)}")
        
        # Validate inputs
        if not image_path and not query:
            return {
                "error": "Either image_path or query must be provided", 
                "success": False
            }
        
        # Only validate image if provided
        if image_path and not os.path.exists(image_path):
            return {
                "error": f"Image not found: {image_path}", 
                "success": False
            }
        
        # Check query validity and provide default if needed
        if query and not isinstance(query, str):
            self.logger.warning(f"Query is not a string: {type(query)}, converting to string")
            query = str(query)  # Convert to string if not already
        
        # Create initial state
        from medical_ai_agents.config import SystemState
        
        # Use provided conversation history or initialize empty list
        history = conversation_history or []
        
        # Create a persistent session_id early so we can use it
        session_id = str(uuid.uuid4())
        if history and len(history) > 0:
            # Try to reuse the session_id from conversation history if available
            session_entry = next((entry for entry in history if "session_id" in entry), None)
            if session_entry:
                session_id = session_entry.get("session_id", session_id)
                self.logger.info(f"Reusing existing session ID: {session_id}")
            else:
                self.logger.info(f"Created new session ID: {session_id}")
        else:
            self.logger.info(f"No history found, using new session ID: {session_id}")
            
        # Debug conversation history
        # self.logger.info(f"Conversation history received: {len(history)} entries")
        
        # CRITICAL FIX: Logic bug - this code was resetting conversation history every time!
        # Only initialize with welcome message if history is empty
        if not history or len(history) == 0:
            # Log the content of each entry in history for debugging
            self.logger.info("No history found. Initializing with welcome message")
            
            # Initialize with system welcome message since this is a new conversation
            history = [{
                "query": "",  # Empty query because this isn't a user question
                "response": "Hello! I'm your Medical AI Assistant specializing in healthcare consultation. How can I assist you with your medical questions today?",
                "timestamp": time.time(),
                "is_system": True,  # Clearly mark as system message
                "type": "init",  # Add a type to identify this as initialization
                "session_id": session_id  # Add session_id to the initial system message
            }]
            self.logger.info("System welcome message added to new conversation")
        else:
            # If history already exists, log what we have
            self.logger.info(f"Using existing conversation history with {len(history)} entries")
            # Log the last entry for debugging
            if history[-1].get("query"):
                self.logger.info(f"Last history entry: '{history[-1].get('query', '')[:30]}...'")
            # Don't reset history!
        
        # Clean up any pending entries that match the current query
        if query:
            history = [entry for entry in history if not (entry.get("is_pending", False) and entry.get("query") == query)]
            # self.logger.info(f"After removing pending entries, history length: {len(history)}")
            
            # Add current query as pending entry for debugging purposes
            history.append({
                "query": query,
                "response": "Processing...",
                "timestamp": time.time(),
                "is_pending": True,
                "session_id": session_id
            })
            # self.logger.info(f"Added current query to history for debugging: {query[:30]}...")
            # self.logger.info(f"History now has {len(history)} entries with pending entry")
        else:
            # self.logger.info("No query provided, not adding pending entry to history")
            pass
        
        initial_state: SystemState = {
            "image_path": image_path or "",
            "query": query,
            "raw_query": query,  # Add raw query for redundancy
            "medical_context": medical_context,
            "session_id": session_id,
            "start_time": time.time(),
            "is_text_only": image_path is None,
            "conversation_history": history,
            
            # Multi-task initialization
            "required_tasks": [],
            "completed_tasks": [],
            "execution_order": [],
            "current_task": None
        }
        
        try:
            # Run the graph
            final_state = None
            for event in self.graph.stream(initial_state):
                self.logger.debug(f"step completed: {list(event.keys())}")
                for node_name, state in event.items():
                    final_state = state
            
            # Check final state
            if final_state is None:
                return {
                    "error": "graph execution failed - no final state",
                    "success": False,
                    "session_id": initial_state["session_id"]
                }
            
            # Create output directory if needed
            if self.config.output_path and image_path:
                session_dir = os.path.join(self.config.output_path, final_state.get("session_id", "unknown"))
                os.makedirs(session_dir, exist_ok=True)
                
                # Save result
                if "final_result" in final_state:
                    result_path = os.path.join(session_dir, "result.json")
                    with open(result_path, 'w', encoding='utf-8') as f:
                        json.dump(final_state["final_result"], f, ensure_ascii=False, indent=2)
            
            # Return final result
            if "final_result" in final_state and final_state["final_result"]:
                final_result = final_state["final_result"]
                
                # Add updated conversation history to the result
                final_result["conversation_history"] = final_state.get("conversation_history", [])
                
                # Fix lỗi trường hợp image không lưu conversation_history
                if image_path and (not "conversation_history" in final_result or len(final_result["conversation_history"]) == 0):
                    self.logger.warning("Image query but conversation_history is empty, creating manual entry")
                    # Tạo một entry cho conversation history nếu không có
                    if query and query.strip():
                        new_entry = {
                            "query": query,
                            "response": final_result.get("final_answer", ""),
                            "timestamp": time.time(),
                            "has_image": True,
                            "session_id": final_result.get("session_id", "")
                        }
                        # Tạo hoặc cập nhật conversation_history
                        if "conversation_history" not in final_result:
                            final_result["conversation_history"] = []
                        final_result["conversation_history"].append(new_entry)
                        self.logger.info(f"Manually added image query to conversation_history: '{query[:30]}...'")
                
                # Log conversation history in final result
                history_entries = final_result.get("conversation_history", [])
                self.logger.info(f"Final result contains {len(history_entries)} conversation history entries")
                if history_entries:
                    for i, entry in enumerate(history_entries[-2:]):  # Log the last 2 entries
                        self.logger.info(f"Final history entry {i}: query='{entry.get('query', 'None')[:30]}...', "
                                        f"is_system={entry.get('is_system', False)}, "
                                        f"is_pending={entry.get('is_pending', False)}")
                
                # Thêm final_answer để app.py có thể sử dụng
                if "response" in final_result and final_result["response"]:
                    final_result["final_answer"] = final_result["response"]
                elif "agent_results" in final_result:
                    # Fallback từ các kết quả agent nếu không có response tổng hợp
                    agent_results = final_result["agent_results"]
                    if "region_result" in agent_results and agent_results["region_result"].get("success", False):
                        region = agent_results["region_result"]
                        final_result["final_answer"] = f"Vùng giải phẫu: {region.get('class_name', 'Unknown')} ({region.get('confidence', 0.0):.1%})\n\n{region.get('analysis', '')}"
                    elif "vqa_result" in agent_results and agent_results["vqa_result"].get("success", False):
                        final_result["final_answer"] = agent_results["vqa_result"].get("answer", "")
                
                return final_result
            else:
                # fallback result
                fallback_result = {
                    "success": True,
                    "session_id": final_state.get("session_id", ""),
                    "task_type": final_state.get("task_type", "unknown"),
                    "query": final_state.get("query", ""),
                    "is_text_only": final_state.get("is_text_only", False),
                    "processing_time": time.time() - initial_state["start_time"],
                    "conversation_history": final_state.get("conversation_history", []),
                    
                    # Multi-task fallback info
                    "multi_task_analysis": {
                        "required_tasks": final_state.get("required_tasks", []),
                        "completed_tasks": final_state.get("completed_tasks", []),
                        "execution_order": final_state.get("execution_order", [])
                    }
                }
                
                # Add agent results if available
                agent_results = {}
                if "detector_result" in final_state:
                    agent_results["detector_result"] = final_state["detector_result"]
                if "modality_result" in final_state:
                    agent_results["modality_result"] = final_state["modality_result"]
                if "region_result" in final_state:
                    agent_results["region_result"] = final_state["region_result"]
                if "vqa_result" in final_state:
                    print('-'*100)
                    print(final_state["vqa_result"])
                    print('-'*100)
                    agent_results["vqa_result"] = final_state["vqa_result"]
                
                fallback_result["agent_results"] = agent_results
                
                # Add legacy fields for compatibility
                if "vqa_result" in agent_results:
                    vqa_result = agent_results["vqa_result"]
                    if vqa_result and vqa_result.get("success", False):
                        fallback_result["answer"] = vqa_result.get("answer", "")
                        fallback_result["answer_confidence"] = vqa_result.get("confidence", 0.0)
                
                if "detector_result" in agent_results:
                    detector_result = agent_results["detector_result"]
                    if detector_result and detector_result.get("success", False):
                        fallback_result["polyps"] = detector_result.get("objects", [])
                        fallback_result["polyp_count"] = len(detector_result.get("objects", []))
                    else:
                        fallback_result["polyps"] = []
                        fallback_result["polyp_count"] = 0
                
                # Check for errors
                if "error" in final_state:
                    fallback_result["error"] = final_state["error"]
                    fallback_result["success"] = False
                
                return fallback_result
            
        except Exception as e:
            import traceback
            error_msg = f"analysis failed: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_msg)
            return {
                "error": error_msg,
                "success": False,
                "session_id": initial_state["session_id"]
            }