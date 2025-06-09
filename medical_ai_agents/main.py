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
        """Enhanced analyze method with improved session tracking and conversation history."""
        import os
        import time
        import json
        import uuid
        
        # Log basic information about the request
        self.logger.info(f"Starting analysis: image={image_path is not None}, text_query={query is not None}")
        
        # Check if we have a conversation history
        if conversation_history is None:
            conversation_history = []
        self.logger.info(f"Initial conversation history length: {len(conversation_history)}")

        # 1. IMPORTANT SESSION ID HANDLING
        # If we don't have a session_id parameter, try to extract it from conversation_history
        if not session_id and conversation_history:
            # Try to find session_id from conversation history
            for entry in conversation_history:
                if entry and isinstance(entry, dict) and "session_id" in entry:
                    session_id = entry["session_id"]
                    self.logger.info(f"Extracted session_id from conversation history: {session_id}")
                    break
        
        # If still no session_id, generate a new one
        if not session_id:
            session_id = str(uuid.uuid4())
            self.logger.info(f"Generated new session_id: {session_id}")
        else:
            self.logger.info(f"Using provided or extracted session_id: {session_id}")
        
        # Log the history we found
        if conversation_history:
            self.logger.info(f"Conversation history contains {len(conversation_history)} entries")
            if len(conversation_history) > 0:
                last_entry = conversation_history[-1]
                self.logger.info(f"Last entry: query='{last_entry.get('query', 'None')[:30]}...', response='{last_entry.get('response', 'None')[:30]}...'")
        else:
            # Initialize with system welcome message if this is a brand new conversation
            self.logger.info("Initializing new conversation with system welcome message")
            conversation_history = [{
                "query": "",  # Empty query because this isn't a user question
                "response": "Hello! I'm your Medical AI Assistant specializing in healthcare consultation. How can I assist you with your medical questions today?",
                "timestamp": time.time(),
                "is_system": True,  # Clearly mark as system message
                "type": "init",  # Add a type to identify this as initialization
                "session_id": session_id  # Add session_id to the initial system message
            }]
            self.logger.info(f"System welcome message added to history - New history length: {len(conversation_history)}")
        
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
        import uuid
        import time
        
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
            
        # Debug information about conversation history
        self.logger.info(f"Conversation history received: {len(history)} entries")
        if history and len(history) > 0:
            # Log the content of each entry in history for debugging
            for i, entry in enumerate(history):
                self.logger.info(f"History entry {i}: query='{entry.get('query', 'None')[:30]}...', "
                                f"is_system={entry.get('is_system', False)}, "
                                f"is_pending={entry.get('is_pending', False)}")
            
            self.logger.info(f"Last conversation entry: {history[-1].get('query', 'No query')[:30]}... - {history[-1].get('timestamp', 'No timestamp')}")
        else:
            # Initialize with system welcome message if this is a brand new conversation
            self.logger.info("Initializing new conversation with system welcome message")
            history = [{
                "query": "",  # Empty query because this isn't a user question
                "response": "Hello! I'm your Medical AI Assistant specializing in healthcare consultation. How can I assist you with your medical questions today?",
                "timestamp": time.time(),
                "is_system": True,  # Clearly mark as system message
                "type": "init",  # Add a type to identify this as initialization
                "session_id": session_id  # Add session_id to the initial system message
            }]
            self.logger.info(f"System welcome message added to history - New history length: {len(history)}")
            
        # For debugging purposes, temporarily append the current query but without a response
        # This helps us track the query flow through the system
        if query:  # Only add if there's an actual query
            # Clean up any previous pending entries to avoid duplicates
            pending_entries = [entry for entry in history if entry.get("is_pending", False)]
            if pending_entries:
                self.logger.info(f"Found {len(pending_entries)} pending entries, removing them")
                history = [entry for entry in history if not entry.get("is_pending", False)]
                self.logger.info(f"After removing pending entries, history length: {len(history)}")
            
            # Add the pending entry for current query
            debug_entry = {
                "query": query,
                "response": "[PENDING RESPONSE]",
                "timestamp": time.time(),
                "is_pending": True,  # Marked as pending so we know it's temporary
                "session_id": session_id  # Make sure to include session_id here
            }
            history.append(debug_entry)
            self.logger.info(f"Added current query to history for debugging: {query[:30]}...")
            self.logger.info(f"History now has {len(history)} entries with pending entry")
            
            # IMPORTANT: Also track the raw query in the state for better tracking
            # This ensures we have the query in multiple places for redundancy
            initial_state_query = query  # Store query for initial state
        else:
            initial_state_query = ""  # Empty string if no query
            self.logger.info("No query provided, not adding pending entry to history")
        
        initial_state: SystemState = {
            "image_path": image_path or "",
            "query": initial_state_query,  # Use the stored query variable
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