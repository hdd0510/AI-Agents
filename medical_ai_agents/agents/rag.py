#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LIGHT RAG AGENT - Simple Document Retrieval
=========================================
RAG agent đơn giản để xử lý PDF và DOC files upload sử dụng LightRAG.
"""

import json
import logging
import os
import sys
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, AsyncIterator

from medical_ai_agents.agents.base_agent import BaseAgent, ThoughtType, ReActStep
from medical_ai_agents.tools.rag.lightrag_tool import LightRagTool

# Kiểm tra và thêm đường dẫn LightRAG vào sys.path
LIGHTRAG_PATH = Path("/mnt/dunghd/medical-ai-agents/LightRAG")
if LIGHTRAG_PATH.exists() and str(LIGHTRAG_PATH) not in sys.path:
    sys.path.append(str(LIGHTRAG_PATH))

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import setup_logger
    from lightrag.kg.shared_storage import initialize_pipeline_status
except ImportError:
    logging.error("LightRAG không được cài đặt. Vui lòng cài đặt LightRAG hoặc kiểm tra đường dẫn.")

# Khởi tạo logger cho lightrag
setup_logger("lightrag", level="INFO")

# Thiết lập logger chi tiết cho RAG Agent
logger = logging.getLogger("medical-ai.rag-agent")
handler = logging.FileHandler("rag_process_flow.log")
handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class RAGAgent(BaseAgent):
    """
    Light RAG Agent for Document Q&A using LightRAG
    
    WORKFLOW:
    1. Load and parse uploaded PDFs/DOCs
    2. Create vector embeddings and knowledge graph
    3. Search relevant chunks based on query using hybrid search
    4. Synthesize answer from retrieved chunks
    """
    
    def __init__(self, 
                 storage_path: str = "./rag_storage",
                 llm_model: str = "gpt-4o-mini", 
                 device: str = "cuda"):
        """Initialize Light RAG Agent with LightRAG."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        logger.info(f"RAGAgent initializing with storage path: {storage_path}")
        logger.info(f"LLM model: {llm_model}, Device: {device}")
        
        super().__init__(name="Light RAG Agent", llm_model=llm_model, device=device)
        
        # Configuration
        self.max_iterations = 3  # Parse -> Search -> Answer
        
        # LightRAG instance
        self.lightrag = None
        self.embedding_func = None
        self.llm_func = None
        self.lightrag_tool = None
        
    def _register_tools(self) -> List[Any]:
        """Register LightRAG tool."""
        logger.info("Registering LightRAG tool")
        self.lightrag_tool = LightRagTool(storage_path=str(self.storage_path))
        
        # Khởi tạo ngay trong _register_tools để đảm bảo tool sẵn sàng
        try:
            # Chạy initialize để đảm bảo tool được khởi tạo đầy đủ
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Nếu loop đang chạy, tạo task và đợi hoàn thành
                init_task = asyncio.create_task(self.lightrag_tool.initialize())
                loop.run_until_complete(init_task)
            else:
                # Nếu không, chạy trong loop mới
                loop.run_until_complete(self.lightrag_tool.initialize())
            logger.info("LightRAG tool initialized successfully during registration")
        except Exception as e:
            logger.error(f"Error initializing LightRAG tool during registration: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        return [self.lightrag_tool]
    
    def _get_agent_description(self) -> str:
        """Agent description."""
        return """I am a document retrieval specialist that can:
        
1. Read and parse PDF/DOC files using LightRAG
2. Create searchable vector indexes and knowledge graphs
3. Find relevant information based on queries
4. Synthesize comprehensive answers from documents

I use LightRAG (Simple and Fast Retrieval-Augmented Generation) for medical document Q&A."""

    def _get_system_prompt(self) -> str:
        """System prompt for RAG agent."""
        return f"""You are a document retrieval and Q&A specialist using LightRAG (Retrieval-Augmented Generation).

AVAILABLE TOOLS:
{self.tool_descriptions}

CAPABILITIES:
- Process various document types (PDF, DOC, DOCX, TXT)
- Create automatic knowledge graphs and vector embeddings
- Execute hybrid search combining vector and knowledge graph results
- Answer questions based on document content

WORKFLOW:
1. When new documents are uploaded, use the lightrag tool with add_document action
2. For user queries, use lightrag tool with query action (default hybrid mode)
3. To explore knowledge graph, use lightrag tool with get_graph action
4. Always check document status with the lightrag tool's status action

RULES:
- Always cite which document and page the information comes from
- If no relevant information found, clearly state that
- Prioritize accuracy over completeness
- Use medical terminology appropriately when dealing with medical documents

For complex medical questions, you should:
1. Use hybrid search mode to combine vector similarity and knowledge graph
2. Provide detailed answers with proper citations
3. Note any uncertainty or limitations in the documents

Follow the ReAct format:
Thought: [your reasoning]
Action: [tool name or Final Answer]
Action Input: {{"param": "value"}}"""

    def initialize(self) -> bool:
        """Initialize RAG agent with LightRAG."""
        try:
            logger.info("Initializing RAG agent")
            
            # Đảm bảo tool đã được khởi tạo
            if not hasattr(self, "lightrag_tool") or self.lightrag_tool is None:
                logger.info("LightRAG tool not registered yet, registering now")
                self._register_tools()
            
            # Kiểm tra trạng thái của tool
            try:
                tool_status = self.lightrag_tool._check_status()
                
                if tool_status.get("success", False):
                    self.initialized = True
                    logger.info(f"Light RAG Agent initialized successfully with LightRAG. Status: {tool_status}")
                    return True
                else:
                    # Nếu chưa khởi tạo thành công, thử khởi tạo lại
                    logger.warning("LightRAG tool status check failed, attempting to initialize again")
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(self.lightrag_tool.initialize())
                    
                    # Kiểm tra lại sau khi khởi tạo
                    tool_status = self.lightrag_tool._check_status()
                    if tool_status.get("success", False):
                        self.initialized = True
                        logger.info(f"Light RAG Agent initialized successfully after retry. Status: {tool_status}")
                        return True
                    else:
                        logger.error(f"Failed to initialize LightRAG after retry: {tool_status.get('error')}")
                        return False
            except Exception as e:
                logger.error(f"Error checking LightRAG tool status: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG Agent: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract RAG task input."""
        # Check for uploaded documents
        uploaded_docs = state.get("uploaded_documents", [])
        logger.info(f"Extracted {len(uploaded_docs)} uploaded documents from state")
        
        # Get conversation history
        conversation_history = state.get("conversation_history", [])
        
        # Add debug to check conversation_history
        if conversation_history:
            logger.info(f"RAG agent received conversation history with {len(conversation_history)} entries")
        else:
            logger.warning("RAG agent did not receive conversation_history")
        
        # Get context from other agents
        medical_context = self._build_rag_context(state)
        logger.info(f"Built medical context from other agents: {json.dumps(medical_context)}")
        
        task_input = {
            "query": state.get("query", ""),
            "uploaded_documents": uploaded_docs,
            "medical_context": medical_context,
            "conversation_history": conversation_history,
            "session_id": state.get("session_id", ""),
            "require_sources": True,
        }
        
        logger.info(f"Extracted task input - Query: '{task_input['query']}'")
        return task_input

    def _build_rag_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build context from other agents for better retrieval."""
        context = {}
        
        # Add detection context
        if "detector_result" in state:
            detector = state["detector_result"]
            if detector.get("success", False):
                context["polyp_findings"] = {
                    "count": detector.get("count", 0),
                    "detected": detector.get("count", 0) > 0
                }
                logger.debug(f"Added polyp findings to context: {context['polyp_findings']}")
        
        # Add classification context
        if "modality_result" in state:
            modality = state["modality_result"]
            if modality.get("success", False):
                context["imaging_type"] = modality.get("class_name", "Unknown")
                logger.debug(f"Added imaging type to context: {context['imaging_type']}")
        
        if "region_result" in state:
            region = state["region_result"]
            if region.get("success", False):
                context["anatomical_region"] = region.get("class_name", "Unknown")
                logger.debug(f"Added anatomical region to context: {context['anatomical_region']}")
        
        logger.info(f"Built context from other agents: {context}")
        return context

    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for LightRAG processing."""
        logger.info("Formatting task input for LightRAG processing")
        
        query = task_input.get("query", "")
        uploaded_docs = task_input.get("uploaded_documents", [])
        medical_context = task_input.get("medical_context", {})
        conversation_history = task_input.get("conversation_history", [])
        
        # Build context string
        context_parts = []
        if medical_context:
            if "polyp_findings" in medical_context:
                findings = medical_context["polyp_findings"]
                context_parts.append(f"- Polyp detection: {findings['count']} polyp(s) found")
            if "imaging_type" in medical_context:
                context_parts.append(f"- Imaging type: {medical_context['imaging_type']}")
            if "anatomical_region" in medical_context:
                context_parts.append(f"- Anatomical region: {medical_context['anatomical_region']}")
        
        context_str = "\n".join(context_parts) if context_parts else "No additional context"
        
        # Format document info
        doc_info = ""
        if uploaded_docs:
            doc_info = f"\nUploaded documents ({len(uploaded_docs)} files):\n"
            for doc in uploaded_docs[:5]:  # Show first 5
                doc_info += f"- {os.path.basename(doc) if os.path.exists(doc) else doc}\n"
        
        # Format conversation history if available
        history_str = ""
        if conversation_history and len(conversation_history) > 0:
            # Get recent conversation entries (last 3)
            recent_history = conversation_history[-3:]
            history_str = "\nRecent conversation history:\n"
            for entry in recent_history:
                if "query" in entry and "response" in entry:
                    history_str += f"User: {entry['query']}\nAssistant: {entry['response'][:150]}...\n\n"
        
        # Build final prompt
        prompt = f"""User Query: {query}

Medical Context:
{context_str}
{doc_info}
{history_str}

TASK:
1. First, check if there are new documents to process using the lightrag tool
2. If there are new documents, add them to the LightRAG system
3. Execute the user's query using LightRAG's hybrid search for the best results
4. Provide a detailed answer based on the documents

If the user query is directly related to document content, use the lightrag tool to search for relevant information.
If the query is a follow-up to a previous question, consider the conversation history in your answer."""

        logger.debug(f"Formatted input prompt: {prompt[:200]}...")
        return prompt

    def _format_agent_result(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format agent result from ReAct loop."""
        # Default format if result is not successful
        if not react_result.get('success', False):
            logger.warning(f"RAG agent ReAct execution failed: {react_result.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": react_result.get('error', 'Unknown error'),
                "history": react_result.get('history', [])
            }
        
        # Format successful result
        answer = react_result.get('answer', '')
        logger.info("RAG agent execution successful, formatting result")
        logger.debug(f"RAG answer: {answer[:100]}...")
        
        # Extract sources if available
        sources = []
        if 'history' in react_result:
            for step in react_result['history']:
                if step.get('type') == 'action' and step.get('action') == 'lightrag' and step.get('action_input', {}).get('action') == 'query':
                    if 'result' in step and isinstance(step['result'], dict) and 'response' in step['result']:
                        response = step['result']['response']
                        if isinstance(response, dict) and 'sources' in response:
                            sources = response['sources']
                            break
                            
        result = {
            "success": True,
            "response": answer,
            "history": react_result.get('history', []),
            "sources": sources,
            "query_complexity": self._determine_query_complexity(react_result),
            "chunks_retrieved": len(sources),
            "documents_processed": [s.get('file', 'Unknown') for s in sources if 'file' in s]
        }
        
        logger.info(f"Formatted result with {len(sources)} sources and complexity: {result['query_complexity']}")
        return result

    def _determine_query_complexity(self, react_result: Dict[str, Any]) -> str:
        """Determine query complexity based on ReAct execution."""
        history = react_result.get('history', [])
        
        # Count iterations and tool calls
        iterations = len([step for step in history if step.get('type') == 'thought'])
        tool_calls = len([step for step in history if step.get('type') == 'action' and step.get('action') != 'Final Answer'])
        
        if iterations >= 5 or tool_calls >= 3:
            return "complex"
        elif iterations >= 3 or tool_calls >= 2:
            return "moderate"
        else:
            return "simple"

    async def process_async(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async version of process method - simplified without streaming.
        """
        logger.info("Starting async RAG processing")
        if not self.initialized:
            logger.error("RAGAgent not initialized")
            return {"success": False, "error": "RAGAgent not initialized"}
            
        try:
            # Parse input
            task_input = self._extract_task_input(state)
            query = task_input.get("query", "")
            uploaded_docs = task_input.get("uploaded_documents", [])
            
            logger.info(f"Processing query: '{query[:50]}...' with {len(uploaded_docs)} documents")
            
            # Process with the main process method which uses ReAct
            start_time = time.time()
            result = self.process(state)
            processing_time = time.time() - start_time
            logger.info(f"RAG processing completed in {processing_time:.2f} seconds")
                
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG async processing: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            error_response = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
            return {**state, "rag_result": error_response}

    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Add documents to LightRAG using the lightrag tool.
        """
        # Đảm bảo agent đã khởi tạo
        if not self.initialized:
            logger.warning("RAG Agent not initialized, initializing now")
            if not self.initialize():
                return {
                    "success": False,
                    "error": "Failed to initialize RAG Agent",
                    "processed_files": [],
                    "errors": []
                }
        
        logger.info(f"Adding {len(file_paths)} documents to LightRAG")
        results = {"success": True, "processed_files": [], "errors": []}
        
        for file_path in file_paths:
            logger.info(f"Processing document: {os.path.basename(file_path)}")
            try:
                result = self.lightrag_tool._run("add_document", file_path=file_path)
                if result.get("success", False):
                    logger.info(f"Successfully added document: {os.path.basename(file_path)}")
                    results["processed_files"].append(file_path)
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Failed to add document {os.path.basename(file_path)}: {error_msg}")
                    results["errors"].append({
                        "file": os.path.basename(file_path),
                        "error": error_msg
                    })
            except Exception as e:
                logger.error(f"Exception adding document {os.path.basename(file_path)}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                results["errors"].append({
                    "file": os.path.basename(file_path),
                    "error": str(e)
                })
        
        # Update overall success
        if not results["processed_files"] and results["errors"]:
            results["success"] = False
            
        logger.info(f"Document addition complete - Success: {results['success']}, "
                  f"Processed: {len(results['processed_files'])}, "
                  f"Errors: {len(results['errors'])}")
            
        return results

    def has_documents(self) -> bool:
        """
        Check if any documents are indexed using LightRAG tool.
        """
        try:
            # Đảm bảo agent đã được khởi tạo
            if not self.initialized:
                logger.warning("RAG Agent not initialized, initializing now")
                self.initialize()
            
            # Đảm bảo LightRAG tool đã được khởi tạo
            if not hasattr(self, "lightrag_tool") or self.lightrag_tool is None:
                logger.error("LightRAG tool not available")
                return False
                
            status = self.lightrag_tool._check_status()
            doc_count = status.get("total_documents", 0)
            logger.info(f"Document check: {doc_count} documents indexed")
            return doc_count > 0
            
        except Exception as e:
            logger.error(f"Error checking document status: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False