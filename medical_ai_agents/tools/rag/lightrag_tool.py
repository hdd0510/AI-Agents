#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Tools - LightRAG Tool
-----------------------------
Tool wrapper cho LightRAG để quản lý tài liệu và thực hiện truy vấn.
"""

import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Thêm đường dẫn LightRAG vào sys.path
LIGHTRAG_PATH = Path("/mnt/dunghd/medical-ai-agents/LightRAG")
if LIGHTRAG_PATH.exists() and str(LIGHTRAG_PATH) not in sys.path:
    sys.path.append(str(LIGHTRAG_PATH))

from medical_ai_agents.tools.base_tools import BaseTool

# Import LightRAG
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import setup_logger
    from lightrag.kg.shared_storage import initialize_pipeline_status
    from lightrag.llm.openai import openai_embed
except ImportError:
    logging.error("LightRAG không được cài đặt. Vui lòng cài đặt LightRAG hoặc kiểm tra đường dẫn.")

# Khởi tạo logger cho lightrag
setup_logger("lightrag", level="INFO")

# Setup detailed logger for the LightRAG tool
logger = logging.getLogger("medical-ai.rag-tool")
handler = logging.FileHandler("lightrag_tool.log")
handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class LightRagTool(BaseTool):
    """Tool sử dụng LightRAG để quản lý và truy vấn tài liệu."""
    
    def __init__(self, 
                storage_path: str, 
                embedding_func=None,
                llm_model_func=None,
                **kwargs):
        """Initialize LightRAG tool."""
        super().__init__(
            name="lightrag",
            description="Công cụ quản lý tài liệu và truy vấn sử dụng LightRAG."
        )
        
        logger.info(f"Initializing LightRAG tool with storage path: {storage_path}")
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.embedding_func = embedding_func or openai_embed
        self.llm_model_func = llm_model_func
        self.lightrag = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize LightRAG instance."""
        logger.info("Initializing LightRAG instance")
        if self.initialized:
            logger.info("LightRAG already initialized")
            return True
        
        try:
            # Initialize LightRAG
            logger.info("Creating LightRAG instance")
            self.lightrag = LightRAG(
                working_dir=str(self.storage_path),
                embedding_func=self.embedding_func,
                llm_model_func=self.llm_model_func,
            )
            
            # Initialize storages
            logger.info("Initializing LightRAG storages")
            await self.lightrag.initialize_storages()
            await initialize_pipeline_status()
            
            self.initialized = True
            logger.info("LightRAG initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LightRAG: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _ensure_initialized(self):
        """Đảm bảo LightRAG đã được khởi tạo trước khi sử dụng."""
        if not self.initialized:
            logger.info("LightRAG not initialized, initializing now")
            loop = asyncio.get_event_loop()
            try:
                init_success = loop.run_until_complete(self.initialize())
                if not init_success:
                    logger.error("Failed to initialize LightRAG")
                    raise RuntimeError("Failed to initialize LightRAG")
            except Exception as e:
                logger.error(f"Error during LightRAG initialization: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
    
    def _run(self, action: str = "status", **kwargs) -> Dict[str, Any]:
        """
        Run action on LightRAG.
        
        Actions:
        - status: Check LightRAG status
        - query: Execute query (requires 'query_text')
        - add_document: Add document to LightRAG (requires 'file_path')
        - get_graph: Get knowledge graph (requires 'node_label')
        """
        try:
            logger.info(f"Running LightRAG action: {action} with params: {kwargs}")
            
            # Đảm bảo LightRAG đã được khởi tạo
            self._ensure_initialized()
            
            if action == "status":
                result = self._check_status()
                logger.info(f"Status check result: success={result.get('success')}, total_documents={result.get('total_documents', 0)}")
                return result
            
            elif action == "query":
                query_text = kwargs.get("query_text")
                search_mode = kwargs.get("search_mode", "hybrid")
                if not query_text:
                    logger.error("Missing query_text for query action")
                    return {"success": False, "error": "query_text required for query action"}
                
                logger.info(f"Executing query: '{query_text[:50]}...' with mode: {search_mode}")
                return self._execute_query(query_text, search_mode)
            
            elif action == "add_document":
                file_path = kwargs.get("file_path")
                if not file_path:
                    logger.error("Missing file_path for add_document action")
                    return {"success": False, "error": "file_path required for add_document action"}
                
                logger.info(f"Adding document: {os.path.basename(file_path)}")
                return self._add_document(file_path)
            
            elif action == "get_graph":
                node_label = kwargs.get("node_label")
                max_depth = kwargs.get("max_depth", 3)
                max_nodes = kwargs.get("max_nodes", 1000)
                if not node_label:
                    logger.error("Missing node_label for get_graph action")
                    return {"success": False, "error": "node_label required for get_graph action"}
                
                logger.info(f"Getting knowledge graph for node: {node_label}, depth: {max_depth}, max nodes: {max_nodes}")
                return self._get_knowledge_graph(node_label, max_depth, max_nodes)
            
            else:
                logger.error(f"Unknown action: {action}")
                return {
                    "success": False,
                    "error": f"Unknown action: {action}. Use 'status', 'query', 'add_document', or 'get_graph'"
                }
                
        except Exception as e:
            import traceback
            logger.error(f"Error in LightRAG tool: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _check_status(self) -> Dict[str, Any]:
        """Check LightRAG status."""
        try:
            # Đảm bảo LightRAG đã được khởi tạo
            self._ensure_initialized()
            
            if not self.lightrag:
                logger.error("LightRAG not initialized during status check")
                return {"success": False, "error": "LightRAG not initialized"}
                
            logger.info("Checking LightRAG status")
            loop = asyncio.get_event_loop()
            processing_status = loop.run_until_complete(self.lightrag.get_processing_status())
            
            graph_labels = loop.run_until_complete(self.lightrag.get_graph_labels())
            
            status = {
                "success": True,
                "status": "initialized" if self.initialized else "not_initialized",
                "document_counts": processing_status,
                "total_documents": sum(processing_status.values()) if processing_status else 0,
                "graph_labels": graph_labels or [],
                "storage_path": str(self.storage_path)
            }
            
            logger.info(f"Status check complete - Documents: {status['total_documents']}, "
                      f"Graph labels: {len(status['graph_labels'])}")
            return status
            
        except Exception as e:
            logger.error(f"Failed to check status: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to check status: {str(e)}"
            }
    
    def _execute_query(self, query_text: str, search_mode: str = "hybrid") -> Dict[str, Any]:
        """Execute query with LightRAG."""
        try:
            start_time = time.time()
            if not self.lightrag:
                logger.error("LightRAG not initialized during query execution")
                return {"success": False, "error": "LightRAG not initialized"}
                
            # Validate search_mode
            valid_modes = ["naive", "local", "global", "hybrid"]
            if search_mode not in valid_modes:
                logger.warning(f"Invalid search mode: {search_mode}, fallback to hybrid")
                search_mode = "hybrid"
                
            # Execute query
            logger.info(f"Executing {search_mode} query: '{query_text[:100]}...'")
            loop = asyncio.get_event_loop()
            response = loop.run_until_complete(
                self.lightrag.query(query_text, param=QueryParam(mode=search_mode))
            )
            
            # Log timing and result
            execution_time = time.time() - start_time
            source_count = len(response.get("sources", []))
            logger.info(f"Query execution completed in {execution_time:.2f}s with {source_count} sources")
            
            # Log sources found
            if source_count > 0:
                for i, source in enumerate(response.get("sources", [])[:3]):  # Log first 3 sources
                    file = source.get("file", "Unknown")
                    page = source.get("page", "?")
                    logger.info(f"Source {i+1}: {file}, page {page}")
            
            return {
                "success": True,
                "query": query_text,
                "mode": search_mode,
                "response": response,
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Query failed: {str(e)}"
            }
    
    def _add_document(self, file_path: str) -> Dict[str, Any]:
        """Add document to LightRAG."""
        try:
            start_time = time.time()
            if not self.lightrag:
                logger.error("LightRAG not initialized during document addition")
                return {"success": False, "error": "LightRAG not initialized"}
                
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return {"success": False, "error": f"File not found: {file_path}"}
                
            # Get file content
            file_ext = Path(file_path).suffix.lower()
            logger.info(f"Processing document with extension: {file_ext}")
            
            if file_ext in ['.txt', '.pdf', '.doc', '.docx']:
                if file_ext == '.txt':
                    logger.info(f"Reading text file: {os.path.basename(file_path)}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    # For binary files, pass the file path
                    content = file_path
                    logger.info(f"Using file path for binary file: {os.path.basename(file_path)}")
                
                # Add document to LightRAG
                logger.info(f"Adding document to LightRAG: {os.path.basename(file_path)}")
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.lightrag.ainsert(
                    input=content,
                    file_paths=file_path
                ))
                
                processing_time = time.time() - start_time
                logger.info(f"Document added in {processing_time:.2f}s: {os.path.basename(file_path)}")
                
                # Check status after addition
                status = self._check_status()
                
                return {
                    "success": True,
                    "message": f"Document added: {os.path.basename(file_path)}",
                    "file_path": file_path,
                    "processing_time": processing_time,
                    "document_count": status.get("total_documents", 0)
                }
            else:
                logger.error(f"Unsupported file type: {file_ext}")
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_ext}"
                }
                
        except Exception as e:
            logger.error(f"Failed to add document: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Failed to add document: {str(e)}"
            }
    
    def _get_knowledge_graph(self, node_label: str, max_depth: int = 3, max_nodes: int = 1000) -> Dict[str, Any]:
        """Get knowledge graph for a given node label."""
        try:
            start_time = time.time()
            if not self.lightrag:
                logger.error("LightRAG not initialized during knowledge graph retrieval")
                return {"success": False, "error": "LightRAG not initialized"}
                
            # Get knowledge graph
            logger.info(f"Getting knowledge graph for node: {node_label}, depth: {max_depth}, max nodes: {max_nodes}")
            loop = asyncio.get_event_loop()
            graph = loop.run_until_complete(self.lightrag.get_knowledge_graph(
                node_label=node_label,
                max_depth=max_depth,
                max_nodes=max_nodes
            ))
            
            processing_time = time.time() - start_time
            
            # Format result
            node_count = len(graph.get("nodes", []))
            edge_count = len(graph.get("edges", []))
            logger.info(f"Knowledge graph retrieved in {processing_time:.2f}s - Nodes: {node_count}, Edges: {edge_count}")
            
            return {
                "success": True,
                "node_label": node_label,
                "nodes": graph.get("nodes", []),
                "edges": graph.get("edges", []),
                "node_count": node_count,
                "edge_count": edge_count,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledge graph: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                "success": False,
                "error": f"Failed to get knowledge graph: {str(e)}"
            }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for the tool parameters."""
        return {
            "action": {
                "type": "string",
                "description": "Action to perform: status, query, add_document, or get_graph",
                "enum": ["status", "query", "add_document", "get_graph"],
                "default": "status"
            },
            "query_text": {
                "type": "string", 
                "description": "Query text for search (required for query action)",
                "required": False
            },
            "search_mode": {
                "type": "string",
                "description": "Search mode: naive, local, global, or hybrid",
                "enum": ["naive", "local", "global", "hybrid"],
                "default": "hybrid",
                "required": False
            },
            "file_path": {
                "type": "string",
                "description": "Path to document file (required for add_document action)",
                "required": False
            },
            "node_label": {
                "type": "string",
                "description": "Node label for knowledge graph (required for get_graph action)",
                "required": False
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth for knowledge graph",
                "default": 3,
                "required": False
            },
            "max_nodes": {
                "type": "integer",
                "description": "Maximum number of nodes for knowledge graph",
                "default": 1000,
                "required": False
            }
        } 