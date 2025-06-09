#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Tools - Chunk Retriever Tool
------------------------------------
Tool để retrieve và quản lý chunks đã được index.
"""

import os
import json
import pickle
from typing import Dict, Any, List, Optional
from pathlib import Path

from medical_ai_agents.tools.base_tools import BaseTool

class ChunkRetrieverTool(BaseTool):
    """Tool retrieve chunks và metadata."""
    
    def __init__(self, storage_path: str, **kwargs):
        """Initialize chunk retriever tool."""
        super().__init__(
            name="chunk_retriever",
            description="Retrieve chunk information and check document index status."
        )
        
        self.storage_path = Path(storage_path)
        self.chunks_path = self.storage_path / "chunks"
        self.metadata_path = self.storage_path / "metadata.json"
    
    def _run(self, action: str = "status", document_name: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve chunk information.
        
        Actions:
        - status: Get index status
        - list_documents: List all indexed documents
        - get_document_chunks: Get chunks for specific document
        """
        try:
            if action == "status":
                return self._get_index_status()
            
            elif action == "list_documents":
                return self._list_documents()
            
            elif action == "get_document_chunks":
                if not document_name:
                    return {
                        "success": False,
                        "error": "document_name required for get_document_chunks action"
                    }
                return self._get_document_chunks(document_name)
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}. Use 'status', 'list_documents', or 'get_document_chunks'"
                }
                
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _get_index_status(self) -> Dict[str, Any]:
        """Get current index status."""
        try:
            # Load metadata
            metadata = {}
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Count total chunks
            total_chunks = 0
            total_documents = len(metadata)
            
            for file_hash, file_info in metadata.items():
                total_chunks += file_info.get("chunks_count", 0)
            
            # Check if vector index exists
            vector_index_exists = (self.storage_path / "vector_index.pkl").exists()
            
            # Get storage size
            storage_size = 0
            if self.chunks_path.exists():
                for file_path in self.chunks_path.glob("*"):
                    storage_size += file_path.stat().st_size
            
            return {
                "success": True,
                "status": "ready" if vector_index_exists else "not_indexed",
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "vector_index_exists": vector_index_exists,
                "storage_size_mb": round(storage_size / (1024 * 1024), 2),
                "storage_path": str(self.storage_path)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get index status: {str(e)}"
            }
    
    def _list_documents(self) -> Dict[str, Any]:
        """List all indexed documents."""
        try:
            # Load metadata
            if not self.metadata_path.exists():
                return {
                    "success": True,
                    "documents": [],
                    "message": "No documents indexed yet"
                }
            
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Create document list
            documents = []
            for file_hash, file_info in metadata.items():
                documents.append({
                    "name": file_info.get("file_name", "Unknown"),
                    "type": file_info.get("file_type", "Unknown"),
                    "chunks": file_info.get("chunks_count", 0),
                    "hash": file_hash,
                    "processed_at": file_info.get("processed_at", "Unknown")
                })
            
            # Sort by name
            documents.sort(key=lambda x: x["name"])
            
            return {
                "success": True,
                "documents": documents,
                "total_count": len(documents)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to list documents: {str(e)}"
            }
    
    def _get_document_chunks(self, document_name: str) -> Dict[str, Any]:
        """Get chunks for specific document."""
        try:
            # Load metadata to find document
            if not self.metadata_path.exists():
                return {
                    "success": False,
                    "error": "No documents indexed"
                }
            
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Find document by name
            target_hash = None
            for file_hash, file_info in metadata.items():
                if file_info.get("file_name", "") == document_name:
                    target_hash = file_hash
                    break
            
            if not target_hash:
                return {
                    "success": False,
                    "error": f"Document not found: {document_name}"
                }
            
            # Load chunks
            chunk_file = self.chunks_path / f"{target_hash}.pkl"
            if not chunk_file.exists():
                return {
                    "success": False,
                    "error": f"Chunks file not found for document: {document_name}"
                }
            
            with open(chunk_file, 'rb') as f:
                chunks = pickle.load(f)
            
            # Format chunks for display
            formatted_chunks = []
            for i, chunk in enumerate(chunks[:10]):  # Limit to first 10
                formatted_chunks.append({
                    "chunk_id": chunk.get("chunk_id", f"chunk_{i}"),
                    "page": chunk.get("page", 0),
                    "content_preview": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    "length": len(chunk["content"])
                })
            
            return {
                "success": True,
                "document_name": document_name,
                "total_chunks": len(chunks),
                "chunks_preview": formatted_chunks,
                "message": f"Showing first {len(formatted_chunks)} chunks"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to get document chunks: {str(e)}"
            }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for the tool parameters."""
        return {
            "action": {
                "type": "string",
                "description": "Action to perform: status, list_documents, or get_document_chunks",
                "enum": ["status", "list_documents", "get_document_chunks"],
                "default": "status"
            },
            "document_name": {
                "type": "string",
                "description": "Document name (required for get_document_chunks action)",
                "required": False
            }
        }