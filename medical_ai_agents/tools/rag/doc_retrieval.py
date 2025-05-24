#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Tools - Document Retrieval Tool
----------------------------------------
Tool truy xuất toàn bộ nội dung document từ knowledge base.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

from medical_ai_agents.tools.base_tools import BaseTool

class DocumentRetrievalTool(BaseTool):
    """Tool truy xuất toàn bộ nội dung document từ knowledge base."""
    
    def __init__(self, knowledge_base_path: str, **kwargs):
        """Initialize document retrieval tool."""
        super().__init__(
            name="document_retrieval",
            description="Truy xuất toàn bộ nội dung của một document cụ thể từ knowledge base."
        )
        
        self.knowledge_base_path = knowledge_base_path
        self.document_cache = {}  # Cache để tránh đọc lại file
    
    def _run(self, document_path: Optional[str] = None, document_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve full document content."""
        try:
            # Determine which document to retrieve
            target_path = None
            
            if document_path:
                # Direct path provided
                if os.path.isabs(document_path):
                    target_path = document_path
                else:
                    # Relative to knowledge base
                    target_path = os.path.join(self.knowledge_base_path, document_path)
            elif document_id:
                # Search by document ID/name
                target_path = self._find_document_by_id(document_id)
            else:
                return {"success": False, "error": "Either document_path or document_id must be provided"}
            
            if not target_path or not os.path.exists(target_path):
                return {"success": False, "error": f"Document not found: {target_path}"}
            
            # Check cache first
            if target_path in self.document_cache:
                self.logger.info(f"Retrieved document from cache: {target_path}")
                return self.document_cache[target_path]
            
            # Read and process document
            result = self._read_document(target_path)
            
            # Cache the result
            if result.get("success", False):
                self.document_cache[target_path] = result
            
            return result
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _find_document_by_id(self, document_id: str) -> Optional[str]:
        """Find document path by ID/name."""
        try:
            knowledge_path = Path(self.knowledge_base_path)
            if not knowledge_path.exists():
                return None
            
            # Search patterns
            search_patterns = [
                f"{document_id}",
                f"{document_id}.*",
                f"*{document_id}*",
            ]
            
            for pattern in search_patterns:
                matches = list(knowledge_path.rglob(pattern))
                if matches:
                    # Return first match
                    return str(matches[0])
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding document by ID {document_id}: {str(e)}")
            return None
    
    def _read_document(self, file_path: str) -> Dict[str, Any]:
        """Read and process document content."""
        try:
            file_path_obj = Path(file_path)
            file_extension = file_path_obj.suffix.lower()
            
            # Process based on file type
            if file_extension == '.pdf':
                return self._read_pdf(file_path)
            elif file_extension == '.json':
                return self._read_json(file_path)
            elif file_extension in ['.txt', '.md']:
                return self._read_text(file_path)
            else:
                # Try to read as text
                return self._read_text(file_path)
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read document: {str(e)}"
            }
    
    def _read_text(self, file_path: str) -> Dict[str, Any]:
        """Read text/markdown files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_path_obj = Path(file_path)
            
            # Extract metadata
            metadata = {
                "file_name": file_path_obj.name,
                "file_size": os.path.getsize(file_path),
                "file_type": file_path_obj.suffix,
                "last_modified": os.path.getmtime(file_path)
            }
            
            # Basic content analysis
            lines = content.split('\n')
            word_count = len(content.split())
            
            return {
                "success": True,
                "content": content,
                "metadata": metadata,
                "statistics": {
                    "character_count": len(content),
                    "word_count": word_count,
                    "line_count": len(lines),
                    "paragraph_count": len([line for line in lines if line.strip()])
                },
                "source": file_path,
                "title": file_path_obj.stem
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read text file: {str(e)}"
            }
    
    def _read_json(self, file_path: str) -> Dict[str, Any]:
        """Read JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            file_path_obj = Path(file_path)
            
            # Extract text content for main content field
            main_content = self._extract_json_content(data)
            
            # Metadata
            metadata = {
                "file_name": file_path_obj.name,
                "file_size": os.path.getsize(file_path),
                "file_type": ".json",
                "last_modified": os.path.getmtime(file_path),
                "json_structure": self._analyze_json_structure(data)
            }
            
            return {
                "success": True,
                "content": main_content,
                "raw_data": data,
                "metadata": metadata,
                "source": file_path,
                "title": file_path_obj.stem
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read JSON file: {str(e)}"
            }
    
    def _read_pdf(self, file_path: str) -> Dict[str, Any]:
        """Read PDF files."""
        try:
            content = ""
            
            # Try pdfplumber first
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            content += f"\n--- Page {page_num + 1} ---\n{text}\n"
                            
            except ImportError:
                # Fallback to PyPDF2
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page_num, page in enumerate(pdf_reader.pages):
                            text = page.extract_text()
                            if text:
                                content += f"\n--- Page {page_num + 1} ---\n{text}\n"
                except ImportError:
                    return {
                        "success": False,
                        "error": "Neither pdfplumber nor PyPDF2 installed. Cannot read PDF files."
                    }
            
            if not content.strip():
                return {
                    "success": False,
                    "error": "Could not extract text from PDF file"
                }
            
            file_path_obj = Path(file_path)
            
            # Metadata
            metadata = {
                "file_name": file_path_obj.name,
                "file_size": os.path.getsize(file_path),
                "file_type": ".pdf",
                "last_modified": os.path.getmtime(file_path)
            }
            
            # Statistics
            word_count = len(content.split())
            lines = content.split('\n')
            
            return {
                "success": True,
                "content": content.strip(),
                "metadata": metadata,
                "statistics": {
                    "character_count": len(content),
                    "word_count": word_count,
                    "line_count": len(lines),
                    "estimated_pages": content.count("--- Page")
                },
                "source": file_path,
                "title": file_path_obj.stem
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read PDF file: {str(e)}"
            }
    
    def _extract_json_content(self, data) -> str:
        """Extract readable content from JSON data."""
        content_parts = []
        
        if isinstance(data, dict):
            # Common content fields
            content_fields = ['content', 'text', 'description', 'summary', 'body', 'article', 'question', 'answer']
            
            for field in content_fields:
                if field in data and isinstance(data[field], str):
                    content_parts.append(f"{field.title()}: {data[field]}")
            
            # Extract title if available
            title_fields = ['title', 'name', 'subject', 'heading']
            for field in title_fields:
                if field in data and isinstance(data[field], str):
                    content_parts.insert(0, f"Title: {data[field]}")
                    break
            
            # If no specific fields found, extract all string values
            if not content_parts:
                for key, value in data.items():
                    if isinstance(value, str) and len(value) > 10:
                        content_parts.append(f"{key.replace('_', ' ').title()}: {value}")
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    item_content = self._extract_json_content(item)
                    if item_content:
                        content_parts.append(f"Item {i+1}:\n{item_content}")
                elif isinstance(item, str):
                    content_parts.append(item)
        
        return "\n\n".join(content_parts)
    
    def _analyze_json_structure(self, data) -> Dict[str, Any]:
        """Analyze JSON structure for metadata."""
        if isinstance(data, dict):
            return {
                "type": "object",
                "keys": list(data.keys()),
                "key_count": len(data)
            }
        elif isinstance(data, list):
            return {
                "type": "array",
                "length": len(data),
                "item_types": list(set(type(item).__name__ for item in data))
            }
        else:
            return {
                "type": type(data).__name__
            }
    
    def list_available_documents(self) -> Dict[str, Any]:
        """List all available documents in knowledge base."""
        try:
            documents = []
            knowledge_path = Path(self.knowledge_base_path)
            
            if not knowledge_path.exists():
                return {"success": False, "error": f"Knowledge base path does not exist: {self.knowledge_base_path}"}
            
            supported_extensions = {'.txt', '.md', '.json', '.pdf'}
            
            for file_path in knowledge_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    try:
                        stat = file_path.stat()
                        documents.append({
                            "name": file_path.name,
                            "path": str(file_path),
                            "relative_path": str(file_path.relative_to(knowledge_path)),
                            "size": stat.st_size,
                            "type": file_path.suffix,
                            "last_modified": stat.st_mtime
                        })
                    except Exception as e:
                        self.logger.warning(f"Error getting info for {file_path}: {str(e)}")
            
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
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear document cache."""
        cache_size = len(self.document_cache)
        self.document_cache.clear()
        return {
            "success": True,
            "message": f"Cleared {cache_size} cached documents"
        }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for the tool parameters."""
        return {
            "document_path": {
                "type": "string",
                "description": "Path to the document file (absolute or relative to knowledge base)",
                "required": False
            },
            "document_id": {
                "type": "string", 
                "description": "Document ID/name to search for",
                "required": False
            }
        }