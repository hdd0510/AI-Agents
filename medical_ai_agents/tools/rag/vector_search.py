#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Tools - Vector Search Tool
----------------------------------
Tool tìm kiếm semantic trong knowledge base sử dụng vector embeddings.
"""

import os
import json
import pickle
import numpy as np
import faiss
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from medical_ai_agents.tools.base_tools import BaseTool

class VectorSearchTool(BaseTool):
    """Tool tìm kiếm semantic trong knowledge base."""
    
    def __init__(self, 
                 knowledge_base_path: str,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cuda",
                 index_path: Optional[str] = None,
                 **kwargs):
        """Initialize vector search tool."""
        super().__init__(
            name="vector_search",
            description="Tìm kiếm semantic trong knowledge base sử dụng vector embeddings."
        )
        
        self.knowledge_base_path = knowledge_base_path
        self.embedding_model_name = embedding_model
        self.device = device
        self.index_path = index_path or os.path.join(knowledge_base_path, "vector_index")
        
        # Initialize components
        self.embedding_model = None
        self.faiss_index = None
        self.document_metadata = []
        self.index_initialized = False
        
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self) -> bool:
        """Initialize sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Move to device if CUDA available
            if self.device == "cuda" and hasattr(self.embedding_model, '_modules'):
                try:
                    self.embedding_model = self.embedding_model.to(self.device)
                except:
                    self.logger.warning("Could not move embedding model to CUDA, using CPU")
            
            return True
            
        except ImportError:
            self.logger.error("sentence-transformers not installed. Please run: pip install sentence-transformers")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            return False
    
    def initialize_index(self) -> bool:
        """Initialize or load vector index."""
        try:
            # Check if index exists
            index_file = os.path.join(self.index_path, "faiss_index.bin")
            metadata_file = os.path.join(self.index_path, "metadata.pkl")
            
            if os.path.exists(index_file) and os.path.exists(metadata_file):
                # Load existing index
                self.logger.info("Loading existing vector index...")
                self.faiss_index = faiss.read_index(index_file)
                
                with open(metadata_file, 'rb') as f:
                    self.document_metadata = pickle.load(f)
                
                self.logger.info(f"Loaded index with {self.faiss_index.ntotal} vectors")
            else:
                # Build new index
                self.logger.info("Building new vector index...")
                self._build_index()
            
            self.index_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize index: {str(e)}")
            return False
    
    def _build_index(self) -> bool:
        """Build vector index from documents."""
        try:
            # Collect all documents
            documents = self._collect_documents()
            if not documents:
                self.logger.warning("No documents found in knowledge base")
                return False
            
            # Create embeddings
            self.logger.info(f"Creating embeddings for {len(documents)} document segments...")
            texts = [doc['content'] for doc in documents]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Store metadata
            self.document_metadata = documents
            
            # Save index
            os.makedirs(self.index_path, exist_ok=True)
            faiss.write_index(self.faiss_index, os.path.join(self.index_path, "faiss_index.bin"))
            
            with open(os.path.join(self.index_path, "metadata.pkl"), 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            self.logger.info(f"Built and saved index with {len(documents)} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build index: {str(e)}")
            return False
    
    def _collect_documents(self) -> List[Dict[str, Any]]:
        """Collect and segment documents from knowledge base."""
        documents = []
        
        try:
            knowledge_path = Path(self.knowledge_base_path)
            if not knowledge_path.exists():
                self.logger.error(f"Knowledge base path does not exist: {self.knowledge_base_path}")
                return documents
            
            # Support different file types
            supported_extensions = {'.txt', '.md', '.json', '.pdf'}
            
            for file_path in knowledge_path.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                    try:
                        file_docs = self._process_file(file_path)
                        documents.extend(file_docs)
                    except Exception as e:
                        self.logger.warning(f"Failed to process {file_path}: {str(e)}")
            
            self.logger.info(f"Collected {len(documents)} document segments")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to collect documents: {str(e)}")
            return documents
    
    def _process_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single file and return document segments."""
        documents = []
        
        try:
            if file_path.suffix.lower() == '.pdf':
                # Handle PDF files
                documents = self._process_pdf(file_path)
            elif file_path.suffix.lower() == '.json':
                # Handle JSON files
                documents = self._process_json(file_path)
            else:
                # Handle text files (txt, md)
                documents = self._process_text(file_path)
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return []
    
    def _process_text(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process text/markdown files."""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into chunks (simple paragraph-based splitting)
            chunks = self._split_text_into_chunks(content)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:  # Filter out very short chunks
                    documents.append({
                        'content': chunk.strip(),
                        'source': str(file_path),
                        'chunk_id': i,
                        'file_type': file_path.suffix,
                        'title': file_path.stem
                    })
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error processing text file {file_path}: {str(e)}")
            return []
    
    def _process_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process JSON files."""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        # Extract text content from dict
                        content = self._extract_text_from_dict(item)
                        if content:
                            documents.append({
                                'content': content,
                                'source': str(file_path),
                                'chunk_id': i,
                                'file_type': '.json',
                                'title': file_path.stem,
                                'metadata': item
                            })
            elif isinstance(data, dict):
                content = self._extract_text_from_dict(data)
                if content:
                    documents.append({
                        'content': content,
                        'source': str(file_path),
                        'chunk_id': 0,
                        'file_type': '.json',
                        'title': file_path.stem,
                        'metadata': data
                    })
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error processing JSON file {file_path}: {str(e)}")
            return []
    
    def _process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process PDF files."""
        documents = []
        
        try:
            # Try to import PyPDF2 or pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text + "\n"
                            
            except ImportError:
                try:
                    import PyPDF2
                    with open(file_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        full_text = ""
                        for page in pdf_reader.pages:
                            full_text += page.extract_text() + "\n"
                except ImportError:
                    self.logger.warning("Neither pdfplumber nor PyPDF2 installed. Cannot process PDF files.")
                    return []
            
            # Split into chunks
            chunks = self._split_text_into_chunks(full_text)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) > 50:
                    documents.append({
                        'content': chunk.strip(),
                        'source': str(file_path),
                        'chunk_id': i,
                        'file_type': '.pdf',
                        'title': file_path.stem
                    })
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error processing PDF file {file_path}: {str(e)}")
            return []
    
    def _extract_text_from_dict(self, data: dict) -> str:
        """Extract text content from dictionary."""
        text_fields = ['content', 'text', 'description', 'summary', 'title', 'question', 'answer']
        
        extracted_texts = []
        for field in text_fields:
            if field in data and isinstance(data[field], str):
                extracted_texts.append(data[field])
        
        # Also extract from nested structures
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 20:  # Reasonable length text
                if key not in text_fields:  # Avoid duplicates
                    extracted_texts.append(f"{key}: {value}")
        
        return "\n".join(extracted_texts)
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size//2, end - 100), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _run(self, query: str, top_k: int = 5, threshold: float = 0.5) -> Dict[str, Any]:
        """Run vector search."""
        if not self.index_initialized:
            success = self.initialize_index()
            if not success:
                return {"success": False, "error": "Failed to initialize vector index"}
        
        if self.embedding_model is None:
            return {"success": False, "error": "Embedding model not initialized"}
        
        if self.faiss_index is None or len(self.document_metadata) == 0:
            return {"success": False, "error": "Vector index not available"}
        
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
            
            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_metadata) and score >= threshold:
                    doc = self.document_metadata[idx]
                    results.append({
                        'content': doc['content'],
                        'source': doc['source'],
                        'title': doc.get('title', ''),
                        'score': float(score),
                        'chunk_id': doc.get('chunk_id', 0),
                        'file_type': doc.get('file_type', ''),
                        'metadata': doc.get('metadata', {})
                    })
            
            return {
                "success": True,
                "results": results,
                "query": query,
                "total_found": len(results)
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for the tool parameters."""
        return {
            "query": {
                "type": "string",
                "description": "Search query to find relevant documents"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top results to return (default: 5)",
                "required": False
            },
            "threshold": {
                "type": "number",
                "description": "Minimum similarity threshold (0-1, default: 0.5)",
                "required": False
            }
        }