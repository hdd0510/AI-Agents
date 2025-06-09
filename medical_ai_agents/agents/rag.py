#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LIGHT RAG AGENT - Simple Document Retrieval
=========================================
RAG agent đơn giản để xử lý PDF và DOC files upload.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import os
from medical_ai_agents.agents.base_agent import BaseAgent, ThoughtType, ReActStep
from medical_ai_agents.tools.base_tools import BaseTool
from medical_ai_agents.tools.rag.doc_reader import PDFDocReaderTool
from medical_ai_agents.tools.rag.vector_search import VectorSearchTool
from medical_ai_agents.tools.rag.chunk_retriever import ChunkRetrieverTool

class RAGAgent(BaseAgent):
    """
    Light RAG Agent for Document Q&A
    
    WORKFLOW:
    1. Load and parse uploaded PDFs/DOCs
    2. Create vector embeddings
    3. Search relevant chunks based on query
    4. Synthesize answer from retrieved chunks
    """
    
    def __init__(self, 
                 storage_path: str = "./rag_storage",
                 llm_model: str = "gpt-4o-mini", 
                 device: str = "cuda",
                 chunk_size: int = 500,
                 overlap: int = 50):
        """Initialize Light RAG Agent."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        super().__init__(name="Light RAG Agent", llm_model=llm_model, device=device)
        
        # Configuration
        self.max_iterations = 4  # Read -> Index -> Search -> Answer
        
        # Tools will be initialized in _register_tools
        self.reader_tool = None
        self.search_tool = None
        self.retriever_tool = None
    
    def _register_tools(self) -> List[BaseTool]:
        """Register RAG tools."""
        self.reader_tool = PDFDocReaderTool(
            storage_path=str(self.storage_path),
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
        
        self.search_tool = VectorSearchTool(
            storage_path=str(self.storage_path),
            device=self.device
        )
        
        self.retriever_tool = ChunkRetrieverTool(
            storage_path=str(self.storage_path)
        )
        
        return [self.reader_tool, self.search_tool, self.retriever_tool]
    
    def _get_agent_description(self) -> str:
        """Agent description."""
        return """I am a document retrieval specialist that can:
        
1. Read and parse PDF/DOC files
2. Create searchable vector indexes
3. Find relevant information based on queries
4. Synthesize comprehensive answers from documents

I use a simple but effective RAG pipeline optimized for medical documents."""

    def _get_system_prompt(self) -> str:
        """System prompt for RAG agent."""
        return f"""You are a document retrieval and Q&A specialist using RAG (Retrieval-Augmented Generation).

AVAILABLE TOOLS:
{self.tool_descriptions}

WORKFLOW:
1. First, check if documents are already indexed using chunk_retriever
2. If new documents uploaded, use pdf_doc_reader to parse them
3. Use simple_vector_search to find relevant chunks based on the query
4. Evaluate query complexity:
   - If it's a COMPLEX MEDICAL QUESTION, prepare context for VQA Agent
   - If it's a SIMPLE DOCUMENT QUESTION, answer directly yourself

EVALUATION CRITERIA:
- Complex medical questions: require specialized medical knowledge, involve diagnoses, treatments, complex procedures
- Simple document questions: factual information directly from documents, basic explanations, definitions

RULES:
- Always cite which document and page the information comes from
- If no relevant information found, clearly state that
- Prioritize accuracy over completeness
- Use medical terminology appropriately when dealing with medical documents

Follow the ReAct format:
Thought: [your reasoning]
Action: [tool name or Final Answer]
Action Input: {{"param": "value"}}"""

    def initialize(self) -> bool:
        """Initialize RAG agent."""
        try:
            # Check if vector index exists
            index_path = self.storage_path / "vector_index.pkl"
            if index_path.exists():
                self.logger.info("Found existing vector index")
            else:
                self.logger.info("No existing index found, will create on first document")
            
            self.initialized = True
            self.logger.info("Light RAG Agent initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG Agent: {str(e)}")
            return False

    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract RAG task input."""
        # Check for uploaded documents
        uploaded_docs = state.get("uploaded_documents", [])
        
        # Get context from other agents
        medical_context = self._build_rag_context(state)
        
        return {
            "query": state.get("query", ""),
            "uploaded_documents": uploaded_docs,
            "medical_context": medical_context,
            "require_sources": True  # Always cite sources
        }

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
        
        # Add classification context
        if "modality_result" in state:
            modality = state["modality_result"]
            if modality.get("success", False):
                context["imaging_type"] = modality.get("class_name", "Unknown")
        
        if "region_result" in state:
            region = state["region_result"]
            if region.get("success", False):
                context["anatomical_region"] = region.get("class_name", "Unknown")
        
        return context

    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for RAG processing."""
        query = task_input.get("query", "")
        uploaded_docs = task_input.get("uploaded_documents", [])
        medical_context = task_input.get("medical_context", {})
        
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
        
        return f"""**DOCUMENT RETRIEVAL TASK**

User Query: "{query if query else 'Please analyze the uploaded documents'}"

Medical Context:
{context_str}
{doc_info}

Your task:
1. If new documents uploaded, read and index them first
2. Search for relevant information based on the query
3. Provide comprehensive answer with proper citations
4. Prepare a summary for VQA to help answer the query better

Begin with checking document status:"""

    def _format_agent_result(self, react_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format RAG result."""
        if not react_result.get("success", False):
            return {
                "rag_result": {
                    "success": False,
                    "error": react_result.get("error", "RAG processing failed"),
                    "approach": "light_rag"
                }
            }
        
        # Extract answer and sources from ReAct history
        answer = react_result.get("answer", "")
        sources = []
        chunks_retrieved = 0
        documents_processed = []
        vqa_summary = ""
        query_complexity = "simple"  # Default to simple
        
        # Analyze ReAct history
        if hasattr(self, 'react_history'):
            for step in self.react_history:
                if step.observation and step.action:
                    try:
                        obs_data = json.loads(step.observation)
                        
                        # Track document processing
                        if "pdf_doc_reader" in step.action and obs_data.get("success"):
                            if "documents_processed" in obs_data:
                                documents_processed.extend(obs_data["documents_processed"])
                        
                        # Track search results
                        if "vector_search" in step.action and obs_data.get("success"):
                            if "chunks" in obs_data:
                                chunks_retrieved = len(obs_data["chunks"])
                                # Extract sources
                                for chunk in obs_data["chunks"]:
                                    source = {
                                        "document": chunk.get("source", "Unknown"),
                                        "page": chunk.get("page", 0),
                                        "score": chunk.get("score", 0.0)
                                    }
                                    if source not in sources:
                                        sources.append(source)
                    except:
                        continue
                
                # Check if a complexity assessment was made in the thought
                if step.thought and ("complex medical" in step.thought.lower() or "specialized knowledge" in step.thought.lower()):
                    query_complexity = "complex"
                
                # Look for explicit complexity evaluation
                if step.thought and "query complexity:" in step.thought.lower():
                    thought_lower = step.thought.lower()
                    if "complex" in thought_lower and "medical" in thought_lower:
                        query_complexity = "complex"
                    elif "simple" in thought_lower and ("document" in thought_lower or "factual" in thought_lower):
                        query_complexity = "simple"
        
        # Create VQA summary if needed for complex queries
        if query_complexity == "complex" and chunks_retrieved > 0:
            vqa_summary = self._create_vqa_summary(react_result)
        
        # Create final result
        rag_result = {
            "success": True,
            "answer": answer,
            "sources": sources[:5],  # Top 5 sources
            "chunks_retrieved": chunks_retrieved,
            "documents_processed": list(set(documents_processed)),
            "approach": "light_rag",
            "has_citations": len(sources) > 0,
            "query_complexity": query_complexity,
            "vqa_summary": vqa_summary if query_complexity == "complex" else ""
        }
        
        # Add quality metrics
        if chunks_retrieved > 0:
            rag_result["retrieval_quality"] = "high" if chunks_retrieved >= 3 else "moderate"
        else:
            rag_result["retrieval_quality"] = "no_retrieval"
        
        return {"rag_result": rag_result}

    def _create_vqa_summary(self, react_result: Dict[str, Any]) -> str:
        """Create a summary for VQA from retrieved chunks."""
        try:
            # Extract chunks from ReAct history
            chunks = []
            if hasattr(self, 'react_history'):
                for step in self.react_history:
                    if step.observation and "vector_search" in step.action:
                        try:
                            obs_data = json.loads(step.observation)
                            if obs_data.get("success") and "chunks" in obs_data:
                                chunks.extend(obs_data["chunks"])
                        except:
                            continue
            
            if not chunks:
                return ""
            
            # Sort chunks by relevance score
            chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            # Take top 3 most relevant chunks
            top_chunks = chunks[:3]
            
            # Create summary
            summary_parts = []
            for chunk in top_chunks:
                content = chunk.get("content", "").strip()
                source = chunk.get("source", "Unknown")
                page = chunk.get("page", 0)
                score = chunk.get("score", 0)
                
                if content:
                    summary_parts.append(f"From {source} (page {page}, relevance: {score:.2f}):\n{content}\n")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Failed to create VQA summary: {str(e)}")
            return ""

    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Public method to add documents to RAG system."""
        if not self.initialized:
            self.initialize()
        
        try:
            # Use reader tool to process documents
            result = self.reader_tool._run(file_paths=file_paths)
            
            if result.get("success"):
                # Trigger indexing
                self.search_tool.update_index()
                
                return {
                    "success": True,
                    "message": f"Successfully processed {len(result.get('documents_processed', []))} documents",
                    "documents": result.get("documents_processed", [])
                }
            else:
                return result
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to add documents: {str(e)}"
            }