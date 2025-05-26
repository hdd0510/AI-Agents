#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Agents - RAG Agent với LightRAG
-----------------------------------------
Agent RAG sử dụng LightRAG để truy xuất thông tin y tế từ knowledge base.
"""

import asyncio
import os
import json
from typing import Dict, Any, List, Optional
import logging

from medical_ai_agents.agents.base_agent import BaseAgent
from medical_ai_agents.tools.base_tools import BaseTool

class LightRAGTool(BaseTool):
    """Tool sử dụng LightRAG để thực hiện RAG operations."""
    
    def __init__(self, working_dir: str = "./rag_storage", device: str = "cuda", **kwargs):
        """Initialize LightRAG tool."""
        super().__init__(
            name="lightrag_query",
            description="Truy xuất thông tin y tế từ knowledge base sử dụng LightRAG với graph-based indexing."
        )
        self.working_dir = working_dir
        self.device = device
        self.rag = None
        self._initialize()
    
    def _initialize(self) -> bool:
        """Initialize LightRAG system."""
        try:
            # Import LightRAG components
            from lightrag import LightRAG, QueryParam
            from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
            from lightrag.utils import setup_logger
            
            # Setup logger
            setup_logger("lightrag", level="INFO")
            
            # Create working directory
            if not os.path.exists(self.working_dir):
                os.makedirs(self.working_dir, exist_ok=True)
            
            # Initialize LightRAG
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=gpt_4o_mini_complete,
                embedding_func=openai_embed,
                # Graph storage configuration
                graph_storage="NetworkXStorage",  # Default graph storage
                # Vector storage configuration  
                vector_storage="NanoVectorDBStorage",  # Default vector storage
                # KV storage configuration
                kv_storage="JsonKVStorage",  # Default key-value storage
                # Processing configuration
                chunk_token_size=1200,
                chunk_overlap_token_size=100,
                max_async=4,
                max_tokens=32768,
                # Retrieval configuration
                top_k=60,
                # Language configuration for medical domain
                language="english"
            )
            
            self.logger.info("LightRAG initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LightRAG: {str(e)}")
            return False
    
    def _run(self, query: str, mode: str = "hybrid", medical_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run RAG query."""
        if self.rag is None:
            return {"success": False, "error": "LightRAG not initialized"}
        
        try:
            # Import QueryParam
            from lightrag import QueryParam
            
            # Create enhanced query with medical context
            enhanced_query = self._enhance_query(query, medical_context)
            
            # Create query parameters
            query_param = QueryParam(
                mode=mode,  # hybrid, local, global, naive, mix
                # Additional parameters for medical domain
                only_need_context=False,
                response_type="stream" if mode == "stream" else "text"
            )
            
            # Execute query - LightRAG is async, so we need to handle it properly
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the async query
                result = loop.run_until_complete(
                    self.rag.aquery(enhanced_query, param=query_param)
                )
            finally:
                loop.close()
            
            # Parse and enhance result
            parsed_result = self._parse_result(result, query, mode)
            
            return {
                "success": True,
                "answer": parsed_result["answer"],
                "context": parsed_result.get("context", []),
                "confidence": parsed_result.get("confidence", 0.8),
                "sources": parsed_result.get("sources", []),
                "mode": mode,
                "query": query,
                "enhanced_query": enhanced_query
            }
            
        except Exception as e:
            import traceback
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _enhance_query(self, query: str, medical_context: Optional[Dict[str, Any]] = None) -> str:
        """Enhance query with medical context."""
        if not medical_context:
            return query
        
        context_parts = []
        
        # Add detection context
        if "detected_polyps" in medical_context:
            count = medical_context["detected_polyps"]
            if count > 0:
                context_parts.append(f"Given that {count} polyp(s) were detected in the image")
                
                if "polyp_details" in medical_context:
                    context_parts.append(f"with details: {medical_context['polyp_details']}")
        
        # Add imaging context
        if "imaging_modality" in medical_context:
            modality = medical_context["imaging_modality"]
            context_parts.append(f"using {modality} imaging technique")
        
        # Add anatomical context
        if "anatomical_region" in medical_context:
            region = medical_context["anatomical_region"]
            context_parts.append(f"in the {region} region")
        
        # Add patient context
        if "patient_history" in medical_context:
            history = medical_context["patient_history"]
            context_parts.append(f"considering patient history: {history}")
        
        # Combine context with query
        if context_parts:
            context_str = ", ".join(context_parts)
            enhanced_query = f"Medical context: {context_str}. Question: {query}"
        else:
            enhanced_query = query
        
        return enhanced_query
    
    def _parse_result(self, result: str, original_query: str, mode: str) -> Dict[str, Any]:
        """Parse and enhance LightRAG result."""
        # Basic parsing - LightRAG returns text response
        parsed = {
            "answer": result,
            "confidence": self._estimate_confidence(result),
            "sources": self._extract_sources(result),
            "context": []
        }
        
        # Add mode-specific insights
        if mode == "local":
            parsed["retrieval_type"] = "Entity-specific retrieval focusing on precise medical facts"
        elif mode == "global":
            parsed["retrieval_type"] = "Concept-level retrieval for broader medical understanding"
        elif mode == "hybrid":
            parsed["retrieval_type"] = "Combined entity and concept retrieval for comprehensive medical analysis"
        elif mode == "naive":
            parsed["retrieval_type"] = "Traditional vector similarity search"
        
        return parsed
    
    def _estimate_confidence(self, answer: str) -> float:
        """Estimate confidence based on answer quality."""
        if not answer or len(answer.strip()) < 10:
            return 0.3
        
        # Lower confidence indicators
        uncertainty_terms = [
            "i'm not sure", "unclear", "cannot determine", "difficult to say",
            "may be", "might be", "possibly", "uncertain", "not clear"
        ]
        
        answer_lower = answer.lower()
        confidence = 0.9
        
        for term in uncertainty_terms:
            if term in answer_lower:
                confidence -= 0.1
                break
        
        # Higher confidence indicators
        certainty_terms = [
            "according to", "based on", "research shows", "studies indicate",
            "medical literature", "clinical evidence"
        ]
        
        for term in certainty_terms:
            if term in answer_lower:
                confidence += 0.1
                break
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_sources(self, answer: str) -> List[str]:
        """Extract potential sources from answer."""
        # Simple heuristic to identify source mentions
        sources = []
        
        # Look for common source patterns
        source_patterns = [
            "according to", "based on", "from", "reference:",
            "study by", "research by", "published in"
        ]
        
        answer_lower = answer.lower()
        for pattern in source_patterns:
            if pattern in answer_lower:
                # This is a simplified extraction - could be enhanced
                sources.append(f"Referenced in answer via '{pattern}'")
        
        return sources
    
    def insert_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Insert documents into LightRAG knowledge base."""
        if self.rag is None:
            return {"success": False, "error": "LightRAG not initialized"}
        
        try:
            # Handle async insertion
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Insert documents asynchronously
                for doc in documents:
                    loop.run_until_complete(self.rag.ainsert(doc))
            finally:
                loop.close()
            
            return {
                "success": True,
                "inserted_count": len(documents),
                "message": f"Successfully inserted {len(documents)} documents into knowledge base"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return JSON schema for the tool parameters."""
        return {
            "query": {
                "type": "string",
                "description": "Medical question or query to search for"
            },
            "mode": {
                "type": "string",
                "description": "Retrieval mode: hybrid, local, global, naive, mix",
                "enum": ["hybrid", "local", "global", "naive", "mix"],
                "default": "hybrid"
            },
            "medical_context": {
                "type": "object",
                "description": "Optional medical context from other agents",
                "required": False
            }
        }


class RAGAgent(BaseAgent):
    """Agent RAG sử dụng LightRAG để truy xuất thông tin y tế."""
    
    def __init__(self, working_dir: str = "./rag_storage", 
                 knowledge_base_path: Optional[str] = None,
                 llm_model: str = "gpt-4o-mini", device: str = "cuda"):
        """
        Khởi tạo RAG Agent với LightRAG.
        
        Args:
            working_dir: Thư mục làm việc cho LightRAG
            knowledge_base_path: Đường dẫn đến file knowledge base (optional)
            llm_model: Mô hình LLM sử dụng làm controller
            device: Device để chạy (cuda/cpu)
        """
        self.working_dir = working_dir
        self.knowledge_base_path = knowledge_base_path
        
        super().__init__(name="RAG Agent", llm_model=llm_model, device=device)
        self.lightrag_tool = None
    
    def _register_tools(self) -> List[BaseTool]:
        """Register tools for this agent."""
        self.lightrag_tool = LightRAGTool(
            working_dir=self.working_dir,
            device=self.device
        )
        return [self.lightrag_tool]
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines this agent's role."""
        return """Bạn là một AI chuyên gia về truy xuất và phân tích thông tin y tế sử dụng RAG (Retrieval-Augmented Generation).
Nhiệm vụ của bạn là trả lời câu hỏi y tế dựa trên knowledge base sử dụng LightRAG với graph-based indexing.

Bạn có thể sử dụng công cụ sau:
1. lightrag_query: Truy xuất thông tin từ knowledge base y tế
   - Tham số: query (str), mode (str), medical_context (Dict, optional)
   - Các mode: hybrid (default), local, global, naive, mix
   - Kết quả: câu trả lời, context, độ tin cậy, và sources

Các mode retrieval:
- hybrid: Kết hợp entity-level và concept-level retrieval (khuyến nghị cho hầu hết trường hợp)
- local: Tập trung vào thông tin cụ thể, chi tiết (cho câu hỏi về thuật ngữ, định nghĩa)  
- global: Tập trung vào khái niệm rộng, mối quan hệ (cho câu hỏi về xu hướng, tổng quan)
- naive: Traditional vector similarity search
- mix: Kết hợp tất cả các phương pháp

Quy trình làm việc của bạn:
1. Phân tích câu hỏi y tế và xác định mode retrieval phù hợp
2. Tích hợp thông tin từ các agent khác (detector, classifier, VQA) vào medical_context
3. Sử dụng công cụ lightrag_query để truy xuất thông tin
4. Phân tích kết quả và đánh giá độ tin cậy
5. Tổng hợp câu trả lời cuối cùng với nguồn tham khảo

Khi trả lời:
- Ưu tiên sử dụng thông tin từ knowledge base
- Kết hợp với context từ hình ảnh nếu có
- Đưa ra độ tin cậy và sources
- Giải thích lý do chọn mode retrieval cụ thể
- Cảnh báo nếu thông tin không đầy đủ hoặc cần xác nhận thêm

Bạn phải trả về JSON với định dạng:
```json
{
  "rag_result": {
    "success": true/false,
    "answer": "câu trả lời chi tiết dựa trên knowledge base",
    "confidence": confidence_value,
    "sources": ["list of sources"],
    "retrieval_mode": "mode đã sử dụng",
    "context_used": "mô tả context từ các agent khác",
    "analysis": "phân tích về chất lượng và độ tin cậy của thông tin"
  }
}
```"""

    def initialize(self) -> bool:
        """Khởi tạo agent và load knowledge base nếu có."""
        try:
            # Tools are already initialized in _register_tools
            
            # Load knowledge base if provided
            if self.knowledge_base_path and os.path.exists(self.knowledge_base_path):
                self._load_knowledge_base()
            
            self.initialized = True
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG agent: {str(e)}")
            self.initialized = False
            return False
    
    def _load_knowledge_base(self):
        """Load knowledge base từ file."""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                if self.knowledge_base_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        documents = data
                    else:
                        documents = [json.dumps(data)]
                else:
                    # Plain text file
                    content = f.read()
                    documents = [content]
            
            # Insert into LightRAG
            result = self.lightrag_tool.insert_documents(documents)
            if result["success"]:
                self.logger.info(f"Loaded knowledge base: {result['message']}")
            else:
                self.logger.error(f"Failed to load knowledge base: {result['error']}")
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {str(e)}")
    
    def _extract_task_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract task-specific input from state."""
        # Get results from other agents for context
        detector_result = state.get("detector_result", {})
        modality_result = state.get("modality_result", {})
        region_result = state.get("region_result", {})
        vqa_result = state.get("vqa_result", {})
        
        # Build medical context
        medical_context = {}
        
        # Add detection context
        if detector_result and detector_result.get("success", False):
            objects = detector_result.get("objects", [])
            medical_context["detected_polyps"] = len(objects)
            
            if objects:
                # Add details about detected polyps
                polyp_details = []
                for i, obj in enumerate(objects[:3]):  # Top 3 objects
                    detail = f"Polyp {i+1}: {obj.get('confidence', 0):.2f} confidence"
                    if 'position_description' in obj:
                        detail += f", location: {obj['position_description']}"
                    polyp_details.append(detail)
                medical_context["polyp_details"] = "; ".join(polyp_details)
        
        # Add classification context
        if modality_result and modality_result.get("success", False):
            medical_context["imaging_modality"] = modality_result.get("class_name", "Unknown")
        
        if region_result and region_result.get("success", False):
            medical_context["anatomical_region"] = region_result.get("class_name", "Unknown")
        
        # Add VQA context if available
        if vqa_result and vqa_result.get("success", False):
            medical_context["previous_analysis"] = vqa_result.get("answer", "")
        
        # Add user-provided context
        user_context = state.get("medical_context", {})
        if user_context:
            medical_context.update(user_context)
        
        return {
            "query": state.get("query", ""),
            "medical_context": medical_context,
            "image_path": state.get("image_path", "")
        }
    
    def _format_task_input(self, task_input: Dict[str, Any]) -> str:
        """Format task input for LLM prompt."""
        query = task_input.get("query", "")
        context = task_input.get("medical_context", {})
        
        context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()]) if context else "None"
        
        return f"""Medical Query: {query if query else "Provide general medical information relevant to the context"}

Available Medical Context:
{context_str}

Hãy phân tích câu hỏi này và sử dụng công cụ lightrag_query để truy xuất thông tin y tế phù hợp.

Bước 1: Xác định mode retrieval phù hợp:
- hybrid: cho câu hỏi phức tạp cần cả thông tin cụ thể và tổng quan
- local: cho câu hỏi về định nghĩa, triệu chứng cụ thể
- global: cho câu hỏi về xu hướng, mối quan hệ rộng

Bước 2: Sử dụng công cụ:
Tool: lightrag_query
Parameters: {{"query": "câu hỏi", "mode": "mode_phù_hợp", "medical_context": context_dict}}

Bước 3: Phân tích kết quả và đưa ra câu trả lời cuối cùng."""
    
    def _format_synthesis_input(self) -> str:
        pass

    def _extract_agent_result(self, synthesis: str) -> Dict[str, Any]:
        """Extract agent result from LLM synthesis."""
        try:
            # Try to extract JSON
            json_start = synthesis.find('{')
            json_end = synthesis.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = synthesis[json_start:json_end]
                rag_result = json.loads(json_str)
                return rag_result
            
            # Fallback: Create result from synthesis text
            return {
                "rag_result": {
                    "success": True,
                    "answer": synthesis,
                    "confidence": 0.7,
                    "sources": [],
                    "retrieval_mode": "unknown",
                    "analysis": "Generated from LLM synthesis without explicit RAG result"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract agent result: {str(e)}")
            return {
                "rag_result": {
                    "success": False,
                    "error": str(e),
                    "answer": synthesis,
                    "confidence": 0.5
                }
            }
    
    def add_documents(self, documents: List[str]) -> Dict[str, Any]:
        """Add documents to knowledge base."""
        if not self.initialized:
            self.initialize()
        
        return self.lightrag_tool.insert_documents(documents)
    