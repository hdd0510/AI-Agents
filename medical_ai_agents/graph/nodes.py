#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI Graph - Nodes
---------------------
Các node chức năng cho LangGraph.
"""

import json
import logging
from typing import Dict, Any, List
import time

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser, HumanMessage

from medical_ai_agents.config import SystemState, TaskType, ReflectionResult

# Task Analyzer Node
def task_analyzer(state: SystemState, llm: ChatOpenAI) -> Dict:
    """Analyzes the query and determines the task type and execution plan."""
    logger = logging.getLogger("graph.nodes.task_analyzer")
    
    # If no query, default to comprehensive analysis
    if not state.get("query"):
        logger.info("No query provided, defaulting to comprehensive analysis")
        return {
            **state, 
            "task_type": TaskType.COMPREHENSIVE
        }
    
    query = state["query"]
    logger.info(f"Analyzing query: {query}")
    
    # Use LLM to analyze query
    prompt = PromptTemplate.from_template(
        """Phân tích yêu cầu sau và xác định loại tác vụ phù hợp nhất:
        
        Yêu cầu: {query}
        
        Loại tác vụ có thể là một trong các loại sau:
        - polyp_detection: Phát hiện polyp
        - modality_classification: Phân loại kỹ thuật nội soi (BLI, WLI, ...)
        - region_classification: Phân loại vị trí trong đường tiêu hóa
        - medical_qa: Trả lời câu hỏi y tế về hình ảnh
        - comprehensive: Yêu cầu tổng hợp nhiều loại phân tích
        
        Trả về duy nhất tên loại tác vụ phù hợp nhất (không có giải thích)."""
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        task_result = chain.invoke({"query": query})
        task_type = task_result.strip().lower()
        logger.info(f"LLM task analysis result: {task_type}")
        
        # Map to TaskType enum
        if "polyp_detection" in task_type:
            task = TaskType.POLYP_DETECTION
        elif "modality_classification" in task_type:
            task = TaskType.MODALITY_CLASSIFICATION
        elif "region_classification" in task_type:
            task = TaskType.REGION_CLASSIFICATION
        elif "medical_qa" in task_type:
            task = TaskType.MEDICAL_QA
        else:
            task = TaskType.COMPREHENSIVE
        
        logger.info(f"Final task type: {task}")
        return {**state, "task_type": task}
        
    except Exception as e:
        logger.warning(f"LLM analysis failed: {str(e)}, falling back to keyword-based analysis")
        
        # Fallback to simple keyword-based analysis
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["polyp", "tổn thương", "khối", "u", "phát hiện"]):
            task = TaskType.POLYP_DETECTION
        elif any(kw in query_lower for kw in ["bli", "wli", "fice", "lci", "kỹ thuật", "phương pháp nội soi"]):
            task = TaskType.MODALITY_CLASSIFICATION
        elif any(kw in query_lower for kw in ["vị trí", "hang vị", "thân vị", "tâm vị", "bờ cong", "thực quản"]):
            task = TaskType.REGION_CLASSIFICATION
        elif any(kw in query_lower for kw in ["?", "tại sao", "làm sao", "như thế nào", "mức độ", "chẩn đoán"]):
            task = TaskType.MEDICAL_QA
        else:
            task = TaskType.COMPREHENSIVE
        
        logger.info(f"Fallback task type: {task}")
        return {**state, "task_type": task}


# Reflection Node
def reflection_node(state: SystemState, llm: ChatOpenAI) -> SystemState:
    """Performs reflection on VQA results to detect and correct biases."""
    logger = logging.getLogger("graph.nodes.reflection")
    
    # Skip if reflection not needed
    if not _needs_reflection(state):
        logger.info("Reflection not needed, skipping")
        return state
    
    vqa_result = state.get("vqa_result", {})
    if not vqa_result or not vqa_result.get("success", False):
        logger.warning("No valid VQA result for reflection")
        return state
    
    logger.info("Performing reflection on VQA result")
    original_answer = vqa_result.get("answer", "")
    query = state.get("query", "")
    
    # Get context from other agents
    detector_result = state.get("detector_result", {})
    modality_result = state.get("modality_result", {})
    region_result = state.get("region_result", {})
    
    # Create prompt for reflection
    prompt = PromptTemplate.from_template(
        """Bạn là chuyên gia y tế cao cấp, đang phân tích câu trả lời của AI cho một câu hỏi về hình ảnh y tế.
        
        Câu hỏi hiện tại: "{query}"
        
        Câu trả lời hiện tại: "{original_answer}"
        
        Độ tin cậy: {confidence:.2f}
        
        Thông tin từ phát hiện:
        {detection_info}
        
        Thông tin từ phân loại:
        - Kỹ thuật nội soi: {modality} (độ tin cậy: {modality_confidence:.2f})
        - Vị trí: {region} (độ tin cậy: {region_confidence:.2f})
        
        Nhiệm vụ của bạn là:
        1. Phân tích chất lượng và độ tin cậy của câu trả lời
        2. Kiểm tra tính nhất quán giữa câu trả lời và kết quả phát hiện
        3. Đánh giá câu trả lời dựa trên bằng chứng y tế
        4. Đưa ra câu trả lời cải thiện dựa trên phân tích của bạn
        
        Định dạng phản hồi:
        {{
            "analysis": "Phân tích chi tiết về câu trả lời",
            "bias_detected": true/false,
            "improved_answer": "Câu trả lời đã cải thiện",
            "confidence": 0.0-1.0
        }}"""
    )
    
    # Prepare detection info
    detection_info = ""
    objects = detector_result.get("objects", [])
    if objects:
        detection_info = f"Phát hiện {len(objects)} polyp trong hình ảnh.\n"
        for i, obj in enumerate(objects[:3]):  # Top 3 objects
            detection_info += f"- Polyp {i+1}: {obj.get('class_name', 'polyp')} "
            detection_info += f"(độ tin cậy: {obj.get('confidence', 0):.2f}), "
            detection_info += f"vị trí: {obj.get('position_description', 'không xác định')}\n"
    else:
        detection_info = "Không phát hiện polyp nào trong hình ảnh."
    
    # Chain for reflection
    chain = prompt | llm | StrOutputParser()
    
    try:
        reflection_text = chain.invoke({
            "query": query,
            "original_answer": original_answer,
            "confidence": vqa_result.get("confidence", 0.5),
            "detection_info": detection_info,
            "modality": modality_result.get("class_name", "Unknown"),
            "modality_confidence": modality_result.get("confidence", 0.0),
            "region": region_result.get("class_name", "Unknown"),
            "region_confidence": region_result.get("confidence", 0.0)
        })
        
        logger.debug(f"Reflection result: {reflection_text}")
        
        # Parse reflection result
        try:
            # Find and extract JSON
            json_start = reflection_text.find('{')
            json_end = reflection_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = reflection_text[json_start:json_end]
                reflection_data = json.loads(json_str)
                
                # Create reflection result
                reflection_result: ReflectionResult = {
                    "original_answer": original_answer,
                    "improved_answer": reflection_data.get("improved_answer", original_answer),
                    "bias_detected": reflection_data.get("bias_detected", False),
                    "confidence": reflection_data.get("confidence", vqa_result.get("confidence", 0.5))
                }
                
                logger.info(f"Reflection complete. Bias detected: {reflection_result['bias_detected']}")
                return {**state, "reflection_result": reflection_result}
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            logger.warning("Failed to parse reflection result JSON")
        
        # Fallback reflection result
        reflection_result: ReflectionResult = {
            "original_answer": original_answer,
            "improved_answer": original_answer,
            "bias_detected": False,
            "confidence": vqa_result.get("confidence", 0.5)
        }
        
        logger.info("Using fallback reflection result")
        return {**state, "reflection_result": reflection_result}
        
    except Exception as e:
        logger.error(f"Reflection failed: {str(e)}")
        return state


def _needs_reflection(state: SystemState) -> bool:
    """Determines if reflection is needed based on VQA result."""
    vqa_result = state.get("vqa_result", {})
    
    # No reflection needed if VQA failed
    if not vqa_result or not vqa_result.get("success", False):
        return False
    
    # Check confidence - low confidence needs reflection
    confidence = vqa_result.get("confidence", 1.0)
    if confidence < 0.7:
        return True
    
    # Check for uncertainty in answer
    answer = vqa_result.get("answer", "").lower()
    uncertainty_phrases = ["có thể", "không chắc chắn", "khó xác định", "có lẽ"]
    if any(phrase in answer for phrase in uncertainty_phrases):
        return True
    
    return False


# ===== SIMPLIFIED RESULT SYNTHESIZER =====
def result_synthesizer(state: SystemState, llm: ChatOpenAI) -> SystemState:
    """
    Simplified synthesizer - giữ nguyên analysis của các agent,
    chỉ tổng hợp final answer.
    """
    logger = logging.getLogger("graph.nodes.synthesizer")
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    logger.info(f"Synthesizing results for task type: {task_type}")

    # --- Merge results from sub-agents if present ---
    merged = {}
    if "results" in state and isinstance(state["results"], dict):
        for agent_name, agent_result in state["results"].items():
            for key in [
                "detector_result", "modality_result", "region_result", 
                "vqa_result", "rag_result", "reflection_result"
            ]:
                if key in agent_result and agent_result[key]:
                    if key == "detector_result":
                        if "detector_result" not in merged:
                            merged["detector_result"] = {"objects": []}
                        objects = agent_result[key].get("objects", [])
                        merged["detector_result"]["objects"] += objects
                        for k, v in agent_result[key].items():
                            if k != "objects":
                                merged["detector_result"][k] = v
                    else:
                        merged[key] = agent_result[key]
    
    state = {**state, **merged}

    # === SIMPLE STRUCTURED RESULT ===
    final_result = {
        # Metadata
        "task_type": task_type,
        "success": True,
        "session_id": state.get("session_id", ""),
        "query": state.get("query", ""),
        "timestamp": time.time(),
        
        # === INDIVIDUAL AGENT RESULTS (KEEP AS-IS) ===
        "agent_results": {},
        
        # === FINAL ANSWER (SYNTHESIZED) ===
        "final_answer": "",
        
        # Performance
        "processing_time": 0.0
    }

    # === POPULATE AGENT RESULTS (KEEP ORIGINAL ANALYSIS) ===
    agent_keys = ["detector_result", "vqa_result", "modality_result", 
                  "region_result", "rag_result", "reflection_result"]
    
    for key in agent_keys:
        if key in state and state[key]:
            final_result["agent_results"][key] = state[key]
            logger.info(f"Added {key} to results")

    # === GENERATE FINAL ANSWER ===
    final_result["final_answer"] = _generate_final_answer(final_result, llm)
    
    # === ADD PERFORMANCE METRICS ===
    if "start_time" in state:
        final_result["processing_time"] = time.time() - state["start_time"]

    logger.info("Simplified result synthesis complete")
    
    return {**state, "final_result": final_result}


def _generate_final_answer(result: Dict[str, Any], llm: ChatOpenAI) -> str:
    """
    Generate a comprehensive, user-friendly answer using LLM with adaptive prompting.
    """
    logger = logging.getLogger("synthesizer")
    agent_results = result["agent_results"]
    query = result.get("query", "")
    task_type = result.get("task_type", "")
    
    # Analyze query type first
    query_lower = query.lower()
    is_greeting = any(word in query_lower for word in ["hello", "hi", "xin chào", "chào"])
    is_identity = any(phrase in query_lower for phrase in ["who are you", "bạn là ai", "what are you", "you are"])
    is_general_chat = any(phrase in query_lower for phrase in ["how are", "thank", "cảm ơn", "help me", "can you"])
    is_medical = any(word in query_lower for word in ["polyp", "nội soi", "đau", "bệnh", "thuốc", "triệu chứng", "chẩn đoán"])
    
    # Prepare context from agents
    context_parts = []
    has_detection = False
    has_medical_analysis = False
    
    if agent_results.get("detector_result", {}).get("success"):
        has_detection = True
        count = agent_results["detector_result"].get("count", 0)
        if count > 0:
            context_parts.append(f"Detection: Found {count} polyp(s)")
            objects = agent_results["detector_result"].get("objects", [])
            for i, obj in enumerate(objects[:3]):
                conf = obj.get("confidence", 0)
                pos = obj.get("position_description", "unknown")
                context_parts.append(f"- Polyp {i+1}: {conf:.2%} confidence, position {pos}")
        else:
            context_parts.append("Detection: No polyps found")
    
    if agent_results.get("modality_result", {}).get("success"):
        context_parts.append(f"Imaging technique: {agent_results['modality_result'].get('class_name', 'unknown')}")
        has_medical_analysis = True
    
    if agent_results.get("region_result", {}).get("success"):
        context_parts.append(f"Anatomical region: {agent_results['region_result'].get('class_name', 'unknown')}")
        has_medical_analysis = True
    
    if agent_results.get("vqa_result", {}).get("success"):
        vqa_answer = agent_results["vqa_result"].get("answer", "")
        if vqa_answer:
            context_parts.append(f"VQA Analysis: {vqa_answer[:200]}...")
            has_medical_analysis = True
    
    context_str = "\n".join(context_parts) if context_parts else "No specific analysis results"
    
    # Create adaptive prompt based on query type
    if is_greeting or is_identity or is_general_chat:
        # Conversational mode
        prompt = f"""
You are a friendly and professional Medical AI Assistant specializing in gastrointestinal endoscopy analysis. 
Your personality is warm, helpful, and approachable while maintaining medical professionalism when needed.

User query: "{query}"

Context from analysis (if any):
{context_str}

Instructions:
- If this is a greeting, respond warmly and briefly introduce your capabilities
- If asked about your identity, explain you're an AI assistant for medical image analysis
- For general questions, be conversational but guide back to how you can help with medical analysis
- Keep responses natural, friendly, and concise
- Don't use medical jargon unless specifically discussing medical topics
- If relevant, mention what you can help with (endoscopy analysis, polyp detection, medical Q&A)

Respond naturally in the same language as the user's query.
"""
    
    elif is_medical or has_detection or has_medical_analysis:
        # Medical mode - original prompt but enhanced
        prompt = f"""
You are an experienced gastroenterology specialist providing consultation through an AI system.
Balance medical expertise with clear, accessible communication.

Patient query: "{query}"

Analysis results:
{context_str}

Guidelines:
- Provide medically accurate information in patient-friendly language
- Start directly with the relevant answer - no greetings or formalities
- Integrate all analysis results into a cohesive medical assessment
- Structure your response logically: findings → interpretation → recommendations
- Use medical terms when necessary but always explain them
- Be empathetic and reassuring while maintaining professionalism
- Include appropriate disclaimers about AI limitations and need for in-person consultation
- Avoid lists or bullet points - use flowing paragraphs
- Keep the tone confident but not overly assertive

Provide a comprehensive yet accessible medical response.
"""
    
    else:
        # Hybrid mode - could be either medical or general
        prompt = f"""
You are a Medical AI Assistant with expertise in gastrointestinal endoscopy and general medical knowledge.
Adapt your response style based on the user's question.

User query: "{query}"

Available context:
{context_str}

Instructions:
- Analyze if this is a medical question or general inquiry
- For medical questions: provide professional, accurate medical information
- For general questions: be conversational and helpful
- Always maintain a warm, professional tone
- If unsure about the query intent, offer both general assistance and medical capabilities
- Keep responses natural and well-structured
- Don't over-explain unless the question requires detail

Respond appropriately to the user's needs.
"""
    
    # Add text-only handling
    if not has_detection and not has_medical_analysis and query:
        prompt += """

Note: This is a text-only consultation without image analysis. Base your response on:
- Medical knowledge and best practices
- The specific question asked
- General medical guidelines
- Always recommend proper medical examination when appropriate
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        answer = response.content.strip()
        
        # Post-process to ensure quality
        if len(answer) < 50 and not (is_greeting or is_general_chat):
            # Too short for a medical response, try again with emphasis
            retry_prompt = prompt + "\n\nPlease provide a more detailed and complete response."
            response = llm.invoke([HumanMessage(content=retry_prompt)])
            answer = response.content.strip()
        
        return answer
        
    except Exception as e:
        logger.error(f"Failed to generate answer from LLM: {str(e)}")
        return _generate_fallback_answer(agent_results)

def _generate_fallback_answer(agent_results: Dict[str, Any]) -> str:
    """Generate simple fallback answer."""
    
    detector_result = agent_results.get("detector_result", {})
    
    if detector_result.get("success"):
        count = detector_result.get("count", 0)
        if count > 0:
            return f"Kết quả phân tích cho thấy phát hiện {count} polyp trong hình ảnh. Bạn nên tham khảo ý kiến bác sĩ chuyên khoa để được tư vấn thêm về kết quả này."
        else:
            return "Kết quả phân tích cho thấy không phát hiện polyp nào trong hình ảnh. Đây là kết quả tích cực cho sức khỏe của bạn."
    else:
        return "Đã hoàn thành phân tích hình ảnh. Vui lòng tham khảo thêm ý kiến bác sĩ chuyên khoa nếu cần thiết."