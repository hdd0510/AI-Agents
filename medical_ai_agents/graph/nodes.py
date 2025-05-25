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


# Result Synthesizer Node
def result_synthesizer(state: SystemState, llm: ChatOpenAI) -> SystemState:
    """Synthesizes the final result from all agent outputs."""
    logger = logging.getLogger("graph.nodes.synthesizer")
    
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    logger.info(f"Synthesizing results for task type: {task_type}")
    
    # Start with a basic result structure
    final_result = {
        "task_type": task_type,
        "success": True,
        "session_id": state.get("session_id", ""),
        "query": state.get("query", "")
    }
    
    # ✅ FIX: Include detector results correctly - check both success and objects
    detector_result = state.get("detector_result", {})
    logger.info(f"Detector result in synthesizer: {detector_result}")
    
    if detector_result:
        # Even if success=False, still check for objects
        objects = detector_result.get("objects", [])
        
        # Use objects from detector regardless of success flag
        final_result["polyps"] = objects
        final_result["polyp_count"] = len(objects)
        
        logger.info(f"Found {len(objects)} polyps from detector")
    else:
        final_result["polyps"] = []
        final_result["polyp_count"] = 0
        logger.info("No detector result found")
    
    # Include classifier results if available
    modality_result = state.get("modality_result", {})
    if modality_result and modality_result.get("success", False):
        final_result["modality"] = {
            "class_name": modality_result.get("class_name", "unknown"),
            "confidence": modality_result.get("confidence", 0.0)
        }
    
    region_result = state.get("region_result", {})
    if region_result and region_result.get("success", False):
        final_result["region"] = {
            "class_name": region_result.get("class_name", "unknown"),
            "confidence": region_result.get("confidence", 0.0)
        }
    
    # Include answer - prefer reflection result if available
    reflection_result = state.get("reflection_result", {})
    if reflection_result:
        final_result["answer"] = reflection_result.get("improved_answer", "")
        final_result["original_answer"] = reflection_result.get("original_answer", "")
        final_result["answer_confidence"] = reflection_result.get("confidence", 0.0)
        final_result["bias_detected"] = reflection_result.get("bias_detected", False)
    else:
        vqa_result = state.get("vqa_result", {})
        if vqa_result and vqa_result.get("success", False):
            final_result["answer"] = vqa_result.get("answer", "")
            final_result["answer_confidence"] = vqa_result.get("confidence", 0.0)
    
    # Generate summary for comprehensive tasks
    if task_type == TaskType.COMPREHENSIVE:
        final_result["summary"] = _generate_summary(final_result)
    
    # Add timing information
    if "start_time" in state:
        final_result["processing_time"] = time.time() - state["start_time"]
    
    logger.info("Result synthesis complete")
    
    # ✅ FIX: Generate natural language answer based on ACTUAL detection results
    query = state.get("query", "")
    if query:
        polyp_count = final_result.get('polyp_count', 0)
        polyps = final_result.get('polyps', [])
        
        if polyp_count > 0:
            # Has polyps
            polyp_details = []
            for i, polyp in enumerate(polyps[:3]):  # Top 3 polyps
                conf = polyp.get('confidence', 0)
                pos = polyp.get('position_description', 'unknown position')
                polyp_details.append(f"Polyp {i+1}: {conf:.2f} confidence at {pos}")
            
            prompt = f"""
            Bạn là bác sĩ chuyên khoa tiêu hóa. Trả lời câu hỏi của bệnh nhân:
            
            Câu hỏi: {query}
            
            Kết quả phát hiện: Phát hiện {polyp_count} polyp
            Chi tiết: {'; '.join(polyp_details)}
            
            Trả lời ngắn gọn, chuyên nghiệp (1-2 câu):
            """
        else:
            # No polyps
            prompt = f"""
            Bạn là bác sĩ chuyên khoa tiêu hóa. Trả lời câu hỏi của bệnh nhân:
            
            Câu hỏi: {query}
            
            Kết quả phát hiện: Không phát hiện polyp nào
            
            Trả lời ngắn gọn, trấn an bệnh nhân (1-2 câu):
            """
        
        try:
            answer = llm.invoke([HumanMessage(content=prompt)]).content
            final_result["answer"] = answer
            logger.info(f"Generated answer: {answer}")
        except Exception as e:
            logger.error(f"Failed to generate answer: {str(e)}")
            # Fallback answer
            if polyp_count > 0:
                final_result["answer"] = f"Phát hiện {polyp_count} polyp trong hình ảnh."
            else:
                final_result["answer"] = "Không phát hiện polyp nào trong hình ảnh."
    
    # Update state
    return {**state, "final_result": final_result}



def _generate_summary(result: Dict[str, Any]) -> str:
    """Generates a summary from the comprehensive results."""
    summary = []
    
    # Summarize polyp detection
    polyp_count = result.get("polyp_count", 0)
    if polyp_count > 0:
        summary.append(f"Phát hiện {polyp_count} polyp trong hình ảnh.")
        
        # Describe largest polyp
        polyps = result.get("polyps", [])
        if polyps:
            # Find largest polyp by area
            largest_polyp = max(polyps, key=lambda p: p.get("area", 0))
            confidence = largest_polyp.get("confidence", 0)
            position = largest_polyp.get("position_description", "không xác định")
            
            summary.append(f"Polyp lớn nhất nằm ở vị trí {position} với độ tin cậy {confidence:.2f}.")
    else:
        summary.append("Không phát hiện polyp nào trong hình ảnh.")
    
    # Summarize region
    if "region" in result:
        region = result["region"]
        summary.append(f"Hình ảnh được chụp tại vị trí {region['class_name']} với độ tin cậy {region['confidence']:.2f}.")
    
    # Summarize modality
    if "modality" in result:
        modality = result["modality"]
        summary.append(f"Kỹ thuật nội soi sử dụng là {modality['class_name']} với độ tin cậy {modality['confidence']:.2f}.")
    
    # Summarize answer if available
    if "answer" in result:
        query = result.get("query", "")
        if query:
            summary.append(f"Trả lời cho câu hỏi: {result['answer']}")
    
    # Add bias warning if detected
    if result.get("bias_detected", False):
        summary.append("Lưu ý: Đã phát hiện và điều chỉnh bias trong câu trả lời.")
    
    return "\n".join(summary)   