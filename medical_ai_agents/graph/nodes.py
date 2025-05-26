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
    """Synthesizes the final result from all agent outputs, including merging results from sub-agents if present."""
    logger = logging.getLogger("graph.nodes.synthesizer")
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    logger.info(f"Synthesizing results for task type: {task_type}")

    # --- NEW: Merge results from sub-agents if present ---
    merged = {}
    if "results" in state and isinstance(state["results"], dict):
        # results is a dict of {agent_name: agent_result_dict}
        for agent_name, agent_result in state["results"].items():
            for key in [
                "detector_result", "modality_result", "region_result", "vqa_result", "rag_result", "reflection_result"
            ]:
                if key in agent_result and agent_result[key]:
                    # Nếu là detector_result thì gộp objects
                    if key == "detector_result":
                        if "detector_result" not in merged:
                            merged["detector_result"] = {"objects": []}
                        # Gộp objects
                        objects = agent_result[key].get("objects", [])
                        merged["detector_result"]["objects"] += objects
                        # Gộp các trường khác nếu có
                        for k, v in agent_result[key].items():
                            if k != "objects":
                                merged["detector_result"][k] = v
                    else:
                        merged[key] = agent_result[key]
    # Merge lại vào state để ưu tiên các trường đã tổng hợp
    state = {**state, **merged}
    # --- END NEW ---

    # Start with a basic result structure
    final_result = {
        "task_type": task_type,
        "success": True,
        "session_id": state.get("session_id", ""),
        "query": state.get("query", "")
    }
    
    # ✅ FIX: Include detector results correctly - giữ nguyên toàn bộ dict
    detector_result = state.get("detector_result", {})
    logger.info(f"Detector result in synthesizer: {detector_result}")

    # Gán nguyên detector_result vào final_result nếu có
    if detector_result:
        final_result["detector_result"] = detector_result
        # Vẫn có thể lấy objects và polyp_count cho các trường hợp tổng hợp khác
        objects = detector_result.get("objects", [])
        final_result["polyps"] = objects
        final_result["polyp_count"] = len(objects)
        logger.info(f"Found {len(objects)} polyps from detector")
    else:
        final_result["detector_result"] = {}
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

            synthesis_prompt = """
Bạn là bác sĩ chuyên khoa tiêu hóa có nhiều năm kinh nghiệm. Hãy trả lời bệnh nhân với giọng điệu chuyên nghiệp, tận tâm và đầy thông cảm.
- Trình bày kết quả một cách tự nhiên, thân thiện, dễ hiểu, tránh khô khan.
- Không sử dụng các con số xác suất cứng nhắc (ví dụ: thay vì 'độ tin cậy 89%' hãy nói 'khả năng rất cao', 'có vẻ như', 'nhiều khả năng', 'có thể quan sát thấy', v.v.).
- Nhấn mạnh ý nghĩa lâm sàng của phát hiện, giải thích rõ ràng cho bệnh nhân.
- Nếu có điểm chưa chắc chắn, hãy diễn đạt một cách mềm mại, không gây lo lắng không cần thiết.
- Đưa ra lời khuyên hoặc hỏi thêm về quy trình nếu thực sự cần thiết.
"""
            # Chuẩn bị thông tin từ tất cả các agent
            vqa_info = ""
            if "vqa_result" in state and state["vqa_result"].get("success", False):
                vqa_result = state["vqa_result"]
                vqa_info = f"""
Thông tin từ phân tích hình ảnh (VQA):
- Câu trả lời: {vqa_result.get('answer', 'Không có thông tin')}
- Phân tích: {vqa_result.get('analysis', 'Không có phân tích chi tiết')}
"""

            modality_info = ""
            if "modality_result" in state and state["modality_result"].get("success", False):
                modality_result = state["modality_result"]
                modality_info = f"""
Thông tin về kỹ thuật nội soi:
- Loại kỹ thuật: {modality_result.get('class_name', 'Không xác định')}
- Đặc điểm: {modality_result.get('class_description', 'Không có thông tin chi tiết')}
"""

            region_info = ""
            if "region_result" in state and state["region_result"].get("success", False):
                region_result = state["region_result"]
                region_info = f"""
Thông tin về vùng giải phẫu:
- Vùng: {region_result.get('class_name', 'Không xác định')}
- Đặc điểm: {region_result.get('class_description', 'Không có thông tin chi tiết')} 
"""

            rag_info = ""
            if "rag_result" in state and state["rag_result"].get("success", False):
                rag_result = state["rag_result"]
                rag_info = f"""
Thông tin từ kiến thức y khoa:
- Câu trả lời: {rag_result.get('answer', 'Không có thông tin')}
- Nguồn: {', '.join(rag_result.get('sources', ['Không có nguồn']))}
"""

            reflection_info = ""
            if "reflection_result" in state and state.get("reflection_result"):
                reflection_result = state["reflection_result"]
                if reflection_result.get("bias_detected", False):
                    reflection_info = f"""
Thông tin từ quá trình kiểm tra sai lệch:
- Đã phát hiện sai lệch: Có
- Câu trả lời ban đầu: {reflection_result.get('original_answer', 'Không có')}
- Câu trả lời cải thiện: {reflection_result.get('improved_answer', 'Không có')}
"""

            # Tạo prompt tổng hợp
            if polyp_count > 0:
                # Has polyps
                prompt = f"""
{synthesis_prompt}
Câu hỏi của bệnh nhân: {query}

Kết quả phát hiện: Phát hiện {polyp_count} polyp
Chi tiết: {'; '.join(polyp_details)}

{vqa_info}
{modality_info}
{region_info}
{rag_info}
{reflection_info}

Hãy viết một câu trả lời chi tiết, toàn diện giúp bệnh nhân hiểu rõ tình trạng và các bước tiếp theo, bao gồm:
1. Mô tả chi tiết về những gì phát hiện được với ngôn ngữ dễ hiểu
2. Ngụ ý y khoa và mức độ quan trọng của phát hiện 
3. Đề xuất các bước tiếp theo (theo dõi, xét nghiệm thêm, điều trị...)
4. Cung cấp một thông điệp trấn an nhưng trung thực
5. Khuyến khích bệnh nhân chia sẻ thêm thông tin về tiền sử bệnh nếu cần thiết

Sử dụng giọng điệu y khoa chuyên nghiệp.
Trả lời theo định dạng sau:
```json
{{
    "answer": "câu trả lời chi tiết",
    "analysis": "phân tích chuyên sâu về ý nghĩa lâm sàng của các phát hiện"
}}
```
"""
            else:
                # No polyps
                prompt = f"""
{synthesis_prompt}
Câu hỏi của bệnh nhân: {query}

Kết quả phát hiện: Không phát hiện polyp nào

{vqa_info}
{modality_info}
{region_info}
{rag_info}
{reflection_info}

Hãy viết một câu trả lời chi tiết, trấn an nhưng chuyên nghiệp, bao gồm:
1. Xác nhận kết quả tốt từ hình ảnh nội soi
2. Giải thích ý nghĩa của việc không phát hiện polyp
3. Đề xuất các biện pháp phòng ngừa và theo dõi định kỳ nếu cần
4. Khuyến khích bệnh nhân duy trì lối sống lành mạnh
5. Hỏi về các triệu chứng hoặc lo lắng khác mà bệnh nhân có thể đang gặp phải

Sử dụng giọng điệu y khoa chuyên nghiệp nhưng ấm áp, đầy thông cảm.
Trả lời theo định dạng sau:
```json
{{
    "answer": "câu trả lời chi tiết",
    "analysis": "phân tích chuyên sâu về ý nghĩa lâm sàng của phát hiện"
}}
```
"""
        
        try:
            # Use higher temperature for more natural language generation
            custom_llm = ChatOpenAI(model=llm.model_name if hasattr(llm, 'model_name') else "gpt-4o-mini", temperature=0.7)
            answer = custom_llm.invoke([HumanMessage(content=prompt)]).content
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