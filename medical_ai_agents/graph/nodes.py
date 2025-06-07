"""
Medical AI Graph - Nodes (MODIFIED for multi-task support)
---------------------
nodes với multi-task analysis và smart routing.
"""

import json
import logging
from typing import Dict, Any, List
import time

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser, HumanMessage

from medical_ai_agents.config import SystemState, TaskType

# Task Analyzer với Multi-Task Support
def task_analyzer(state: SystemState, llm: ChatOpenAI) -> Dict:
    """task analyzer với multi-task parsing support."""
    logger = logging.getLogger("graph.nodes.task_analyzer")
    
    query = state.get("query", "")
    is_text_only = state.get("is_text_only", False)
    
    if is_text_only:
        logger.info("Text-only mode detected")
        return {
            **state,
            "task_type": TaskType.TEXT_ONLY,
            "required_tasks": ["medical_qa"],
            "completed_tasks": [],
            "execution_order": ["medical_qa"]
        }
    
    if not query:
        logger.info("No query provided, defaulting to comprehensive analysis")
        return {
            **state,
            "task_type": TaskType.COMPREHENSIVE,
            "required_tasks": ["polyp_detection", "modality_classification", "region_classification"],
            "completed_tasks": [],
            "execution_order": ["polyp_detection", "modality_classification", "region_classification"]
        }
    
    logger.info(f"Analyzing multi-task query: {query}")
    
    # prompt for multi-task analysis
    prompt = PromptTemplate.from_template(
        """Phân tích yêu cầu sau và xác định các tác vụ cần thiết để trả lời đầy đủ:
        
        Yêu cầu: {query}
        
        Các tác vụ có thể bao gồm (có thể chọn nhiều tác vụ):
        - polyp_detection: Phát hiện polyp và đối tượng bất thường
        - modality_classification: Phân loại kỹ thuật nội soi (BLI, WLI, FICE, LCI)
        - region_classification: Phân loại vị trí giải phẫu trong đường tiêu hóa
        - medical_qa: Trả lời câu hỏi y tế, tư vấn, giải thích
        
        Hướng dẫn:
        - Nếu hỏi về polyp/tổn thương/phát hiện → bao gồm polyp_detection
        - Nếu hỏi về kỹ thuật/modality/BLI/WLI → bao gồm modality_classification
        - Nếu hỏi về vị trí/anatomy/region → bao gồm region_classification
        - Nếu cần giải thích/tư vấn/phân tích → bao gồm medical_qa
        - Câu hỏi phức tạp có thể cần nhiều tác vụ
        
        Trả về danh sách các tác vụ cần thiết, cách nhau bởi dấu phẩy.
        Ví dụ: polyp_detection, medical_qa
        Hoặc: modality_classification, region_classification, medical_qa
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        # Get LLM analysis
        task_result = chain.invoke({"query": query})
        logger.info(f"LLM task analysis result: {task_result}")
        
        # Parse multiple tasks
        required_tasks = _parse_multiple_tasks(task_result.strip())
        
        if not required_tasks:
            # Fallback to keyword-based analysis
            required_tasks = _keyword_based_task_analysis(query)
        
        # Determine execution order
        execution_order = _determine_execution_order(required_tasks)
        
        # Set task type
        if len(required_tasks) == 1:
            task_type = TaskType(required_tasks[0])
        else:
            task_type = TaskType.MULTI_TASK
        
        logger.info(f"Final analysis - Tasks: {required_tasks}, Order: {execution_order}")
        
        return {
            **state,
            "task_type": task_type,
            "required_tasks": required_tasks,
            "completed_tasks": [],
            "execution_order": execution_order,
            "current_task": execution_order[0] if execution_order else None
        }
        
    except Exception as e:
        logger.error(f"task analysis failed: {str(e)}")
        # Fallback to comprehensive
        return {
            **state,
            "task_type": TaskType.COMPREHENSIVE,
            "required_tasks": ["polyp_detection", "modality_classification", "region_classification"],
            "completed_tasks": [],
            "execution_order": ["polyp_detection", "modality_classification", "region_classification"]
        }


def _parse_multiple_tasks(task_result: str) -> List[str]:
    """Parse comma-separated tasks from LLM output."""
    logger = logging.getLogger("graph.nodes.task_parser")
    
    # Clean and split
    tasks = [task.strip().lower() for task in task_result.split(",")]
    
    # Valid task names
    valid_tasks = {
        "polyp_detection", "modality_classification", 
        "region_classification", "medical_qa"
    }
    
    # Filter valid tasks
    parsed_tasks = []
    for task in tasks:
        # Handle variations in naming
        if task in valid_tasks:
            parsed_tasks.append(task)
        elif "polyp" in task or "detection" in task:
            parsed_tasks.append("polyp_detection")
        elif "modality" in task or "classification" in task and ("bli" in task or "wli" in task):
            parsed_tasks.append("modality_classification")
        elif "region" in task or "location" in task or "anatomy" in task:
            parsed_tasks.append("region_classification")
        elif "qa" in task or "question" in task or "medical" in task:
            parsed_tasks.append("medical_qa")
    
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for task in parsed_tasks:
        if task not in seen:
            seen.add(task)
            result.append(task)
    
    logger.info(f"Parsed tasks: {result}")
    return result


def _keyword_based_task_analysis(query: str) -> List[str]:
    """Fallback keyword-based task analysis."""
    logger = logging.getLogger("graph.nodes.keyword_analyzer")
    
    query_lower = query.lower()
    required_tasks = []
    
    # Detection keywords
    if any(kw in query_lower for kw in ["polyp", "tổn thương", "phát hiện", "detect", "find", "abnormal", "lesion"]):
        required_tasks.append("polyp_detection")
    
    # Modality keywords
    if any(kw in query_lower for kw in ["bli", "wli", "fice", "lci", "technique", "modality", "imaging", "kỹ thuật"]):
        required_tasks.append("modality_classification")
    
    # Region keywords
    if any(kw in query_lower for kw in ["location", "region", "anatomy", "vị trí", "hang vị", "thân vị", "antrum", "fundus"]):
        required_tasks.append("region_classification")
    
    # Medical QA keywords
    if any(kw in query_lower for kw in ["?", "what", "how", "why", "explain", "tại sao", "như thế nào", "giải thích", "tư vấn"]):
        required_tasks.append("medical_qa")
    
    # Default to comprehensive if nothing specific
    if not required_tasks:
        required_tasks = ["polyp_detection", "modality_classification", "region_classification"]
    
    logger.info(f"Keyword-based analysis: {required_tasks}")
    return required_tasks


def _determine_execution_order(required_tasks: List[str]) -> List[str]:
    """Determine optimal execution order based on dependencies."""
    
    # Priority order (dependencies considered)
    priority_order = [
        "polyp_detection",        # 1. Always first (provides context for others)
        "modality_classification", # 2. Technical analysis
        "region_classification",   # 3. Anatomical analysis
        "medical_qa"              # 4. Always last (synthesis/explanation)
    ]
    
    # Sort required tasks by priority
    execution_order = []
    for task in priority_order:
        if task in required_tasks:
            execution_order.append(task)
    
    return execution_order


# Task Progress Tracker
def _mark_task_completed(state: SystemState, completed_task: str) -> SystemState:
    """Mark a task as completed - FIXED VERSION"""
    
    # CRITICAL: Don't create new dict, modify existing state
    completed_tasks = list(state.get("completed_tasks", []))
    execution_order = state.get("execution_order", [])
    
    # Add to completed if not already there
    if completed_task not in completed_tasks:
        completed_tasks.append(completed_task)
        print(f"🔧 DEBUG: Marked '{completed_task}' as completed. List: {completed_tasks}")
    
    # Find next task
    current_task = None
    for task in execution_order:
        if task not in completed_tasks:
            current_task = task
            break
    
    # CRITICAL: Update the state object directly
    state["completed_tasks"] = completed_tasks
    state["current_task"] = current_task
    
    print(f"🔧 DEBUG: Updated state - completed: {completed_tasks}, current: {current_task}")
    return state

# Result Synthesizer
def result_synthesizer(state: SystemState, llm: ChatOpenAI) -> SystemState:
    """Synthesize results from all agents into a final answer."""
    import time
    logger = logging.getLogger("med.ai.synthesizer")
    
    task_type = state.get("task_type", "unknown")
    logger.info(f"Synthesizing results for task_type: {task_type}")
    
    # Get all agent results
    agent_results = {}
    for key, val in state.items():
        if key.endswith("_result") and isinstance(val, dict):
            agent_results[key] = val

    # Identify required and completed tasks
    required_tasks = []
    completed_tasks = []
    
    if task_type == "polyp_detection":
        required_tasks = ["detector"]
        if "detector_result" in agent_results:
            if agent_results["detector_result"].get("success", False):
                completed_tasks.append("detector")
    
    elif task_type == "medical_vqa":
        required_tasks = ["vqa"]
        if "vqa_result" in agent_results:
            if agent_results["vqa_result"].get("success", False):
                completed_tasks.append("vqa")
    
    elif task_type == "comprehensive":
        required_tasks = ["detector", "modality", "region", "vqa"]
        for task in required_tasks:
            result_key = f"{task}_result"
            if result_key in agent_results:
                if agent_results[result_key].get("success", False):
                    completed_tasks.append(task)
    
    # Build final result
    final_result = {
        "task_type": task_type,
        "success": True,
        "session_id": state.get("session_id", ""),
        "query": state.get("query", ""),
        "timestamp": time.time(),
        "multi_task_analysis": {
            "required_tasks": required_tasks,
            "completed_tasks": completed_tasks,
            "execution_order": state.get("execution_order", []),
            "task_completion_rate": len(completed_tasks) / len(required_tasks) if required_tasks else 1.0
        },
        "agent_results": agent_results,
        "processing_time": time.time() - state.get("start_time", time.time())
    }
    
    # ALWAYS synthesize with LLM
    prompt = f"""You are a medical AI assistant providing analysis of an endoscopic image. The user's query is: "{state.get('query', '')}"

Below are the structured results from my analysis. Synthesize these into a natural, conversational response.

I've analyzed the following aspects of the image:
"""
    for key, val in agent_results.items():
        task_name = key.replace('_result', '').title()
        prompt += f"--- {task_name} Analysis ---\n"
        # Only include relevant fields from each result
        important_fields = ['class_name', 'confidence', 'description', 'count', 'objects', 'answer', 'analysis']
        for k in important_fields:
            if k in val and val[k] is not None and val[k] != "" and k != "visualization_base64":
                # Format confidence as percentage
                if k == 'confidence' and isinstance(val[k], (int, float)):
                    prompt += f"{k}: {val[k]:.1%}\n"
                # Format count more naturally
                elif k == 'count':
                    prompt += f"Found {val[k]} polyp(s) or abnormalities\n"
                # Format objects in a more natural way
                elif k == 'objects' and isinstance(val[k], list) and len(val[k]) > 0:
                    prompt += f"Detected objects:\n"
                    for i, obj in enumerate(val[k][:3]):  # Limit to top 3 objects
                        conf = obj.get("confidence", 0)
                        pos = obj.get("position_description", "unknown")
                        prompt += f"  - {pos} with {conf:.1%} confidence\n"
                else:
                    prompt += f"{k}: {val[k]}\n"
        prompt += "\n"
    
    prompt += """
Please create a natural, conversational response that:
1. Addresses the user directly without mentioning "agents" or "tools"
2. Presents findings in simple medical terms a patient would understand
3. Provides a cohesive narrative that integrates all the analysis components
4. Avoids technical jargon about confidence scores or agent names
5. Makes recommendations based on the findings if appropriate
6. Sounds like a unified AI assistant rather than a collection of separate analyses

Response should be in Vietnamese or English matching the user's query language.
"""
    
    # Initialize empty answer string
    final_answer_text = ""
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful, conversational medical AI assistant. You communicate clearly and naturally, as a unified AI system that provides cohesive medical analysis. You avoid technical jargon about confidence scores or agents and present findings in simple terms a patient would understand."},
            {"role": "user", "content": prompt}
        ]
        
        # Stream the answer from the LLM
        print(f"🔧 DEBUG: Starting LLM streaming")
        
        # Create generator config for streaming
        streaming_llm = llm.with_config({"streaming": True})
        
        # Process streaming chunks
        full_response = ""
        for chunk in streaming_llm.stream(messages):
            chunk_text = chunk.content
            full_response += chunk_text
            
            # Display incremental updates
            print(f"🔧 DEBUG: LLM chunk: {chunk_text}")
            
        # Use the complete streamed response
        answer = full_response.strip()
        print(f"🔧 DEBUG: Final LLM answer: {answer[:100]}...")
        
        if answer and len(answer) > 10:
            # Remove the heading for a more natural conversational flow
            final_result["final_answer"] = answer
            # Also store the raw response for streaming access
            final_result["final_answer_raw"] = answer
            final_result["streaming_enabled"] = True
        else:
            final_result["final_answer"] = _generate_multi_task_answer(final_result, llm)
            final_result["streaming_enabled"] = False
    except Exception as e:
        logger.error(f"LLM synthesis failed: {str(e)}")
        final_result["final_answer"] = _generate_multi_task_answer(final_result, llm)
        final_result["streaming_enabled"] = False
        
    print(f"🔧 DEBUG: Final answer: {final_result.get('final_answer', '')[:200]}...")
    logger.info("Synthesis complete with streaming capability")
    return {**state, "final_result": final_result}

def _generate_multi_task_answer(result: Dict[str, Any], llm: ChatOpenAI) -> str:
    """Generate a natural, conversational fallback answer."""
    agent_results = result["agent_results"]
    query = result.get("query", "")
    
    # Gather information from different results
    findings = []
    
    # Detection information
    if "detector_result" in agent_results:
        det = agent_results["detector_result"]
        if det.get("success"):
            count = det.get("count", 0)
            if count > 0:
                objects = det.get("objects", [])
                if objects:
                    # Just gather information, don't format yet
                    for obj in objects[:3]:  # Limit to top 3
                        findings.append({
                            "type": "polyp",
                            "confidence": obj.get("confidence", 0),
                            "position": obj.get("position_description", "")
                        })
            else:
                findings.append({"type": "no_polyps"})
    
    # Modality information
    if "modality_result" in agent_results:
        mod = agent_results["modality_result"]
        if mod.get("success"):
            modality = mod.get("class_name", "Unknown")
            if modality != "Unknown":
                findings.append({
                    "type": "modality",
                    "name": modality,
                    "description": mod.get("description", "")
                })
    
    # Region information
    if "region_result" in agent_results:
        reg = agent_results["region_result"]
        if reg.get("success"):
            region = reg.get("class_name", "Unknown")
            if region != "Unknown":
                findings.append({
                    "type": "region",
                    "name": region,
                    "description": reg.get("description", "")
                })
    
    # VQA information (direct answers)
    vqa_answer = ""
    if "vqa_result" in agent_results:
        vqa = agent_results["vqa_result"]
        if vqa.get("success"):
            vqa_answer = vqa.get("answer", "")
    
    # Now create a natural language response
    
    # If we have a VQA answer, use it as the primary response
    if vqa_answer:
        return vqa_answer
    
    # Otherwise, create a response from findings
    response_parts = []
    
    # Handle polyp findings
    polyp_findings = [f for f in findings if f.get("type") == "polyp"]
    if polyp_findings:
        if len(polyp_findings) == 1:
            p = polyp_findings[0]
            response_parts.append(f"Tôi đã phát hiện một polyp ở vị trí {p['position']}.")
        else:
            response_parts.append(f"Tôi đã phát hiện {len(polyp_findings)} polyp trong hình ảnh.")
            for i, p in enumerate(polyp_findings[:2], 1):
                response_parts.append(f"- Polyp {i} ở vị trí {p['position']}")
    elif any(f.get("type") == "no_polyps" for f in findings):
        response_parts.append("Tôi không phát hiện polyp nào trong hình ảnh này.")
    
    # Handle modality and region in one sentence if available
    modality = next((f for f in findings if f.get("type") == "modality"), None)
    region = next((f for f in findings if f.get("type") == "region"), None)
    
    if modality and region:
        response_parts.append(f"Đây là hình ảnh nội soi loại {modality['name']} của vùng {region['name']}.")
    elif modality:
        response_parts.append(f"Đây là hình ảnh nội soi loại {modality['name']}.")
    elif region:
        response_parts.append(f"Hình ảnh cho thấy vùng {region['name']}.")
    
    # If we have no findings to report, provide a generic response
    if not response_parts:
        response_parts.append("Tôi đã phân tích hình ảnh nội soi của bạn nhưng không thể đưa ra kết luận cụ thể. Vui lòng thử lại với hình ảnh rõ ràng hơn hoặc cung cấp thêm thông tin.")
    
    # If this is a response to a specific query, add that context
    if query:
        response_parts.append(f"Về câu hỏi của bạn: '{query}', tôi chưa thể đưa ra câu trả lời cụ thể. Vui lòng cung cấp thêm thông tin hoặc đặt câu hỏi rõ ràng hơn.")
    
    return " ".join(response_parts)