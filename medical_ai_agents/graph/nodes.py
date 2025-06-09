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
    uploaded_docs = state.get("uploaded_documents", [])
    
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
        tasks = ["polyp_detection", "modality_classification", "region_classification"]
        if uploaded_docs:
            tasks.append("document_qa")
        return {
            **state,
            "task_type": TaskType.COMPREHENSIVE,
            "required_tasks": tasks,
            "completed_tasks": [],
            "execution_order": tasks
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
        - document_qa: Trả lời câu hỏi liên quan đến tài liệu, tài liệu PDF
        
        Hướng dẫn:
        - Nếu hỏi về polyp/tổn thương/phát hiện → bao gồm polyp_detection
        - Nếu hỏi về kỹ thuật/modality/BLI/WLI → bao gồm modality_classification
        - Nếu hỏi về vị trí/anatomy/region → bao gồm region_classification
        - Nếu cần giải thích/tư vấn/phân tích → bao gồm medical_qa
        - Nếu hỏi về tài liệu/PDF → bao gồm document_qa
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
        "region_classification", "medical_qa", "document_qa"
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
        elif "qa" in task or "question" in task:
            if "document" in task or "pdf" in task or "file" in task:
                parsed_tasks.append("document_qa")
            else:
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
        "document_qa",            # 4. Document analysis
        "medical_qa"              # 5. Always last (synthesis/explanation)
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
    """Tổng hợp kết quả từ nhiều agent thành phản hồi cuối cùng."""
    logger = logging.getLogger("graph.nodes.result_synthesizer")
    
    # Calculate processing time
    start_time = state.get("start_time", time.time())
    processing_time = time.time() - start_time
    
    # Prepare agent results
    agent_results = {}
    
    # Add detector results
    if "detector_result" in state:
        agent_results["detector"] = state["detector_result"]
    
    # Add classifier results
    if "modality_result" in state:
        agent_results["modality"] = state["modality_result"]
    
    if "region_result" in state:
        agent_results["region"] = state["region_result"]
    
    # Add VQA results
    if "vqa_result" in state:
        agent_results["vqa"] = state["vqa_result"]
    
    # Add RAG results
    if "rag_result" in state:
        agent_results["rag"] = state["rag_result"]
    
    # Build LLM prompt based on available results
    has_detection = "detector" in agent_results
    has_modality = "modality" in agent_results
    has_region = "region" in agent_results
    has_vqa = "vqa" in agent_results 
    has_rag = "rag" in agent_results
    
    # Build simple prompt for synthesis
    task_context = []
    
    if has_detection:
        task_context.append("polyp detection")
    if has_modality:
        task_context.append("modality classification")
    if has_region:
        task_context.append("anatomical region classification")
    if has_rag:
        task_context.append("document analysis")
    if has_vqa:
        task_context.append("medical question answering")
    
    tasks_str = ", ".join(task_context)
    
    # Determine if the response should prioritize RAG or combined results
    prioritize_rag = has_rag and "rag" in agent_results and agent_results["rag"].get("query_complexity", "simple") == "simple"
    
    # Set up different prompts based on the scenario
    if prioritize_rag:
        prompt_template = """You are a medical AI assistant synthesizing results from document analysis. 
        
The user asked: "{query}"

The document analysis yielded the following information:
{rag_answer}

Sources cited:
{rag_sources}

Your task:
1. Respond to the user's query directly using the document analysis results
2. Maintain all citations and references to documents
3. Format your response in a clear, professional manner
4. Do not add medical disclaimers or warnings"""
    else:
        prompt_template = """You are a medical AI assistant synthesizing results from multiple analysis tasks including {tasks}.
        
The user asked: "{query}"

Combined analysis results:
{combined_results}

Your task:
1. Synthesize these results into a comprehensive, cohesive response
2. Directly address the user's query with relevant findings
3. Format your response in a clear, professional manner
4. IMPORTANT: If any classifications show LOW CONFIDENCE, clearly mention this uncertainty and explain the LLM's analysis
5. Do not add medical disclaimers or warnings"""
    
    # Format the prompt arguments
    prompt_args = {
        "query": state.get("query", ""),
        "tasks": tasks_str,
    }
    
    # Add RAG-specific information if prioritizing RAG
    if prioritize_rag and has_rag:
        rag_result = agent_results["rag"]
        rag_answer = rag_result.get("answer", "")
        
        # Format sources
        sources = rag_result.get("sources", [])
        sources_text = ""
        for i, source in enumerate(sources):
            sources_text += f"{i+1}. Document: {source.get('document', 'Unknown')}, Page: {source.get('page', '?')}\n"
        
        prompt_args["rag_answer"] = rag_answer
        prompt_args["rag_sources"] = sources_text
    
    # Default to combined results
    combined_results = ""
    
    # Add detailed results for different agents
    if has_detection:
        detector = agent_results["detector"]
        if detector.get("success", False):
            combined_results += f"Polyp Detection: {detector.get('count', 0)} polyp(s) found\n"
            if detector.get("boxes", []):
                combined_results += f"Locations: {len(detector.get('boxes', []))} location(s) identified\n"
    
    if has_modality:
        modality = agent_results["modality"]
        if modality.get("success", False):
            confidence = modality.get("confidence", 0.0)
            class_name = modality.get("class_name", "Unknown")
            
            # Handle low confidence results
            if modality.get("is_low_confidence", False):
                combined_results += f"Imaging Modality: {class_name} (LOW CONFIDENCE: {confidence:.1%})\n"
                combined_results += f"LLM Analysis of Modality: {modality.get('analysis', '')[:300]}...\n"
            else:
                combined_results += f"Imaging Modality: {class_name} ({confidence:.1%} confidence)\n"
            
    if has_region:
        region = agent_results["region"]
        if region.get("success", False):
            confidence = region.get("confidence", 0.0)
            class_name = region.get("class_name", "Unknown")
            
            # Handle low confidence results
            if region.get("is_low_confidence", False):
                combined_results += f"Anatomical Region: {class_name} (LOW CONFIDENCE: {confidence:.1%})\n"
                combined_results += f"LLM Analysis of Region: {region.get('analysis', '')[:300]}...\n"
            else:
                combined_results += f"Anatomical Region: {class_name} ({confidence:.1%} confidence)\n"
    if has_rag:
        rag = agent_results["rag"]
        if rag.get("success", False):
            combined_results += f"Document Analysis: {len(rag.get('documents_processed', []))} document(s) processed\n"
            combined_results += f"Found {rag.get('chunks_retrieved', 0)} relevant passages\n"
            combined_results += f"Document Answer: {rag.get('answer', 'No clear answer found')}\n"
    
    if has_vqa:
        vqa = agent_results["vqa"]
        if vqa.get("success", False):
            combined_results += f"Medical Analysis: {vqa.get('answer', 'No clear answer provided')}\n"
    
    prompt_args["combined_results"] = combined_results
    
    # Build the prompt with the appropriate template
    prompt = PromptTemplate.from_template(
        prompt_template 
    )
    
    # Create the chain
    chain = prompt | llm | StrOutputParser()
    
    # Invoke the chain
    try:
        synthesized_response = chain.invoke(prompt_args)
        
        # Build the final result
        final_result = {
            "task_type": state.get("task_type", "comprehensive"),
            "success": True,
            "session_id": state.get("session_id", ""),
            "query": state.get("query", ""),
            "timestamp": time.time(),
            "multi_task_analysis": {
                "tasks_requested": state.get("required_tasks", []),
                "tasks_completed": state.get("completed_tasks", []),
                "execution_order": state.get("execution_order", [])
            },
            "agent_results": agent_results,
            "response": synthesized_response,
            "processing_time": processing_time
        }
        
        return {**state, "final_result": final_result}
        
    except Exception as e:
        logger.error(f"Result synthesis failed: {str(e)}")
        
        # Fallback to direct response
        if has_vqa:
            fallback_response = agent_results["vqa"].get("answer", "")
        elif has_rag:
            fallback_response = agent_results["rag"].get("answer", "")
        else:
            fallback_response = "I couldn't synthesize a comprehensive answer due to an error."
        
        final_result = {
            "task_type": state.get("task_type", "comprehensive"),
            "success": False,
            "error": str(e),
            "session_id": state.get("session_id", ""),
            "query": state.get("query", ""),
            "timestamp": time.time(),
            "response": fallback_response,
            "processing_time": processing_time
        }
        
        return {**state, "final_result": final_result}