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
    """Result synthesizer with LLM-based natural language synthesis for all agent results"""
    logger = logging.getLogger("graph.nodes.synthesizer")
    
    required_tasks = state.get("required_tasks", [])
    completed_tasks = state.get("completed_tasks", [])
    task_type = state.get("task_type", TaskType.COMPREHENSIVE)
    
    print(f"🔧 DEBUG: Synthesizer - Required: {required_tasks}, Completed: {completed_tasks}")
    print(f"🔧 DEBUG: State keys: {list(state.keys())}")
    
    # Collect agent results
    agent_results = {}
    if "vqa_result" in state:
        vqa_result = state["vqa_result"]
        if isinstance(vqa_result, dict) and vqa_result.get("success", False):
            agent_results["vqa_result"] = vqa_result
            if "medical_qa" not in completed_tasks:
                completed_tasks = list(completed_tasks)
                completed_tasks.append("medical_qa")
                state["completed_tasks"] = completed_tasks
    for result_key in ["detector_result", "modality_result", "region_result"]:
        if result_key in state:
            result_data = state[result_key]
            if isinstance(result_data, dict) and result_data.get("success", False):
                agent_results[result_key] = result_data
    print(f"🔧 DEBUG: Final agent_results keys: {list(agent_results.keys())}")
    
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
    
    # 1. If VQA exists, keep current logic
    if agent_results.get("vqa_result"):
        vqa_result = agent_results["vqa_result"]
        vqa_answer = vqa_result.get("answer", "")
        if vqa_answer and len(vqa_answer.strip()) > 5:
            print(f"🔧 DEBUG: Using VQA answer as final_answer: {vqa_answer[:100]}...")
            if task_type == TaskType.TEXT_ONLY:
                final_result["final_answer"] = vqa_answer
            else:
                final_result["final_answer"] = f"🏥 **Medical Analysis:**\n{vqa_answer}\n\n⚠️ **Medical Disclaimer:** This AI analysis is for informational purposes. Please consult healthcare professionals for medical decisions."
        else:
            print(f"🔧 DEBUG: VQA answer empty, using disclaimer")
            final_result["final_answer"] = "⚠️ **Medical Disclaimer:** This AI analysis is for informational purposes. Please consult healthcare professionals for medical decisions."
    else:
        # 2. Prefer agent-level LLM synthesis if available
        natural_answers = []
        for key, val in agent_results.items():
            if isinstance(val, dict):
                ans = val.get("analysis") or val.get("answer")
                if ans and len(ans.strip()) > 10:
                    natural_answers.append(f"--- {key.replace('_result','').title()} ---\n{ans}")
        if natural_answers:
            final_result["final_answer"] = "\n\n".join(natural_answers) + "\n\n⚠️ **Medical Disclaimer:** This AI analysis is for informational purposes. Please consult healthcare professionals for medical decisions."
        else:
            # 3. LLM synthesis for all agent results (as before)
            print(f"🔧 DEBUG: No agent-level analysis, using LLM to synthesize all agent results")
            prompt = """You are a medical AI assistant. Synthesize a comprehensive, natural language answer for the user based on the following structured results from multiple medical analysis agents. Explain findings clearly, mention confidence, and provide clinical recommendations if possible. Always include a medical disclaimer at the end.\n\n"""
            for key, val in agent_results.items():
                prompt += f"--- {key.replace('_result','').title()} ---\n"
                for k, v in val.items():
                    if k == "visualization_base64":
                        continue
                    prompt += f"{k}: {v}\n"
                prompt += "\n"
            prompt += "\nSynthesize the above into a single, clear, professional answer for the user.\n"
            try:
                messages = [
                    {"role": "system", "content": "You are a medical AI assistant."},
                    {"role": "user", "content": prompt}
                ]
                answer = llm.invoke(messages).content.strip()
                if answer and len(answer) > 10:
                    final_result["final_answer"] = f"🏥 **Medical Analysis:**\n{answer}\n\n⚠️ **Medical Disclaimer:** This AI analysis is for informational purposes. Please consult healthcare professionals for medical decisions."
                else:
                    final_result["final_answer"] = _generate_multi_task_answer(final_result, llm)
            except Exception as e:
                logger.error(f"LLM synthesis failed: {str(e)}")
                final_result["final_answer"] = _generate_multi_task_answer(final_result, llm)
    print(f"🔧 DEBUG: Final answer: {final_result.get('final_answer', '')[:200]}...")
    logger.info("Synthesis complete with enhanced logic")
    return {**state, "final_result": final_result}

def _generate_multi_task_answer(result: Dict[str, Any], llm: ChatOpenAI) -> str:
    """Generate answer highlighting multi-task execution."""
    agent_results = result["agent_results"]
    multi_task_info = result["multi_task_analysis"]
    required_tasks = multi_task_info["required_tasks"]
    
    # Build task-specific sections
    answer_sections = []
    
    # Detection section
    if "detector_result" in agent_results:
        det = agent_results["detector_result"]
        if det.get("success"):
            count = det.get("count", 0)
            if count > 0:
                answer_sections.append(f"🔍 **Polyp Detection:** Found {count} polyp(s) in the image")
                # Add confidence details
                objects = det.get("objects", [])
                if objects:
                    for i, obj in enumerate(objects[:2]):
                        conf = obj.get("confidence", 0)
                        pos = obj.get("position_description", "unknown")
                        answer_sections.append(f"   - Polyp {i+1}: {conf:.1%} confidence, {pos}")
            else:
                answer_sections.append("🔍 **Polyp Detection:** No polyps detected")
    
    # Modality section
    if "modality_result" in agent_results:
        mod = agent_results["modality_result"]
        if mod.get("success"):
            modality = mod.get("class_name", "Unknown")
            confidence = mod.get("confidence", 0)
            answer_sections.append(f"📸 **Imaging Modality:** {modality} ({confidence:.1%} confidence)")
    
    # Region section
    if "region_result" in agent_results:
        reg = agent_results["region_result"]
        if reg.get("success"):
            region = reg.get("class_name", "Unknown")
            confidence = reg.get("confidence", 0)
            answer_sections.append(f"📍 **Anatomical Region:** {region} ({confidence:.1%} confidence)")
    
    # Medical QA section
    if "vqa_result" in agent_results:
        vqa = agent_results["vqa_result"]
        if vqa.get("success"):
            vqa_answer = vqa.get("answer", "")
            if vqa_answer:
                answer_sections.append(f"🏥 **Medical Analysis:**\n{vqa_answer}")
    
    # Multi-task summary
    completed_count = len(multi_task_info["completed_tasks"])
    required_count = len(required_tasks)
    completion_rate = multi_task_info["task_completion_rate"]
    
    if len(answer_sections) > 1:
        answer_sections.append(f"\n📊 **Multi-Task Summary:** Completed {completed_count}/{required_count} tasks ({completion_rate:.1%})")
        answer_sections.append(f"**Tasks executed:** {', '.join(multi_task_info['completed_tasks'])}")
    
    # Medical disclaimer
    answer_sections.append("\n⚠️ **Medical Disclaimer:** This AI analysis is for informational purposes. Please consult healthcare professionals for medical decisions.")
    
    return "\n".join(answer_sections)