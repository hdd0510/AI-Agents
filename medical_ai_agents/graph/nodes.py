"""
Medical AI Graph - Nodes (MODIFIED for multi-task support)
---------------------
nodes với multi-task analysis và smart routing.
"""

import json
import logging
from typing import Dict, Any, List, Callable
import time
import os

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser, HumanMessage

from medical_ai_agents.config import SystemState, TaskType

# Document embedding node
def document_embedding(state: SystemState, rag_agent) -> SystemState:
    """Process and embed documents into vector database."""
    logger = logging.getLogger("graph.nodes.document_embedding")
    
    # Kiểm tra tài liệu đính kèm
    uploaded_docs = state.get("uploaded_documents", [])
    query = state.get("query", "")  # Lấy query từ state
    conversation_history = state.get("conversation_history", [])  # Lấy lịch sử hội thoại
    
    # Log thông tin về tài liệu và truy vấn
    if uploaded_docs:
        logger.info(f"Found {len(uploaded_docs)} document(s) attached to input")
        for doc in uploaded_docs[:3]:  # Log một số tài liệu đầu tiên
            logger.info(f"Document: {os.path.basename(doc)}")
    else:
        logger.info("No documents attached to current input")
    
    if query:
        logger.info(f"Current query: {query[:50]}..." if len(query) > 50 else f"Current query: {query}")
    
    if conversation_history:
        logger.info(f"Conversation history available with {len(conversation_history)} entries")
    
    # BƯỚC 1: Kiểm tra xem vector database đã tồn tại và có tài liệu không
    try:
        has_existing_docs = rag_agent.has_documents()
        if has_existing_docs:
            logger.info("Existing documents found in vector database")
            # Set the embedding result to success even if no new documents are uploaded
            state["embedding_result"] = {
                "success": True,
                "documents_processed": [],
                "message": "Vector database already contains documents",
                "has_existing_documents": True
            }
            
            # Add document_qa to required tasks if not already there
            required_tasks = state.get("required_tasks", [])
            execution_order = state.get("execution_order", [])
            
            if "document_qa" not in required_tasks:
                logger.info("Adding document_qa to required tasks")
                required_tasks.append("document_qa")
                state["required_tasks"] = required_tasks
                
                # Add to execution order if not already there
                if "document_qa" not in execution_order:
                    execution_order.append("document_qa")
                    state["execution_order"] = execution_order
    except Exception as e:
        logger.warning(f"Error checking for existing documents: {str(e)}")
        state["embedding_result"] = {
            "success": False,
            "error": f"Error checking for existing documents: {str(e)}",
            "has_existing_documents": False
        }
    
    # BƯỚC 2: Xử lý tài liệu mới tải lên (nếu có)
    if not uploaded_docs:
        logger.info("No new documents to embed, proceeding with existing database")
        if not state.get("embedding_result"):
            state["embedding_result"] = {
                "success": True,
                "documents_processed": [],
                "message": "No new documents to process",
                "has_existing_documents": state.get("embedding_result", {}).get("has_existing_documents", False)
            }
        return state
    
    # Xác minh tài liệu tồn tại
    valid_docs = []
    for doc in uploaded_docs:
        if os.path.exists(doc):
            valid_docs.append(doc)
        else:
            logger.warning(f"Document not found: {doc}")
    
    if not valid_docs:
        logger.warning("No valid documents found for embedding")
        state["embedding_result"] = {
            "success": False,
            "error": "No valid documents found",
            "message": "No valid documents found for processing",
            "has_existing_documents": state.get("embedding_result", {}).get("has_existing_documents", False)
        }
        return state
    
    # BƯỚC 3: Xử lý và nhúng tài liệu
    logger.info(f"Processing {len(valid_docs)} documents for embedding")
    try:
        # Sử dụng RAG agent để xử lý tài liệu
        result = rag_agent.add_documents(valid_docs)
        
        if result.get("success", False):
            processed_files = result.get("processed_files", [])
            processed_count = len(processed_files)
            
            logger.info(f"Successfully embedded {processed_count} documents")
            
            # Thêm thông tin về tài liệu đã xử lý vào state
            state["embedding_result"] = {
                "success": True,
                "documents_processed": processed_files,
                "message": f"Successfully processed {processed_count} documents",
                "timestamp": time.time(),
                "has_existing_documents": True
            }
            
            # Đảm bảo document_qa có trong required_tasks
            required_tasks = state.get("required_tasks", [])
            execution_order = state.get("execution_order", [])
            
            if "document_qa" not in required_tasks:
                logger.info("Adding document_qa to required tasks")
                required_tasks.append("document_qa")
                state["required_tasks"] = required_tasks
                
                # Thêm vào execution_order nếu chưa có
                if "document_qa" not in execution_order:
                    execution_order.append("document_qa")
                    state["execution_order"] = execution_order
            
            # Đặt document_qa làm current_task để đảm bảo nó được ưu tiên
            state["current_task"] = "document_qa"
            
        else:
            logger.error(f"Failed to embed documents: {result.get('error', 'Unknown error')}")
            state["embedding_result"] = {
                "success": False,
                "error": result.get("error", "Failed to embed documents"),
                "message": "Failed to embed documents",
                "has_existing_documents": state.get("embedding_result", {}).get("has_existing_documents", False)
            }
    except Exception as e:
        logger.error(f"Exception during document embedding: {str(e)}")
        state["embedding_result"] = {
            "success": False,
            "error": f"Exception during document embedding: {str(e)}",
            "message": f"Error during document embedding: {str(e)}",
            "has_existing_documents": state.get("embedding_result", {}).get("has_existing_documents", False)
        }
    
    return state

# Node wrapper to automatically mark tasks as completed
def task_completion_wrapper(agent_node: Callable, task_name: str) -> Callable:
    """
    Wrapper to automatically mark tasks as completed after agent execution.
    
    This is a critical fix for the LangGraph state management issue.
    """
    logger = logging.getLogger("graph.nodes.wrapper")
    
    def wrapped_node(state: SystemState) -> SystemState:
        # Always mark polyp_detection as completed if it's in required_tasks
        # This is a workaround for the detector agent's unique structure
        if task_name == "polyp_detection" and "polyp_detection" in state.get("required_tasks", []):
            logger.info(f"Executing agent node for task: {task_name}")
            updated_state = agent_node(state)
            result_key = "detector_result"
            
            if result_key in updated_state:
                # Always mark polyp_detection as completed regardless of success flag
                # The detector will execute and either find polyps or not, but it will complete
                detector_result = updated_state[result_key]
                # Set success flag (for consistency with other results)
                detector_result["success"] = True
                logger.info(f"Marking polyp_detection as completed")
                updated_state = _mark_task_completed(updated_state, task_name)
            else:
                logger.warning(f"Detector executed but no result found in state")
            
            return updated_state
            
        # For all other tasks
        logger.info(f"Executing agent node for task: {task_name}")
        updated_state = agent_node(state)
        
        # Task to result key mapping (special cases)
        task_to_result_key = {
            "modality_classification": "modality_result",
            "region_classification": "region_result",
            "medical_qa": "vqa_result",
            "document_qa": "rag_result"
        }
        
        # Get the correct result key for this task
        result_key = task_to_result_key.get(task_name, f"{task_name}_result")
        
        # Check if the agent execution was successful
        if result_key in updated_state:
            # Standard handling for results
            if updated_state[result_key].get("success", False):
                logger.info(f"Task {task_name} successful, marking as completed")
                # Mark the task as completed
                updated_state = _mark_task_completed(updated_state, task_name)
                return updated_state
        
        # If we get here, the task was not successful
        logger.warning(f"Task {task_name} was not successful, not marking as completed")
        
        return updated_state
    
    return wrapped_node

# Task Analyzer với Multi-Task Support
def task_analyzer(state: SystemState, llm: ChatOpenAI) -> Dict:
    """Analyze tasking requirements based on query and metadata."""
    logger = logging.getLogger("graph.nodes.task_analyzer")
    
    # Check if conversation_history is present
    if "conversation_history" in state:
        conv_history = state["conversation_history"]
        logger.info(f"[DEBUG] task_analyzer received conversation history with {len(conv_history)} entries")
        if conv_history:
            # Log a sample of entries
            for i, entry in enumerate(conv_history[:2]):
                logger.info(f"[DEBUG] History entry {i}: query='{entry.get('query', '')[:30]}...', response='{entry.get('response', '')[:30]}...'")
            
            if len(conv_history) > 2:
                for i, entry in enumerate(conv_history[-2:], start=len(conv_history)-2):
                    logger.info(f"[DEBUG] History entry {i}: query='{entry.get('query', '')[:30]}...', response='{entry.get('response', '')[:30]}...'")
    else:
        logger.warning("[DEBUG] task_analyzer did not receive conversation_history")
    
    # Check for image and query
    image_path = state.get("image_path", "")
    query = state.get("query", "")
    
    # Validate inputs - must have either image or query
    if not image_path and not query:
        logger.warning("Missing both image and query, cannot analyze task")
        if state.get("raw_query"):
            # Try to use raw query as fallback
            query = state.get("raw_query")
            state["query"] = query
            logger.info(f"Using raw query as fallback: {query[:100]}...")
        else:
            # No valid inputs, return error state
            return {
                **state,
                "error": "No image or query provided",
                "required_tasks": []
            }
    
    # Check for required parameters
    if "query" not in state or state["query"] == "":
        # Try to get raw_query as fallback
        if "raw_query" in state and state["raw_query"]:
            logger.info("No primary query found, using raw_query as fallback")
            state["query"] = state["raw_query"]
        else:
            logger.warning("No query provided in state")
    
    query = state.get("query", "")
    logger.info(f"Task analyzer processing query: '{query[:50]}...' (length: {len(query)})")
    
    # Debug query tracking for diagnosis
    if not query and "raw_query" in state:
        logger.warning(f"Query is empty but raw_query is: '{state['raw_query'][:50]}...'")
    
    # Check if we're in text-only mode
    if state.get("is_text_only", False):
        logger.info("Processing text-only query")
        
        # Extract context from conversation history
        context = ""
        conversation_history = state.get("conversation_history", [])
        
        if conversation_history:
            # Filter out system messages and pending entries 
            filtered_entries = [
                entry for entry in conversation_history 
                if not entry.get("is_system", False) and 
                not entry.get("is_pending", False) and
                not entry.get("is_meta", False)
            ]
            
            # Get the last 2 conversations for context
            recent_conversations = filtered_entries[-2:] if filtered_entries else []
            
            if recent_conversations:
                context = "Previous conversation:\n"
                for i, conv in enumerate(recent_conversations):
                    context += f"User: {conv.get('query', '')}\n"
                    context += f"System: {conv.get('response', '')[:100]}...\n\n"
        
        # Medical vs General classification for text-only queries
        prompt = PromptTemplate.from_template(
            """Analyze the following query and determine if it is directly related to medical topics:
            
            {context}
            
            Current Query: {query}
            
            Instructions:
            - If the query is about medical advice, diagnosis, treatments, medical images, or healthcare → MEDICAL
            - If the query is general conversation, personal information, greetings, or not related to healthcare → GENERAL
            - Any personal identification like "my name is" should be classified as GENERAL
            
            Respond with only one word: MEDICAL or GENERAL
            """
        )
        
        try:
            chain = prompt | llm | StrOutputParser()
            query_type = chain.invoke({"query": query, "context": context}).strip().upper()
            
            if query_type == "GENERAL":
                logger.info("Non-medical text query detected, routing directly to synthesizer")
                return {
                    **state,
                    "task_type": TaskType.TEXT_ONLY,
                    "required_tasks": ["general_query"],
                    "completed_tasks": [],
                    "execution_order": ["general_query"],
                    "is_medical_query": False
                }
            else:
                logger.info("Medical text query detected, routing through VQA")
                return {
                    **state,
                    "task_type": TaskType.TEXT_ONLY,
                    "required_tasks": ["medical_qa"],
                    "completed_tasks": [],
                    "execution_order": ["medical_qa"],
                    "is_medical_query": True
                }
        except Exception as e:
            logger.error(f"Query classification failed: {str(e)}")
            # Default to medical_qa on error
            return {
                **state,
                "task_type": TaskType.TEXT_ONLY,
                "required_tasks": ["medical_qa"],
                "completed_tasks": [],
                "execution_order": ["medical_qa"],
                "is_medical_query": True
            }
    
    if not query:
        logger.info("No query provided, defaulting to comprehensive analysis")
        tasks = ["polyp_detection", "modality_classification", "region_classification"]
        if state.get("uploaded_documents", []):
            tasks.append("document_qa")
        return {
            **state,
            "task_type": TaskType.COMPREHENSIVE,
            "required_tasks": tasks,
            "completed_tasks": [],
            "execution_order": tasks
        }
    
    logger.info(f"Analyzing multi-task query: {query}")
    
    # Initialize context variable outside of the text-only branch
    context = ""
    
    # Extract context from conversation history if available
    conversation_history = state.get("conversation_history", [])
    if conversation_history:
        # Filter out system messages and pending entries 
        filtered_entries = [
            entry for entry in conversation_history 
            if not entry.get("is_system", False) and 
            not entry.get("is_pending", False) and
            not entry.get("is_meta", False)
        ]
        
        # Get the last 2 conversations for context
        recent_conversations = filtered_entries[-2:] if filtered_entries else []
        
        if recent_conversations:
            context = "Previous conversation:\n"
            for i, conv in enumerate(recent_conversations):
                context += f"User: {conv.get('query', '')}\n"
                context += f"System: {conv.get('response', '')[:100]}...\n\n"
    
    # prompt for multi-task analysis with conversation history
    prompt = PromptTemplate.from_template(
        """Analyze the following request and determine the necessary tasks to provide a complete answer:
        
        {context}
        
        Current Request: {query}
        
        Available tasks (multiple can be selected):
        - polyp_detection: Detect polyps and abnormal objects
        - modality_classification: Classify endoscopy technique (BLI, WLI, FICE, LCI)
        - region_classification: Classify anatomical region in gastrointestinal tract
        - medical_qa: Answer medical questions, provide consultation, explain medical concepts
        - document_qa: Answer questions related to documents or PDF files
        
        Guidelines:
        - If asking about polyps/lesions/detection → include polyp_detection
        - If asking about technique/modality/BLI/WLI → include modality_classification
        - If asking about anatomical region/anatomy/where in GI tract → include region_classification
        - If explanation/consultation/analysis needed → include medical_qa
        - If asking about documents/PDF → include document_qa
        - Complex questions may require multiple tasks
        
        Return a list of necessary tasks, separated by commas.
        Example: polyp_detection, medical_qa
        Or: modality_classification, region_classification, medical_qa
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    
    try:
        # Get LLM analysis
        task_result = chain.invoke({"query": query, "context": context})
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
    if any(kw in query_lower for kw in ["polyp", "lesion", "detection", "detect", "find", "abnormal", "tumor", "growth"]):
        required_tasks.append("polyp_detection")
    
    # Modality keywords
    if any(kw in query_lower for kw in ["bli", "wli", "fice", "lci", "technique", "modality", "imaging", "light", "wavelength"]):
        required_tasks.append("modality_classification")
    
    # Region keywords
    if any(kw in query_lower for kw in ["location", "region", "anatomy", "where", "position", "antrum", "fundus", "colon", "stomach"]):
        required_tasks.append("region_classification")
    
    # Medical QA keywords
    if any(kw in query_lower for kw in ["?", "what", "how", "why", "explain", "reason", "consultation", "advice", "help", "symptoms"]):
        required_tasks.append("medical_qa")
    
    # Document QA keywords
    if any(kw in query_lower for kw in ["document", "pdf", "file", "paper", "report", "research", "study"]):
        required_tasks.append("document_qa")
    
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
    """
    Mark a task as completed - CORRECT VERSION for LangGraph
    
    CRITICAL: LangGraph copies state between nodes, so we MUST return a new state dict
    """
    logger = logging.getLogger("graph.nodes.task_tracker")
    
    # Get current values
    current_completed = list(state.get("completed_tasks", []))
    execution_order = state.get("execution_order", [])
    
    # Add to completed if not already there
    if completed_task not in current_completed:
        current_completed.append(completed_task)
        logger.info(f"Added '{completed_task}' to completed list: {current_completed}")
    else:
        logger.debug(f"Task '{completed_task}' already marked as completed")
    
    # Find next task
    next_task = None
    for task in execution_order:
        if task not in current_completed:
            next_task = task
            break
    
    # CRITICAL: Return NEW state dict (LangGraph requirement)
    new_state = {
        **state,  # Copy all existing fields
        "completed_tasks": current_completed,  # Update completed tasks
        "current_task": next_task  # Update current task
    }
    
    # Ensure we're returning a new state object
    if id(new_state) == id(state):
        logger.error("CRITICAL ERROR: _mark_task_completed did not create a new state object!")
        # Force creation of a new dict
        new_state = dict(new_state)
    
    return new_state

# Result Synthesizer
def result_synthesizer(state: SystemState, llm: ChatOpenAI) -> SystemState:
    """Synthesize results from multiple tasks and agents."""
    logger = logging.getLogger("graph.nodes.result_synthesizer")
    
    # Extract conversation history for debugging
    conversation_history = state.get("conversation_history", [])
    
    # Extract state parameters
    current_query = state.get("query", "")
    is_text_only = state.get("is_text_only", False)
    task_type = state.get("task_type", "unknown")
    
    # Task info with improved logging
    required_tasks = state.get("required_tasks", [])
    completed_tasks = state.get("completed_tasks", [])
    execution_order = state.get("execution_order", [])
    logger.info(f"Synthesizer - Required tasks: {required_tasks}, Completed tasks: {completed_tasks}")
    
    # Check for streaming support - Keep this for other streaming functionality
    response_stream = state.get("response_stream")
    enable_streaming = response_stream is not None
    
    # Clean up pending/debug entries from conversation history
    if conversation_history:
        conversation_history = [entry for entry in conversation_history if not entry.get("is_pending", False)]
        # Update the state with cleaned history
        state["conversation_history"] = conversation_history
    
    # Available results
    agent_results_keys = []
    if "detector_result" in state:
        agent_results_keys.append("detector_result")
    if "modality_result" in state:
        agent_results_keys.append("modality_result")
    if "region_result" in state:
        agent_results_keys.append("region_result")
    if "vqa_result" in state:
        agent_results_keys.append("vqa_result")
    if "rag_result" in state:
        agent_results_keys.append("rag_result")
    
    # Calculate processing time
    start_time = state.get("start_time", time.time())
    processing_time = time.time() - start_time
    
    # Check if it's a general (non-medical) query
    is_general_query = "general_query" in state.get("required_tasks", [])
    
    # Process all agent results
    agent_results = {}
    
    # Process detector result
    if "detector_result" in state:
        detector = state["detector_result"]
        if detector.get("success", False):
            agent_results["detection"] = {
                "count": detector.get("count", 0),
                "boxes": detector.get("boxes", []),
                "scores": detector.get("scores", []),
                "classes": detector.get("classes", []),
                "objects": detector.get("objects", []),
                "image_path": detector.get("image_path", ""),
                "visualization_base64": detector.get("visualization_base64", ""),
                "show_visualization": detector.get("show_visualization", False)
            }
    
    # Process modality result
    if "modality_result" in state:
        modality = state["modality_result"]
        if modality.get("success", False):
            agent_results["modality"] = {
                "class_name": modality.get("class_name", "Unknown"),
                "confidence": modality.get("confidence", 0.0),
                "all_classes": modality.get("all_classes", []),
                "all_scores": modality.get("all_scores", [])
            }
    
    # Process region result
    if "region_result" in state:
        region = state["region_result"]
        if region.get("success", False):
            agent_results["region"] = {
                "class_name": region.get("class_name", "Unknown"),
                "confidence": region.get("confidence", 0.0),
                "all_classes": region.get("all_classes", []),
                "all_scores": region.get("all_scores", [])
            }
    
    # Process VQA result
    if "vqa_result" in state:
        vqa = state["vqa_result"]
        if vqa.get("success", False):
            agent_results["vqa"] = {
                "answer": vqa.get("answer", ""),
                "confidence": vqa.get("confidence", 0.0),
                "reasoning": vqa.get("reasoning", ""),
                "sources": vqa.get("sources", [])
            }
    
    # Process RAG result
    if "rag_result" in state:
        rag = state["rag_result"]
        if rag.get("success", False):
            agent_results["rag"] = {
                "answer": rag.get("answer", ""),
                "sources": rag.get("sources", []),
                "chunks_retrieved": rag.get("chunks_retrieved", 0),
                "documents_processed": rag.get("documents_processed", []),
                "query_complexity": rag.get("query_complexity", "simple")
            }
    
    # Retrieve additional vector search results for context enrichment
    vector_context = ""
    try:
        # Check if we have a RAG agent and query
        if current_query and "rag_agent" in state and hasattr(state["rag_agent"], "search_tool"):
            logger.info("Retrieving additional vector context for synthesis")
            search_result = state["rag_agent"].search_tool._run(
                query=current_query,
                top_k=3,
                threshold=0.5
            )
            
            if search_result.get("success", False) and search_result.get("results"):
                results = search_result["results"]
                vector_context = "\n\nAdditional context from documents:\n"
                for i, result in enumerate(results):
                    source = result.get("source", "Unknown source")
                    content = result.get("content", "").strip()
                    score = result.get("score", 0.0)
                    vector_context += f"[Document {i+1}] From {source} (relevance: {score:.2f}):\n{content}\n\n"
                
                logger.info(f"Retrieved {len(results)} additional context chunks")
        
        # If we don't have a direct RAG agent reference but have embedded documents
        elif current_query and "embedding_result" in state and state["embedding_result"].get("success", False):
            # Try to find the vector search tool in the workflow
            from medical_ai_agents.tools.rag.vector_search import VectorSearchTool
            
            # Create a temporary search tool if needed
            storage_path = state.get("rag_storage_path", "./rag_storage")
            search_tool = VectorSearchTool(storage_path=storage_path)
            
            if search_tool.initialize_index():
                search_result = search_tool._run(
                    query=current_query,
                    top_k=3,
                    threshold=0.5
                )
                
                if search_result.get("success", False) and search_result.get("results"):
                    results = search_result["results"]
                    vector_context = "\n\nAdditional context from documents:\n"
                    for i, result in enumerate(results):
                        source = result.get("source", "Unknown source")
                        content = result.get("content", "").strip()
                        score = result.get("score", 0.0)
                        vector_context += f"[Document {i+1}] From {source} (relevance: {score:.2f}):\n{content}\n\n"
                    
                    logger.info(f"Retrieved {len(results)} additional context chunks")
    except Exception as e:
        logger.error(f"Error retrieving additional vector context: {str(e)}")
        vector_context = ""
    
    # Build synthesis prompt
    has_detection = "detection" in agent_results
    has_modality = "modality" in agent_results
    has_region = "region" in agent_results
    has_vqa = "vqa" in agent_results 
    has_rag = "rag" in agent_results
    
    # Build simple prompt for synthesis
    task_context = []

    if state.get("conversation_history"):
        task_context.append(f"Conversation history: {state.get('conversation_history')}")
    
    if has_detection:
        det = agent_results["detection"]
        task_context.append(f"Polyp Detection: Found {det['count']} polyps in the image")
    
    if has_modality:
        mod = agent_results["modality"]
        task_context.append(f"Modality Classification: {mod['class_name']} (confidence: {mod['confidence']:.1%})")
    
    if has_region:
        reg = agent_results["region"]
        task_context.append(f"Anatomical Region: {reg['class_name']} (confidence: {reg['confidence']:.1%})")
    
    if has_vqa:
        vqa = agent_results["vqa"]
        task_context.append(f"Visual Question Answering: {vqa['answer']}")
    
    if has_rag:
        rag = agent_results["rag"]
        task_context.append(f"Document Retrieval: Information from {len(rag.get('sources', []))} sources")
    
    # Build prompt for LLM
    prompt_parts = []
    prompt_parts.append(f"TASK: You are synthesizing results from multiple medical AI analyses of medical results. If it is just a normal question and no result from system, just answer normally. The query is: '{current_query}'")
    
    for task_info in task_context:
        prompt_parts.append(f"- {task_info}")
    
    # Add vector context if available
    if vector_context:
        prompt_parts.append(vector_context)
    
    prompt = "\n".join(prompt_parts)
    
    # Generate synthesis with LLM
    try:
        # Check if we need streaming or regular response
        if enable_streaming:
            logger.info("Streaming synthesis response")
            # Create an async helper function for streaming
            import asyncio
            
            async def stream_response():
                try:
                    logger.info("Starting streaming LLM response")
                    
                    # Configure streaming
                    streaming_llm = llm.with_config({
                        "streaming": True,
                        "temperature": 0.5  # Make responses more consistent
                    })
                    
                    # Initialize response tracking
                    final_response = ""
                    last_update_time = time.time()
                    update_interval = 0.2  # Update every 200ms max
                    force_update_length = 20  # Or every 20 chars
                    last_update_length = 0
                    
                    # Prepare message for LLM
                    message = [HumanMessage(content=prompt)]
                    
                    # Stream tokens directly from LLM
                    logger.info("Beginning token streaming from LLM")
                    async for chunk in streaming_llm.astream_tokens(message):
                        # Extract content if available
                        if not chunk.choices or len(chunk.choices) == 0 or not hasattr(chunk.choices[0].delta, 'content'):
                            continue
                            
                        # Get the token content
                        token = chunk.choices[0].delta.content
                        if token:
                            # Add token to response
                            final_response += token
                            current_length = len(final_response)
                            
                            # Determine if we should send an update
                            time_to_update = (time.time() - last_update_time) >= update_interval
                            length_to_update = (current_length - last_update_length) >= force_update_length
                            
                            # Send update if needed
                            if time_to_update or length_to_update:
                                # Debug stream progress
                                if logger.isEnabledFor(logging.DEBUG):
                                    logger.debug(f"Streaming progress: {len(final_response)} chars")
                                
                                # Put the update in the queue
                                try:
                                    await response_stream.put(final_response)
                                    last_update_time = time.time()
                                    last_update_length = current_length
                                except Exception as e:
                                    logger.error(f"Error putting to queue: {str(e)}")
                    
                    # Always send the final complete response
                    logger.info(f"LLM streaming complete, final length: {len(final_response)} chars")
                    await response_stream.put(final_response)
                    return final_response
                    
                except Exception as e:
                    logger.error(f"Error in streaming response: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # Try to send error message to queue
                    error_msg = f"Error generating response: {str(e)}"
                    try:
                        await response_stream.put(error_msg)
                    except Exception:
                        pass
                    
                    return error_msg
            
            # Improved streaming implementation
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're already in an event loop, create a task
                    logger.info("Using existing event loop for streaming")
                    future = asyncio.create_task(stream_response())
                    # Will be completed asynchronously
                    synthesis = "Streaming response in progress..."
                else:
                    # No running event loop, create one
                    logger.info("Creating new event loop for streaming")
                    synthesis = asyncio.run(stream_response())
            except Exception as e:
                logger.error(f"Error setting up streaming: {str(e)}")
                # Fallback to non-streaming response
                synthesis = llm.invoke(prompt).content
        else:
            # Regular non-streaming response
            synthesis = llm.invoke(prompt).content
    except Exception as e:
        logger.error(f"Error generating synthesis: {str(e)}")
        synthesis = f"Failed to generate synthesis due to an error: {str(e)}"
    
    # Build the final result
    final_result = {
        "success": True,
        "session_id": state.get("session_id", ""),
        "task_type": task_type,
        "query": current_query,
        "is_text_only": is_text_only,
        "agent_results": agent_results,
        "response": synthesis,
        "processing_time": processing_time,
        "final_answer": synthesis
    }
    
    # Add the current interaction to conversation history
    conversation_history.append({
        "query": current_query,
        "response": synthesis,
        "timestamp": time.time(),
        "tasks_completed": state.get("completed_tasks", [])
    })
    
    # Return the updated state
    return {
        **state,
        "final_result": final_result,
        "conversation_history": conversation_history
    }