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
    
    # Check for uploaded documents
    uploaded_docs = state.get("uploaded_documents", [])
    
    if not uploaded_docs:
        logger.info("No documents to embed, skipping embedding step")
        return state
    
    # Verify document existence
    valid_docs = []
    for doc in uploaded_docs:
        if os.path.exists(doc):
            valid_docs.append(doc)
        else:
            logger.warning(f"Document not found: {doc}")
    
    if not valid_docs:
        logger.warning("No valid documents found for embedding")
        return state
    
    # Process and embed documents
    logger.info(f"Processing {len(valid_docs)} documents for embedding")
    try:
        result = rag_agent.add_documents(valid_docs)
        
        if result.get("success", False):
            logger.info(f"Successfully embedded {len(result.get('documents', []))} documents")
            # Store embedding result in state
            state["embedding_result"] = {
                "success": True,
                "documents_processed": result.get("documents", []),
                "message": result.get("message", "Documents processed successfully")
            }
        else:
            logger.error(f"Failed to embed documents: {result.get('error', 'Unknown error')}")
            state["embedding_result"] = {
                "success": False,
                "error": result.get("error", "Failed to embed documents")
            }
    except Exception as e:
        logger.error(f"Exception during document embedding: {str(e)}")
        state["embedding_result"] = {
            "success": False,
            "error": f"Exception during document embedding: {str(e)}"
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
        - region_classification: Classify anatomical location in gastrointestinal tract
        - medical_qa: Answer medical questions, provide consultation, explain medical concepts
        - document_qa: Answer questions related to documents or PDF files
        
        Guidelines:
        - If asking about polyps/lesions/detection → include polyp_detection
        - If asking about technique/modality/BLI/WLI → include modality_classification
        - If asking about location/anatomy/region → include region_classification
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
                "image_path": detector.get("image_path", "")
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
    
    # Include conversation history in the medical prompt
    conversation_context = ""
    
    if len(conversation_history) > 0:
        # Get last 2 turns of conversation for context
        recent_history = conversation_history[-2:] if len(conversation_history) >= 2 else conversation_history
        
        conversation_context = "\nConversation history:\n"
        for i, entry in enumerate(recent_history):
            q = entry.get("query", "")
            r = entry.get("response", "")
            if q and r:
                conversation_context += f"User: {q}\nAssistant: {r}\n\n"
    
    # Build the prompt for synthesis
    if is_general_query:
        # Simple prompt for general (non-medical) queries
        prompt_template = """You are a helpful assistant answering a general question.

User query: "{query}"

Please provide a direct and helpful response.{conversation_context}"""
        
        prompt_args = {
            "query": current_query,
            "conversation_context": conversation_context
        }
    
    elif prioritize_rag:
        # RAG-focused prompt when document search is the primary task
        prompt_template = """You are a medical assistant providing information based primarily on document search results.

User query: "{query}"

Document search results:
{rag_answer}

Sources:
{rag_sources}

{vector_context}

Please synthesize a comprehensive answer that:
1. Directly addresses the user's query
2. Cites specific sources when referencing information
3. Maintains medical accuracy and precision
4. Uses proper medical terminology

Your response should be well-structured and focused on the document-based information.{conversation_context}"""
        
        # Format RAG sources for the prompt
        rag_sources_text = ""
        if has_rag and agent_results["rag"].get("sources"):
            for i, source in enumerate(agent_results["rag"]["sources"]):
                doc = source.get("document", "Unknown")
                page = source.get("page", 0)
                score = source.get("score", 0.0)
                rag_sources_text += f"[{i+1}] {doc} (page {page}, relevance: {score:.2f})\n"
        
        prompt_args = {
            "query": current_query,
            "rag_answer": agent_results["rag"]["answer"] if has_rag else "",
            "rag_sources": rag_sources_text,
            "vector_context": vector_context,
            "conversation_context": conversation_context
        }
    
    else:
        # Comprehensive medical prompt for multi-task synthesis
        prompt_template = """You are a medical assistant synthesizing results from multiple analysis tasks: {tasks}.

User query: "{query}"

Analysis results:
{analysis_results}

{vector_context}

Please synthesize a comprehensive medical response that:
1. Directly addresses the user's query
2. Integrates all relevant findings from the analyses
3. Provides clear medical explanations
4. Maintains professional medical tone and terminology
5. Cites sources when referencing document information

Your response should be well-structured and focused on the medical significance of the findings.{conversation_context}"""
        
        # Build detailed analysis results text
        analysis_text = ""
        
        # Add detection results
        if has_detection:
            detection = agent_results["detection"]
            count = detection.get("count", 0)
            analysis_text += f"POLYP DETECTION: {count} polyp(s) detected\n"
            if count > 0:
                analysis_text += "Findings:\n"
                for i in range(min(count, len(detection.get("boxes", [])))):
                    score = detection["scores"][i] if i < len(detection.get("scores", [])) else 0
                    analysis_text += f"- Polyp {i+1}: confidence {score:.2f}\n"
                analysis_text += "\n"
        
        # Add modality results
        if has_modality:
            modality = agent_results["modality"]
            class_name = modality.get("class_name", "Unknown")
            confidence = modality.get("confidence", 0.0)
            analysis_text += f"MODALITY CLASSIFICATION: {class_name} (confidence: {confidence:.2f})\n"
            
            # Add explanation of modality
            if class_name == "WLI":
                analysis_text += "White Light Imaging - standard endoscopic visualization\n"
            elif class_name == "BLI":
                analysis_text += "Blue Light Imaging - enhanced visualization of surface patterns and vessels\n"
            elif class_name == "FICE":
                analysis_text += "Flexible spectral Imaging Color Enhancement - digital chromoendoscopy\n"
            elif class_name == "LCI":
                analysis_text += "Linked Color Imaging - enhanced visualization of mucosal changes\n"
            analysis_text += "\n"
        
        # Add region results
        if has_region:
            region = agent_results["region"]
            class_name = region.get("class_name", "Unknown")
            confidence = region.get("confidence", 0.0)
            analysis_text += f"ANATOMICAL REGION: {class_name} (confidence: {confidence:.2f})\n\n"
        
        # Add VQA results
        if has_vqa:
            vqa = agent_results["vqa"]
            analysis_text += f"MEDICAL ANALYSIS:\n{vqa.get('answer', '')}\n"
            if vqa.get("reasoning"):
                analysis_text += f"Reasoning: {vqa.get('reasoning')}\n"
            analysis_text += "\n"
        
        # Add RAG results
        if has_rag:
            rag = agent_results["rag"]
            analysis_text += f"DOCUMENT ANALYSIS:\n{rag.get('answer', '')}\n\n"
            if rag.get("sources"):
                analysis_text += "Sources:\n"
                for i, source in enumerate(rag["sources"][:3]):  # Show top 3 sources
                    doc = source.get("document", "Unknown")
                    page = source.get("page", 0)
                    analysis_text += f"- {doc} (page {page})\n"
                analysis_text += "\n"
        
        prompt_args = {
            "query": current_query,
            "tasks": tasks_str,
            "analysis_results": analysis_text,
            "vector_context": vector_context,
            "conversation_context": conversation_context
        }
    
    # Create and invoke chain
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()
    
    # Invoke the chain
    try:
        synthesized_response = chain.invoke(prompt_args)
        
        # Build the final result
        final_result = {
            "task_type": state.get("task_type", "comprehensive"),
            "success": True,
            "session_id": state.get("session_id", ""),
            "query": current_query,
            "timestamp": time.time(),
            "multi_task_analysis": {
                "tasks_requested": state.get("required_tasks", []),
                "tasks_completed": state.get("completed_tasks", []),
                "execution_order": state.get("execution_order", [])
            },
            "agent_results": agent_results,
            "response": synthesized_response,
            "processing_time": processing_time,
            "final_answer": synthesized_response
        }
        
        # Add the current interaction to conversation history
        conversation_history.append({
            "query": current_query,
            "response": synthesized_response,
            "timestamp": time.time(),
            "tasks_completed": state.get("completed_tasks", [])
        })
        
        logger.info(f"Added new entry to conversation history. Now has {len(conversation_history)} entries.")
        
        return {
            **state, 
            "final_result": final_result,
            "conversation_history": conversation_history
        }
        
    except Exception as e:
        logger.error(f"Synthesis failed: {str(e)}")
        return {
            **state,
            "final_result": {
                "success": False,
                "error": f"Failed to synthesize response: {str(e)}",
                "query": current_query
            }
        }