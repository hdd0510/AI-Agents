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
            - If the query is general conversation, non-medical topics, or not related to healthcare → GENERAL
            
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
    """Synthesize results from multiple tasks and agents."""
    logger = logging.getLogger("graph.nodes.result_synthesizer")
    
    # Extract conversation history for debugging
    conversation_history = state.get("conversation_history", [])
    
    logger.info(f"Synthesizer received conversation history: {len(conversation_history)} entries")
    if conversation_history:
        logger.info(f"First history entry: {conversation_history[0].get('query', 'Unknown')[:30]}...")
        logger.info(f"Last history entry: {conversation_history[-1].get('query', 'Unknown')[:30]}...")
        
        # Log each entry for detailed debugging
        for i, entry in enumerate(conversation_history[-2:]):  # Show last 2 entries
            logger.info(f"History entry {i}: query='{entry.get('query', 'None')[:30]}...', "
                         f"is_system={entry.get('is_system', False)}, "
                         f"is_pending={entry.get('is_pending', False)}")
    
    # Extract state parameters
    current_query = state.get("query", "")
    is_text_only = state.get("is_text_only", False)
    task_type = state.get("task_type", "unknown")
    
    # Task info
    required_tasks = state.get("required_tasks", [])
    completed_tasks = state.get("completed_tasks", [])
    execution_order = state.get("execution_order", [])
    logger.info(f"REQUIRED TASKS: {required_tasks}")
    logger.info(f"COMPLETED TASKS: {completed_tasks}")
    logger.info(f"EXECUTION ORDER: {execution_order}")
    
    # Clean up pending/debug entries from conversation history
    if conversation_history:
        conversation_history = [entry for entry in conversation_history if not entry.get("is_pending", False)]
        # Update the state with cleaned history
        state["conversation_history"] = conversation_history
        logger.info(f"Cleaned conversation history, removed pending entries. Now has {len(conversation_history)} entries.")
    
    logger.info(f"CONVERSATION HISTORY ENTRIES: {len(conversation_history)}")
    if conversation_history:
        for i, entry in enumerate(conversation_history[-10:]):  # Show last 10 entries thay vì 3
            logger.info(f"HISTORY ENTRY {i}:")
            logger.info(f"  - QUERY: {entry.get('query', 'None')[:50]}...")
            resp = entry.get('response', 'None')
            if resp and len(resp) > 50:
                resp = resp[:50] + "..."
            logger.info(f"  - RESPONSE: {resp}")
            logger.info(f"  - TIMESTAMP: {entry.get('timestamp', 'None')}")
            logger.info(f"  - IS_SYSTEM: {entry.get('is_system', False)}")
    
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
    
    logger.info(f"AVAILABLE AGENT RESULTS: {agent_results_keys}")
    logger.info("=" * 50)
    
    # Calculate processing time
    start_time = state.get("start_time", time.time())
    processing_time = time.time() - start_time
    
    # Debug information about conversation history
    logger.info(f"Synthesizer received conversation history: {len(conversation_history)} entries")
    if conversation_history:
        logger.info(f"First history entry: {conversation_history[0].get('query', 'Unknown')[:30]}...")
        logger.info(f"Last history entry: {conversation_history[-1].get('query', 'Unknown')[:30]}...")
        
        # Add detailed logging of conversation pairs
        logger.info("=" * 50)
        logger.info("CONVERSATION HISTORY PAIRS:")
        for i, entry in enumerate(conversation_history):
            is_system = entry.get("is_system", False)
            entry_type = "SYSTEM" if is_system else "USER"
            logger.info(f"ENTRY {i} [{entry_type}]:")
            logger.info(f"  - QUERY: {entry.get('query', 'None')[:50]}...")
            logger.info(f"  - RESPONSE: {entry.get('response', 'None')[:50]}...")
            logger.info(f"  - TIMESTAMP: {entry.get('timestamp', 'None')}")
        logger.info("=" * 50)
    
    # Special handling for "what did I say" meta-queries
    if current_query and current_query.lower() in ["what did i say", "what did i ask"]:
        logger.info("[META-QUERY] Processing 'what did I say' type question")
        
        # Extract only actual user messages, excluding system messages, pending, and meta-queries
        user_messages = []
        for entry in conversation_history:
            # Skip system messages, pending entries and meta queries
            if (not entry.get("is_system", False) and 
                not entry.get("is_pending", False) and 
                not entry.get("is_meta", False) and 
                entry.get("query", "").strip()):
                
                user_query = entry.get("query", "").strip()
                # Also skip the current query itself
                if user_query.lower() != current_query.lower():
                    user_messages.append(user_query)
        
        logger.info(f"[META-QUERY] Found {len(user_messages)} user messages: {user_messages}")
        
        if not user_messages:
            response = "You haven't asked any questions yet."
        elif len(user_messages) == 1:
            response = f"You asked: \"{user_messages[0]}\""
        else:
            # Return the most recent question (excluding the current meta-query)
            response = f"You asked: \"{user_messages[-1]}\""
        
        logger.info(f"[META-QUERY] Response: {response}")
        
        # Create final result
        final_result = {
            "task_type": "meta_query",
            "success": True,
            "session_id": state.get("session_id", ""),
            "query": current_query,
            "timestamp": time.time(),
            "response": response,
            "processing_time": processing_time,
            "final_answer": response
        }
        
        # Add the current interaction to conversation history
        conversation_history.append({
            "query": current_query,
            "response": response,
            "timestamp": time.time(),
            "is_meta": True  # Mark as meta-query
        })
        
        logger.info(f"[META-QUERY] Added meta-query to history. Now has {len(conversation_history)} entries")
        
        return {
            **state, 
            "final_result": final_result,
            "conversation_history": conversation_history
        }
    
    # Check if it's a general (non-medical) query
    is_general_query = "general_query" in state.get("required_tasks", [])
    # is_text_only = state.get("is_text_only", False)

    # For general queries, use a simpler prompt without medical context
    if is_general_query:
        logger.info("Processing general (non-medical) query")
        
        # Include conversation history in prompt
        conversation_context = ""
        if conversation_history:
            # Filter out system messages, meta-queries and pending messages for context
            filtered_entries = [
                entry for entry in conversation_history 
                if not entry.get("is_system", False) and 
                not entry.get("is_meta", False) and 
                not entry.get("is_pending", False)
            ]
            
            # Get the last 10 relevant interactions thay vì 3
            recent_conversations = filtered_entries[-10:] if filtered_entries else []
            
            if recent_conversations:
                conversation_context = "Previous conversation:\n"
                for i, conv in enumerate(recent_conversations):
                    conversation_context += f"User: {conv.get('query', '')}\n"
                    # Limit system response to prevent overly long prompts
                    system_response = conv.get('response', '')
                    if len(system_response) > 150:
                        system_response = system_response[:147] + "..."
                    conversation_context += f"System: {system_response}\n\n"
                
                # Add a separator to make the context more visible to the model
                conversation_context += "--------------------\n"
            
            logger.info(f"Built general query context with {len(recent_conversations)} recent user-system exchanges")
        
        general_prompt = PromptTemplate.from_template(
            """You are a Medical AI Assistant providing advice and information. The user asked a question that's not specifically medical in nature, but you should still maintain your medical identity.

{conversation_context}

Current Question: "{query}"

Respond to this question directly and conversationally. Even though this isn't a medical question, make it clear you are a Medical AI Assistant designed primarily for healthcare-related questions. Use a professional tone but avoid unnecessary medical terminology when responding to general questions.
"""
        )
        
        try:
            chain = general_prompt | llm | StrOutputParser()
            general_response = chain.invoke({
                "query": current_query,
                "conversation_context": conversation_context
            })
            
            final_result = {
                "task_type": state.get("task_type", "text_only"),
                "success": True,
                "session_id": state.get("session_id", ""),
                "query": current_query,
                "timestamp": time.time(),
                "response": general_response,
                "processing_time": processing_time,
                "final_answer": general_response,
                "is_general_query": True
            }
            
            # Add the current interaction to conversation history
            conversation_history.append({
                "query": current_query,
                "response": general_response,
                "timestamp": time.time()
            })
            
            logger.info(f"Added new general query entry to conversation history. Now has {len(conversation_history)} entries.")
            logger.info(f"New entry query: {current_query[:30]}...")
            logger.info(f"New entry response: {general_response[:30]}...")
            
            return {
                **state, 
                "final_result": final_result,
                "conversation_history": conversation_history
            }
            
        except Exception as e:
            logger.error(f"General query processing failed: {str(e)}")
            fallback_response = "I couldn't process your question. Could you please rephrase it?"
            
            final_result = {
                "task_type": state.get("task_type", "text_only"),
                "success": False,
                "error": str(e),
                "session_id": state.get("session_id", ""),
                "query": current_query,
                "timestamp": time.time(),
                "response": fallback_response,
                "processing_time": processing_time,
                "final_answer": fallback_response,
                "is_general_query": True
            }
            
            # Add the current interaction to conversation history even if it failed
            conversation_history.append({
                "query": current_query,
                "response": fallback_response,
                "timestamp": time.time(),
                "error": True
            })
            
            return {
                **state, 
                "final_result": final_result,
                "conversation_history": conversation_history
            }
    
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
    
    # Include conversation history in the medical prompt
    conversation_context = ""
    if conversation_history:
        # Filter out system initialization messages and meta-queries for context
        relevant_entries = [
            entry for entry in conversation_history 
            if not entry.get("is_system", False) and not entry.get("is_meta", False) and not entry.get("is_pending", False)
        ]
        
        # Get the last 10 interactions at most thay vì 3
        recent_conversations = relevant_entries[-10:] if relevant_entries else []
        
        if recent_conversations:
            conversation_context = "Previous conversation context:\n"
            for i, conv in enumerate(recent_conversations):
                conversation_context += f"User: {conv.get('query', '')}\n"
                # Limit system response to prevent overly long prompts
                system_response = conv.get('response', '')
                if len(system_response) > 150:  # Shorter limit for synthesizer to save space
                    system_response = system_response[:147] + "..."
                conversation_context += f"System: {system_response}\n\n"
            
            # Add a separator to make the context more visible to the model
            conversation_context += "--------------------\n"
        
        logger.info(f"Built conversation context with {len(recent_conversations)} recent user-system exchanges")
    
    # Set up different prompts based on the scenario
    if prioritize_rag:
        prompt_template = """You are a Medical AI Assistant specializing in healthcare and medical consultation. Your purpose is to provide accurate medical information based on document analysis.
        
{conversation_context}

The user asked: "{query}"

The document analysis yielded the following information:
{rag_answer}

Sources cited:
{rag_sources}

Your task:
1. Consider the previous conversation context if relevant
2. Respond to the user's query directly using the document analysis results
3. Maintain all citations and references to documents
4. Format your response in a clear, professional manner suited for medical consultation
5. Always identify yourself as a Medical AI Assistant
6. Respond in English, maintaining medical accuracy"""
    else:
        prompt_template = """You are a Medical AI Assistant specializing in healthcare and medical consultation. Your purpose is to provide comprehensive medical analysis.

{conversation_context}

The user asked: "{query}"

I have analyzed the available medical information:
{combined_results}

IMPORTANT INSTRUCTIONS:
1. Respond as a UNIFIED MEDICAL ASSISTANT, not as a collection of tools or agents
2. DO NOT mention separate tools, agents or model names (like LLaVA, modality detector, etc.)
3. DO NOT use phrases like "Based on the analysis" or "According to the medical AI system"
4. DO NOT list or itemize different analyses - integrate everything into a cohesive, natural response
5. Speak directly as a knowledgeable medical assistant (use "I" not "the system")
6. If the analysis contains uncertainties, incorporate them naturally without mentioning confidence scores
7. Maintain a professional, confident, and conversational medical tone

Your response must read like it comes from a single, unified medical assistant with deep expertise in gastroenterology and medical image analysis.
"""
    
    # Format the prompt arguments
    prompt_args = {
        "query": current_query,
        "tasks": tasks_str,
        "conversation_context": conversation_context
    }
    
    # Debug the exact conversation context being sent
    logger.info("=" * 50)
    logger.info("EXACT CONVERSATION CONTEXT FOR LLM:")
    logger.info(f"\n{conversation_context}")
    logger.info("=" * 50)
    
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
        logger.info(f"New entry query: {current_query[:30]}...")
        logger.info(f"New entry response: {synthesized_response[:30]}...")
        
        return {
            **state, 
            "final_result": final_result,
            "conversation_history": conversation_history
        }
        
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
            "query": current_query,
            "timestamp": time.time(),
            "response": fallback_response,
            "processing_time": processing_time,
            "final_answer": fallback_response
        }
        
        # Add the current interaction to conversation history even if it failed
        conversation_history.append({
            "query": current_query,
            "response": fallback_response,
            "timestamp": time.time(),
            "error": True
        })
        
        return {
            **state, 
            "final_result": final_result,
            "conversation_history": conversation_history
        }