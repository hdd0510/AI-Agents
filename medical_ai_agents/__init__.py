#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Package Initialization
----------------------------------------
Initialization file for the Medical AI System package.
"""

import logging
import os

# Set up logging
log_level = os.environ.get("MEDICAL_AI_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import main components
from medical_ai_agents.config import MedicalGraphConfig, SystemState, TaskType
from medical_ai_agents.main import MedicalAISystem
# from medical_ai_agents.tools.rag.vector_search import VectorSearchTool
# from medical_ai_agents.tools.rag.doc_retrieval import DocumentRetrievalTool
__all__ = [
    'MedicalGraphConfig',
    'SystemState',
    'TaskType',
    'MedicalAISystem',
    # 'VectorSearchTool',
    # 'DocumentRetrievalTool'
]

# Define package version
__version__ = "1.0.0"

# Silence specific loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)