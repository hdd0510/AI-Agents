#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - Package Initialization (MODIFIED)
----------------------------------------
initialization với multi-task support.
"""

import logging
import os

# Set up logging
log_level = os.environ.get("MEDICAL_AI_LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import components
from medical_ai_agents.config import MedicalGraphConfig, SystemState, TaskType
from medical_ai_agents.main import EnhancedMedicalAISystem

# Backward compatibility aliases
MedicalAISystem = EnhancedMedicalAISystem  # For backward compatibility

__all__ = [
    'MedicalGraphConfig',
    'SystemState', 
    'TaskType',
    'MedicalAISystem',           # Backward compatibility
    'EnhancedMedicalAISystem'    # New system
]

# Define package version
__version__ = "1.1.0"

# Silence specific loggers
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
