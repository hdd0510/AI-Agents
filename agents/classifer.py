#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - API Documentation
-----------------------------------
FastAPI documentation and implementation for the Medical AI System API.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid
import os
import logging
import shutil
import time
from datetime import datetime

from medical_ai_system import MedicalAISystem, MedicalGraphConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("medical-ai-api")

# Initialize app
app = FastAPI(
    title="Medical AI System API",
    description="API for Medical AI System with LangGraph for endoscopy image analysis",
    version="1.0.0",
    openapi_url="/api/openapi.json",
    docs_url=None,
    redoc_url=None
)

# Define models
class MedicalContext(BaseModel):
    """Medical context information."""
    patient_history: Optional[str] = None
    previous_findings: Optional[str] = None
    patient_age: Optional[int] = None
    other_info: Optional[Dict[str, Any]] = None

class AnalysisRequest(BaseModel):
    """Analysis request model."""
    query: Optional[str] = Field(None, 
                              description="Medical question about the image")
    medical_context: Optional[MedicalContext] = Field(None,
                                                  description="Medical context information")

class AnalysisResponse(BaseModel):
    """Analysis response model."""
    session_id: str = Field(..., description="Unique session ID")
    task_type: str = Field(..., description="Type of analysis task")
    success: bool = Field(..., description="Whether analysis succeeded")
    polyp_count: Optional[int] = Field(None, description="Number of polyps detected")
    polyps: Optional[List[Dict[str, Any]]] = Field(None, description="Detected polyps information")
    modality: Optional[Dict[str, Any]] = Field(None, description="Imaging modality information")
    region: Optional[Dict[str, Any]] = Field(None, description="Anatomical region information")
    answer: Optional[str] = Field(None, description="Answer to the query")
    answer_confidence: Optional[float] = Field(None, description="Confidence in the answer")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    error: Optional[str] = Field(None, description="Error message if any")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
UPLOAD_DIR = "uploads"
RESULTS_DIR = "api_results"
system = None

# Create required directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize system
def get_system() -> MedicalAISystem:
    """Get or initialize the Medical AI System."""
    global system
    if system is None:
        logger.info("Initializing Medical AI System")
        config = MedicalGraphConfig(
            device="cuda",  # Change as needed
            output_path=RESULTS_DIR,
            use_reflection=True
        )
        system = MedicalAISystem(config)
    return system

# Custom docs endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )

@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """ReDoc documentation."""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# Analysis endpoint
@app.post("/analyze", 
         response_model=AnalysisResponse, 
         tags=["Analysis"],
         summary="Analyze medical image",
         description="Analyze a medical endoscopy image, optionally with a query")
async def analyze_image(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Medical image to analyze"),
    query: Optional[str] = Form(None, description="Medical question about the image"),
    medical_context: Optional[Dict[str, Any]] = Form(None, description="Medical context in JSON format"),
    system: MedicalAISystem = Depends(get_system)
):
    """
    Analyze a medical endoscopy image.
    
    - **image**: Medical image file (PNG, JPG)
    - **query**: Optional medical question about the image
    - **medical_context**: Optional medical context information in JSON format
    
    Returns analysis results including polyp detection, classification, and answer to query.
    """
    try:
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded image
        image_path = os.path.join(UPLOAD_DIR, f"{session_id}_{image.filename}")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Parse medical context if provided
        context_data = None
        if medical_context:
            context_data = medical_context
        
        # Start analysis
        logger.info(f"Starting analysis for session {session_id}")
        start_time = time.time()
        
        # Run analysis
        result = system.analyze(
            image_path=image_path,
            query=query,
            medical_context=context_data
        )
        
        # Add processing time
        if "processing_time" not in result:
            result["processing_time"] = time.time() - start_time
        
        # Schedule cleanup in background
        background_tasks.add_task(cleanup_files, image_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

def cleanup_files(file_path: str):
    """Clean up temporary files."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {str(e)}")

# Static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except:
    logger.warning("Static directory not found, static file serving disabled")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": f"Internal server error: {str(exc)}"}
    )

# Main entry point for Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)