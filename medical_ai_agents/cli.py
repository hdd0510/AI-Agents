#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Medical AI System - CLI
---------------------
Command Line Interface cho hệ thống AI y tế.
"""

import argparse
import json
import sys
import os
from typing import Optional, Dict, Any

from medical_ai_agents import MedicalAISystem, MedicalGraphConfig, __version__

def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Medical AI System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Version command
    parser_version = subparsers.add_parser("version", help="Show version")
    
    # Analyze command
    parser_analyze = subparsers.add_parser("analyze", help="Analyze a medical image")
    parser_analyze.add_argument("--image", required=True, help="Path to the image file")
    parser_analyze.add_argument("--query", help="Optional query or question about the image")
    parser_analyze.add_argument("--context", help="Medical context in JSON format")
    parser_analyze.add_argument("--output", help="Output file path for results (default: stdout)")
    parser_analyze.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    
    # Serve command
    parser_serve = subparsers.add_parser("serve", help="Start API server")
    parser_serve.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser_serve.add_argument("--device", default="cuda", help="Device to use (cuda or cpu)")
    
    args = parser.parse_args()
    
    # If no command provided, show help
    if not args.command:
        parser.print_help()
        return
    
    # Handle version command
    if args.command == "version":
        print(f"Medical AI System version {__version__}")
        return
    
    # Handle analyze command
    if args.command == "analyze":
        # Create config
        config = MedicalGraphConfig(
            device=args.device,
            use_reflection=True
        )
        
        # Initialize system
        system = MedicalAISystem(config)
        
        # Parse context if provided
        context = None
        if args.context:
            try:
                context = json.loads(args.context)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in context: {args.context}")
                return
        
        # Analyze image
        result = system.analyze(
            image_path=args.image,
            query=args.query,
            medical_context=context
        )
        
        # Output result
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        
        return
    
    # Handle serve command
    if args.command == "serve":
        try:
            from fastapi import FastAPI, UploadFile, File, Form
            from fastapi.middleware.cors import CORSMiddleware
            import uvicorn
            from pydantic import BaseModel
            import tempfile
            import shutil
        except ImportError:
            print("Error: FastAPI, uvicorn and other dependencies required for serve are not installed.")
            print("Please install with: pip install fastapi uvicorn python-multipart")
            return
        
        # Create config
        config = MedicalGraphConfig(
            device=args.device,
            use_reflection=True
        )
        
        # Initialize system
        system = MedicalAISystem(config)
        
        # Create API app
        app = FastAPI(title="Medical AI API", version=__version__)
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # API model
        class AnalysisResponse(BaseModel):
            success: bool
            answer: Optional[str] = None
            error: Optional[str] = None
            polyp_count: Optional[int] = None
            processing_time: Optional[float] = None
        
        # Root endpoint
        @app.get("/")
        def read_root():
            return {"message": "Medical AI API is running", "version": __version__}
        
        # Health check
        @app.get("/health")
        def health_check():
            return {"status": "ok"}
        
        # Analyze endpoint
        @app.post("/analyze", response_model=AnalysisResponse)
        async def analyze_image(
            image: UploadFile = File(...),
            query: str = Form(None),
            context: str = Form(None)
        ):
            # Create temp file for image
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                # Copy uploaded file to temp file
                shutil.copyfileobj(image.file, temp_file)
                temp_path = temp_file.name
            
            try:
                # Parse context if provided
                medical_context = None
                if context:
                    try:
                        medical_context = json.loads(context)
                    except json.JSONDecodeError:
                        return AnalysisResponse(
                            success=False,
                            error=f"Invalid JSON in context: {context}"
                        )
                
                # Analyze image
                result = system.analyze(
                    image_path=temp_path,
                    query=query,
                    medical_context=medical_context
                )
                
                # Clean up temp file
                os.unlink(temp_path)
                
                # Return result
                if "error" in result:
                    return AnalysisResponse(
                        success=False,
                        error=result["error"]
                    )
                
                return AnalysisResponse(
                    success=True,
                    answer=result.get("answer"),
                    polyp_count=result.get("polyp_count", 0),
                    processing_time=result.get("processing_time")
                )
                
            except Exception as e:
                # Clean up temp file
                os.unlink(temp_path)
                
                return AnalysisResponse(
                    success=False,
                    error=f"Analysis failed: {str(e)}"
                )
        
        # Start server
        print(f"Starting API server at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        return

if __name__ == "__main__":
    main()