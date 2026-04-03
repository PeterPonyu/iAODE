#!/usr/bin/env python3
"""
Production Server Runner for iAODE
Serves the static frontend build and API backend together
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Ensure we are running from the api directory
    # This helps with relative path resolution for static files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # Add current directory to python path to find modules
    sys.path.append(current_dir)
    
    # Also add parent directory to find iaode package
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)

    print("╔════════════════════════════════════════════════════════════════╗")
    print("║          iAODE Production Server - Starting...                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    print(f"🚀 Starting Server from: {current_dir}")
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Server Information:")
    print("  Application URL:    http://localhost:8000/")
    print("  API Documentation:  http://localhost:8000/docs")
    print("  API Base:           http://localhost:8000/")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    # Run Uvicorn
    # reload=True allows you to change python code without restarting
    # For production deployment, set reload=False
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )
