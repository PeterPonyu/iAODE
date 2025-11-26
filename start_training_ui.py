#!/usr/bin/env python3
"""
Start iAODE Training UI with integrated backend
Cross-platform Python launcher
"""

import subprocess
import sys
import os
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    RESET = '\033[0m'
    
    @classmethod
    def green(cls, text):
        return f"{cls.GREEN}{text}{cls.RESET}"
    
    @classmethod
    def blue(cls, text):
        return f"{cls.BLUE}{text}{cls.RESET}"
    
    @classmethod
    def yellow(cls, text):
        return f"{cls.YELLOW}{text}{cls.RESET}"
    
    @classmethod
    def red(cls, text):
        return f"{cls.RED}{text}{cls.RESET}"

def print_header():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║          iAODE Training UI - Integrated Application           ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()

def check_directory():
    """Check if we're in the right directory"""
    if not Path("api").exists() or not Path("frontend").exists():
        print(Colors.red("Error: Must run from iAODE_dev root directory"))
        sys.exit(1)

def check_frontend_deps():
    """Check and install frontend dependencies if needed"""
    print()
    print(Colors.blue("Checking frontend dependencies..."))
    
    node_modules = Path("frontend/node_modules")
    if not node_modules.exists():
        print(Colors.yellow("Installing npm dependencies..."))
        os.chdir("frontend")
        subprocess.run(["npm", "install"], check=True)
        os.chdir("..")
        print(Colors.green("✓ Frontend dependencies installed"))
    else:
        print(Colors.green("✓ Frontend dependencies installed"))

def check_python_deps():
    """Check if Python dependencies are installed"""
    print()
    print(Colors.blue("Checking Python dependencies..."))
    
    try:
        import fastapi
        print(Colors.green("✓ FastAPI installed"))
    except ImportError:
        print(Colors.red("Error: FastAPI not installed"))
        print("Install with: pip install fastapi uvicorn")
        sys.exit(1)
    
    try:
        import iaode
        print(Colors.green("✓ iAODE package installed"))
    except ImportError:
        print(Colors.red("Error: iaode package not installed"))
        print("Install with: pip install -e .")
        sys.exit(1)

def start_server():
    """Start the integrated server"""
    print()
    print(Colors.blue("Starting integrated server..."))
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(Colors.green("Server Information:"))
    print("  API Backend:        http://localhost:8000/")
    print("  API Documentation:  http://localhost:8000/docs")
    print("  Training UI:        http://localhost:3000/")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print(Colors.yellow("Note: Frontend runs on port 3000, Backend API on port 8000"))
    print()
    print(Colors.yellow("Press Ctrl+C to stop all servers"))
    print()
    
    # Start backend and frontend in parallel
    import threading
    
    def start_backend():
        try:
            subprocess.run([
                sys.executable, "-m", "uvicorn",
                "api.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ])
        except KeyboardInterrupt:
            pass
    
    def start_frontend():
        try:
            os.chdir("frontend")
            subprocess.run(["npm", "run", "dev"])
        except KeyboardInterrupt:
            pass
        finally:
            os.chdir("..")
    
    # Start backend in a separate thread
    backend_thread = threading.Thread(target=start_backend, daemon=True)
    backend_thread.start()
    
    # Give backend time to start
    import time
    time.sleep(2)
    
    # Start frontend in main thread
    try:
        start_frontend()
    except KeyboardInterrupt:
        print()
        print(Colors.yellow("Servers stopped"))

def main():
    print_header()
    check_directory()
    check_frontend_deps()
    check_python_deps()
    start_server()

if __name__ == "__main__":
    main()
