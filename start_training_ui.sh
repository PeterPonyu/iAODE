#!/bin/bash
# Start iAODE Training UI with integrated backend

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║          iAODE Training UI - Integrated Application           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -d "api" ] || [ ! -d "frontend" ]; then
    echo -e "${RED}Error: Must run from iAODE_dev root directory${NC}"
    exit 1
fi

# Check if frontend is built
if [ ! -d "frontend/out" ]; then
    echo -e "${YELLOW}Frontend not built. Building now...${NC}"
    cd frontend
    npm install
    npm run build
    cd ..
    echo -e "${GREEN}✓ Frontend built successfully${NC}"
else
    echo -e "${GREEN}✓ Frontend already built${NC}"
fi

# Check Python dependencies
echo ""
echo -e "${BLUE}Checking Python dependencies...${NC}"
python -c "import fastapi" 2>/dev/null || {
    echo -e "${RED}Error: FastAPI not installed${NC}"
    echo "Install with: pip install fastapi uvicorn"
    exit 1
}
echo -e "${GREEN}✓ FastAPI installed${NC}"

python -c "import iaode" 2>/dev/null || {
    echo -e "${RED}Error: iaode package not installed${NC}"
    echo "Install with: pip install -e ."
    exit 1
}
echo -e "${GREEN}✓ iAODE package installed${NC}"

# Start the server
echo ""
echo -e "${BLUE}Starting integrated server...${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}Server Information:${NC}"
echo "  API Documentation:  http://localhost:8000/docs"
echo "  Training UI:        http://localhost:8000/ui"
echo "  API Root:           http://localhost:8000/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start uvicorn
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
