#!/bin/bash
# Development mode script with hot reload
# Starts container with source code mounted as volumes

set -e

echo "======================================"
echo "RAG Chatbot - Development Mode"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please create .env file with your configuration:"
    echo "  cp .env.example .env"
    echo "  # Edit .env and add your API keys"
    exit 1
fi

# Source environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check if Podman is installed
if ! command -v podman &> /dev/null; then
    echo -e "${RED}Error: Podman is not installed${NC}"
    echo "Please install Podman: https://podman.io/getting-started/installation"
    exit 1
fi

echo -e "${GREEN}✓ Podman is installed${NC}"

# Stop and remove existing container if running
echo -e "\n${YELLOW}Stopping existing development container...${NC}"
podman stop rag-chatbot 2>/dev/null || true
podman rm rag-chatbot 2>/dev/null || true

# Check if image exists, build if needed
if ! podman image exists rag-chatbot:latest; then
    echo -e "\n${YELLOW}Image not found. Building...${NC}"
    ./scripts/build.sh
else
    echo -e "\n${GREEN}✓ Image rag-chatbot:latest exists${NC}"
fi

# Create necessary directories
echo -e "\n${YELLOW}Creating necessary directories...${NC}"
mkdir -p data/documents data/vector_stores logs

# Start container with source code mounted
echo -e "\n${GREEN}Starting development container with hot reload...${NC}"
podman run -d \
    --name rag-chatbot \
    -p 8501:8501 \
    --user root \
    -e GROQ_API_KEY="$GROQ_API_KEY" \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e OLLAMA_BASE_URL="$OLLAMA_BASE_URL" \
    -v ./app.py:/app/app.py:z \
    -v ./src:/app/src:z \
    -v ./config:/app/config:z \
    -v ./.streamlit:/app/.streamlit:z \
    -v ./data/documents:/app/data/documents:z \
    -v ./data/vector_stores:/app/data/vector_stores:z \
    -v ./logs:/app/logs:z \
    rag-chatbot:latest

# Wait for container to start
echo -e "\n${YELLOW}Waiting for container to start...${NC}"
sleep 3

# Check if container is running
if podman ps | grep -q rag-chatbot; then
    echo -e "\n${GREEN}======================================"
    echo "✓ Development mode started!"
    echo "======================================${NC}"
    echo ""
    echo "Application is running at: http://localhost:8501"
    echo ""
    echo "Hot reload enabled for:"
    echo "  - app.py"
    echo "  - src/"
    echo "  - config/"
    echo "  - .streamlit/"
    echo ""
    echo "Changes to these files will be reflected immediately!"
    echo ""
    echo "Useful commands:"
    echo "  View logs:        make logs"
    echo "  Stop:             make stop"
    echo "  Restart:          make dev (stop + start)"
    echo "  Container status: podman ps"
    echo ""
else
    echo -e "\n${RED}======================================"
    echo "✗ Failed to start development mode!"
    echo "======================================${NC}"
    echo "Check logs with: podman logs rag-chatbot"
    exit 1
fi
