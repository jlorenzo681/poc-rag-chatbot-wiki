#!/bin/bash
# Deployment script for RAG Chatbot using Podman

set -e

echo "======================================"
echo "RAG Chatbot - Podman Deployment"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please create .env file with your API keys:"
    echo "  cp .env.example .env"
    echo "  # Edit .env and add your GROQ_API_KEY"
    exit 1
fi

# Source environment variables
export $(cat .env | grep -v '^#' | xargs)

# Check for required API key
if [ -z "$GROQ_API_KEY" ]; then
    echo -e "${RED}Error: GROQ_API_KEY not set in .env file${NC}"
    exit 1
fi

# Check if Podman is installed
if ! command -v podman &> /dev/null; then
    echo -e "${RED}Error: Podman is not installed${NC}"
    echo "Please install Podman: https://podman.io/getting-started/installation"
    exit 1
fi

echo -e "${GREEN}✓ Podman is installed${NC}"

# Check if podman-compose is available
if command -v podman-compose &> /dev/null; then
    COMPOSE_CMD="podman-compose"
    echo -e "${GREEN}✓ Using podman-compose${NC}"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    echo -e "${YELLOW}⚠ Using docker-compose with Podman${NC}"
else
    echo -e "${YELLOW}⚠ Neither podman-compose nor docker-compose found${NC}"
    echo "Installing podman-compose..."
    pip install podman-compose
    COMPOSE_CMD="podman-compose"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p data/documents data/vector_stores logs

# Build the image only if it doesn't exist
if ! podman image exists rag-chatbot:latest; then
    echo -e "\n${YELLOW}Image not found. Building container image...${NC}"
    podman build -t rag-chatbot:latest -f Containerfile .
else
    echo -e "\n${GREEN}✓ Image rag-chatbot:latest already exists${NC}"
fi

# Stop and remove existing container if running
if podman ps -a | grep -q rag-chatbot; then
    echo -e "\n${YELLOW}Stopping existing container...${NC}"
    podman stop rag-chatbot || true
    podman rm rag-chatbot || true
fi

# Start the application
echo -e "\n${GREEN}Starting RAG Chatbot...${NC}"
$COMPOSE_CMD up -d

# Wait for application to start
echo -e "\n${YELLOW}Waiting for application to start...${NC}"
sleep 5

# Check if container is running
if podman ps | grep -q rag-chatbot; then
    echo -e "\n${GREEN}======================================"
    echo "✓ Deployment successful!"
    echo "======================================${NC}"
    echo ""
    echo "Application is running at: http://localhost:8501"
    echo ""
    echo "Useful commands:"
    echo "  View logs:        podman logs -f rag-chatbot"
    echo "  Stop:             $COMPOSE_CMD down"
    echo "  Restart:          $COMPOSE_CMD restart"
    echo "  Container status: podman ps"
else
    echo -e "\n${RED}======================================"
    echo "✗ Deployment failed!"
    echo "======================================${NC}"
    echo "Check logs with: podman logs rag-chatbot"
    exit 1
fi
