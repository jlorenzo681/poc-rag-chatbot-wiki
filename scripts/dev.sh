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

# Parse flags
PULL_MODELS=false
MODEL_TO_PULL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --pull-models)
            PULL_MODELS=true
            shift
            ;;
        --pull-model)
            PULL_MODELS=true
            MODEL_TO_PULL="$2"
            shift 2
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            echo "Usage: $0 [--pull-models] [--pull-model MODEL_NAME]"
            echo "  --pull-models           Pull all Ollama models after starting"
            echo "  --pull-model MODEL      Pull specific Ollama model (e.g., llama3.1:8b)"
            exit 1
            ;;
    esac
done

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

# Check if podman-compose is available
if command -v podman-compose &> /dev/null; then
    COMPOSE_CMD="podman-compose"
    echo -e "${GREEN}✓ Using podman-compose${NC}"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    echo -e "${YELLOW}⚠ Using docker-compose with Podman${NC}"
else
    echo -e "${YELLOW}⚠ podman-compose not found, installing...${NC}"
    pip install podman-compose
    COMPOSE_CMD="podman-compose"
fi

# Start Ollama service first using compose
echo -e "\n${YELLOW}Starting Ollama service...${NC}"
$COMPOSE_CMD up -d ollama

# Wait for Ollama to be ready
echo -e "${YELLOW}Waiting for Ollama to start...${NC}"
sleep 5

# Stop and remove existing rag-chatbot development container if running
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

# Get the compose network name
NETWORK_NAME=$(podman network ls --format '{{.Name}}' | grep 'rag-network' | head -1)
if [ -z "$NETWORK_NAME" ]; then
    NETWORK_NAME="poc-rag-chatbot-wiki_rag-network"
fi

# Start container with source code mounted and connected to the compose network
echo -e "\n${GREEN}Starting development container with hot reload...${NC}"
podman run -d \
    --name rag-chatbot \
    -p 8501:8501 \
    --network "$NETWORK_NAME" \
    --user root \
    -e GROQ_API_KEY="$GROQ_API_KEY" \
    -e OPENAI_API_KEY="$OPENAI_API_KEY" \
    -e OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}" \
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

# Check if containers are running
if podman ps | grep -q rag-chatbot; then
    echo -e "\n${GREEN}======================================"
    echo "✓ Development mode started!"
    echo "======================================${NC}"
    echo ""
    echo "Services running:"
    echo "  - RAG Chatbot: http://localhost:8501"
    echo "  - Ollama:      http://localhost:11434"
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
    echo "  View logs:           podman logs -f rag-chatbot"
    echo "  View ollama logs:    podman logs -f ollama"
    echo "  Stop dev container:  podman stop rag-chatbot"
    echo "  Stop all:            $COMPOSE_CMD down"
    echo "  Pull ollama models:  ./scripts/pull-ollama-models.sh [--all|model_name]"
    echo ""

    # Pull Ollama models if requested
    if [ "$PULL_MODELS" = true ]; then
        echo -e "${YELLOW}Waiting for Ollama to be fully ready...${NC}"
        sleep 3

        if [ -n "$MODEL_TO_PULL" ]; then
            echo -e "${GREEN}Pulling Ollama model: $MODEL_TO_PULL${NC}"
            ./scripts/pull-ollama-models.sh "$MODEL_TO_PULL"
        else
            echo -e "${GREEN}Pulling all Ollama models...${NC}"
            ./scripts/pull-ollama-models.sh --all
        fi
    else
        echo -e "${YELLOW}Note: No Ollama models pulled. To pull models, run:${NC}"
        echo "  ./scripts/pull-ollama-models.sh --all"
        echo "  or: ./scripts/pull-ollama-models.sh llama3.1:8b"
    fi
else
    echo -e "\n${RED}======================================"
    echo "✗ Failed to start development mode!"
    echo "======================================${NC}"
    echo "Check logs with: podman logs rag-chatbot"
    exit 1
fi
