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
            echo "  --pull-models           Pull all Ollama models after deployment"
            echo "  --pull-model MODEL      Pull specific Ollama model (e.g., llama3.1:8b)"
            exit 1
            ;;
    esac
done

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

# Compose file location
COMPOSE_FILE="podman-compose.yml"

# Check if podman compose is available (Podman Desktop includes this)
if podman compose version &> /dev/null; then
    COMPOSE_CMD="podman compose -f $COMPOSE_FILE"
    echo -e "${GREEN}✓ Using podman compose${NC}"
elif command -v podman-compose &> /dev/null; then
    COMPOSE_CMD="podman-compose -f $COMPOSE_FILE"
    echo -e "${GREEN}✓ Using podman-compose${NC}"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose -f $COMPOSE_FILE"
    echo -e "${YELLOW}⚠ Using docker-compose with Podman${NC}"
else
    echo -e "${RED}Error: No compose tool found${NC}"
    echo "Please install Podman Desktop or podman-compose:"
    echo "  brew install podman-compose"
    echo "  or: pip install podman-compose"
    exit 1
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

# Stop existing containers via compose
echo -e "\n${YELLOW}Stopping existing containers...${NC}"
$COMPOSE_CMD down 2>/dev/null || true

# Start all services using compose
echo -e "\n${GREEN}Starting RAG Chatbot and Ollama services...${NC}"
$COMPOSE_CMD up -d

# Wait for application to start
echo -e "\n${YELLOW}Waiting for application to start...${NC}"
sleep 5

# Check if containers are running
if podman ps | grep -q rag-chatbot && podman ps | grep -q ollama; then
    echo -e "\n${GREEN}======================================"
    echo "✓ Deployment successful!"
    echo "======================================${NC}"
    echo ""
    echo "Services running:"
    echo "  - RAG Chatbot: http://localhost:8501"
    echo "  - Ollama:      http://localhost:11434"
    echo ""
    echo "Useful commands:"
    echo "  View logs:           podman logs -f rag-chatbot"
    echo "  View ollama logs:    podman logs -f ollama"
    echo "  Stop all:            $COMPOSE_CMD down"
    echo "  Restart all:         $COMPOSE_CMD restart"
    echo "  Container status:    $COMPOSE_CMD ps"
    echo "  Pull ollama models:  ./scripts/pull-ollama-models.sh [--all|model_name]"
    echo ""

    # Pull Ollama models if requested
    if [ "$PULL_MODELS" = true ]; then
        echo -e "${YELLOW}Waiting for Ollama to be fully ready...${NC}"
        sleep 5

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
    echo "✗ Deployment failed!"
    echo "======================================${NC}"
    echo "Check logs with:"
    echo "  podman logs rag-chatbot"
    echo "  podman logs ollama"
    exit 1
fi
