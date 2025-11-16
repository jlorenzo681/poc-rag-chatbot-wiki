#!/bin/bash
# Cleanup script for RAG Chatbot
# Removes containers (keeps images and volumes by default)
# Use flags: --images to remove images, --volumes to remove volumes, --all for everything

set -e

echo "======================================"
echo "RAG Chatbot - Cleanup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse flags
REMOVE_IMAGES=false
REMOVE_VOLUMES=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --images)
            REMOVE_IMAGES=true
            shift
            ;;
        --volumes)
            REMOVE_VOLUMES=true
            shift
            ;;
        --all)
            REMOVE_IMAGES=true
            REMOVE_VOLUMES=true
            shift
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            echo "Usage: $0 [--images] [--volumes] [--all]"
            echo "  --images   Remove images too"
            echo "  --volumes  Remove volumes too (Ollama models)"
            echo "  --all      Remove everything (images + volumes)"
            exit 1
            ;;
    esac
done

# Check if Podman is installed
if ! command -v podman &> /dev/null; then
    echo -e "${RED}Error: Podman is not installed${NC}"
    exit 1
fi

# Stop and remove rag-chatbot container
echo -e "\n${YELLOW}Stopping and removing rag-chatbot container...${NC}"
if podman ps -a | grep -q rag-chatbot; then
    podman stop rag-chatbot 2>/dev/null || true
    podman rm rag-chatbot 2>/dev/null || true
    echo -e "${GREEN}✓ Container removed${NC}"
else
    echo -e "${YELLOW}⚠ Container not found${NC}"
fi

# Stop and remove ollama container (if using podman-compose)
echo -e "\n${YELLOW}Stopping and removing ollama container...${NC}"
if podman ps -a | grep -q ollama; then
    podman stop ollama 2>/dev/null || true
    podman rm ollama 2>/dev/null || true
    echo -e "${GREEN}✓ Ollama container removed${NC}"
else
    echo -e "${YELLOW}⚠ Ollama container not found${NC}"
fi

# Remove images if flag set
if [ "$REMOVE_IMAGES" = true ]; then
    echo -e "\n${YELLOW}Removing images...${NC}"

    if podman image exists rag-chatbot:latest; then
        podman rmi rag-chatbot:latest 2>/dev/null || true
        echo -e "${GREEN}✓ rag-chatbot image removed${NC}"
    else
        echo -e "${YELLOW}⚠ rag-chatbot image not found${NC}"
    fi

    if podman image exists ollama/ollama:latest; then
        podman rmi ollama/ollama:latest 2>/dev/null || true
        echo -e "${GREEN}✓ Ollama image removed${NC}"
    else
        echo -e "${YELLOW}⚠ Ollama image not found${NC}"
    fi
else
    echo -e "\n${YELLOW}⚠ Images kept (use --images flag to remove)${NC}"
fi

# Remove volumes if flag set
if [ "$REMOVE_VOLUMES" = true ]; then
    echo -e "\n${YELLOW}Removing volumes...${NC}"

    if podman volume exists ollama-data; then
        podman volume rm ollama-data 2>/dev/null || true
        echo -e "${GREEN}✓ Ollama data volume removed${NC}"
    else
        echo -e "${YELLOW}⚠ Ollama data volume not found${NC}"
    fi
else
    echo -e "\n${YELLOW}⚠ Volumes kept (use --volumes flag to remove Ollama models)${NC}"
fi

# Remove network if it exists and has no containers
echo -e "\n${YELLOW}Cleaning up network...${NC}"
if podman network exists rag-network; then
    # Check if network has any containers
    NETWORK_CONTAINERS=$(podman network inspect rag-network --format '{{len .Containers}}' 2>/dev/null || echo "0")
    if [ "$NETWORK_CONTAINERS" = "0" ]; then
        podman network rm rag-network 2>/dev/null || true
        echo -e "${GREEN}✓ Network removed${NC}"
    else
        echo -e "${YELLOW}⚠ Network has active containers, keeping it${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Network not found${NC}"
fi

# Summary
echo -e "\n${GREEN}======================================"
echo "✓ Cleanup complete!"
echo "======================================${NC}"
echo ""
echo "Remaining resources:"
echo ""
echo "Containers:"
podman ps -a | grep -E 'CONTAINER|rag-chatbot|ollama' || echo "  None"
echo ""
echo "Images:"
podman images | grep -E 'REPOSITORY|rag-chatbot|ollama' || echo "  None"
echo ""
echo "Volumes:"
podman volume ls | grep -E 'DRIVER|ollama-data' || echo "  None"
echo ""
