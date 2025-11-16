#!/bin/bash
# Pull Ollama models used by the RAG Chatbot application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================"
echo "Ollama Model Pulling"
echo "======================================${NC}"

# Models available in the app (from app.py line 291)
MODELS=(
    "llama3.1:8b"
    "mistral:latest"
    "deepseek-r1:8b"
)

# Check if ollama container is running
if ! podman ps | grep -q ollama; then
    echo -e "${RED}Error: Ollama container is not running${NC}"
    echo "Please start it first with: podman-compose up -d ollama"
    exit 1
fi

echo -e "${GREEN}✓ Ollama container is running${NC}"
echo ""

# Parse arguments
PULL_ALL=false
SELECTED_MODELS=()

if [ $# -eq 0 ] || [ "$1" = "--all" ]; then
    PULL_ALL=true
    SELECTED_MODELS=("${MODELS[@]}")
    echo -e "${YELLOW}Pulling all available models...${NC}"
else
    # Pull specific models
    for model in "$@"; do
        if [[ " ${MODELS[@]} " =~ " ${model} " ]]; then
            SELECTED_MODELS+=("$model")
        else
            echo -e "${YELLOW}Warning: Model '$model' not in the app's model list${NC}"
            echo "Available models: ${MODELS[*]}"
        fi
    done
fi

if [ ${#SELECTED_MODELS[@]} -eq 0 ]; then
    echo -e "${RED}No valid models selected${NC}"
    echo "Usage: $0 [--all|model1 model2 ...]"
    echo "Available models: ${MODELS[*]}"
    exit 1
fi

echo ""
echo "Models to pull:"
for model in "${SELECTED_MODELS[@]}"; do
    echo "  - $model"
done
echo ""

# Pull each model
PULLED=0
SKIPPED=0
FAILED=0

for model in "${SELECTED_MODELS[@]}"; do
    echo -e "${BLUE}Checking model: ${model}${NC}"

    # Check if model already exists
    if podman exec ollama ollama list | grep -q "^${model}"; then
        echo -e "${GREEN}✓ Model ${model} already exists, skipping${NC}"
        SKIPPED=$((SKIPPED + 1))
    else
        echo -e "${YELLOW}Pulling ${model}... (this may take a while)${NC}"
        if podman exec ollama ollama pull "$model"; then
            echo -e "${GREEN}✓ Successfully pulled ${model}${NC}"
            PULLED=$((PULLED + 1))
        else
            echo -e "${RED}✗ Failed to pull ${model}${NC}"
            FAILED=$((FAILED + 1))
        fi
    fi
    echo ""
done

# Summary
echo -e "${BLUE}======================================"
echo "Summary"
echo "======================================${NC}"
echo -e "Models pulled:  ${GREEN}${PULLED}${NC}"
echo -e "Models skipped: ${YELLOW}${SKIPPED}${NC}"
echo -e "Models failed:  ${RED}${FAILED}${NC}"
echo ""

# List all available models
echo "Available Ollama models:"
podman exec ollama ollama list

exit $FAILED
