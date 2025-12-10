#!/bin/bash
# Stop RAG Chatbot and Ollama containers

set -e

echo "Stopping RAG Chatbot and Ollama services..."

# Compose file
COMPOSE_FILE="podman-compose.yml"

# Check if podman-compose or docker-compose is available and use correct command
if podman compose version &> /dev/null; then
    echo "Using podman compose..."
    podman compose -f $COMPOSE_FILE down
elif command -v podman-compose &> /dev/null; then
    echo "Using podman-compose..."
    podman-compose -f $COMPOSE_FILE down
elif command -v docker-compose &> /dev/null; then
    echo "Using docker-compose..."
    docker-compose -f $COMPOSE_FILE down
else
    # Manual stop - stop both containers
    echo "Stopping containers manually..."
    podman stop rag-chatbot 2>/dev/null || true
    podman stop ollama 2>/dev/null || true
    podman rm rag-chatbot 2>/dev/null || true
    podman rm ollama 2>/dev/null || true
fi

echo "✓ All services stopped"
