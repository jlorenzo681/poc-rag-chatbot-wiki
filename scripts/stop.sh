#!/bin/bash
# Stop RAG Chatbot and Ollama containers

set -e

echo "Stopping RAG Chatbot and Ollama services..."

# Check if podman-compose is available
if command -v podman-compose &> /dev/null; then
    podman-compose down
elif command -v docker-compose &> /dev/null; then
    docker-compose down
else
    # Manual stop - stop both containers
    echo "Stopping containers manually..."
    podman stop rag-chatbot 2>/dev/null || true
    podman stop ollama 2>/dev/null || true
    podman rm rag-chatbot 2>/dev/null || true
    podman rm ollama 2>/dev/null || true
fi

echo "✓ All services stopped"
