#!/bin/bash
# Stop RAG Chatbot container

set -e

echo "Stopping RAG Chatbot..."

# Check if podman-compose is available
if command -v podman-compose &> /dev/null; then
    podman-compose down
elif command -v docker-compose &> /dev/null; then
    docker-compose down
else
    # Manual stop
    podman stop rag-chatbot || true
    podman rm rag-chatbot || true
fi

echo "✓ RAG Chatbot stopped"
