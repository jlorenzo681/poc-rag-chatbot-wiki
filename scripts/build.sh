#!/bin/bash
# Build RAG Chatbot container image

set -e

echo "Building RAG Chatbot container image..."

podman build --no-cache -t rag-chatbot:latest -f Containerfile .

echo "✓ Build complete!"
echo ""
echo "Image details:"
podman images | grep rag-chatbot || true
