#!/bin/bash
# View logs from RAG Chatbot container

set -e

echo "Viewing RAG Chatbot logs (Ctrl+C to exit)..."
echo "=============================================="

podman logs -f rag-chatbot
