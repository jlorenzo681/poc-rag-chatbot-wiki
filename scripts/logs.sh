#!/bin/bash
# View logs from RAG Chatbot container

set -e

# Parse arguments
# Parse arguments
CONTAINER="rag-chatbot"


echo "Viewing $CONTAINER logs (Ctrl+C to exit)..."
echo "=============================================="

docker logs -f $CONTAINER
