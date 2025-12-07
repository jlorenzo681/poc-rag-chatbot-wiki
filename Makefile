# Makefile for RAG Chatbot Project

.PHONY: help build run stop logs clean clean-all clean-images clean-volumes deploy test install dev pull-models

# Default target
help:
	@echo "RAG Chatbot - Podman Deployment"
	@echo ""
	@echo "Available commands:"
	@echo "  make install      - Install dependencies locally"
	@echo "  make build        - Build Podman container image"
	@echo "  make run          - Run application locally (non-containerized)"
	@echo "  make deploy       - Deploy with Podman"
	@echo "  make dev          - Deploy in development mode (containerized, hot reload)"
	@echo "  make stop         - Stop Podman containers"
	@echo "  make restart      - Restart Podman containers"
	@echo "  make logs         - View container logs"
	@echo "  make shell        - Open shell in running container"
	@echo "  make clean        - Clean up containers only"
	@echo "  make clean-images - Clean up containers and images"
	@echo "  make clean-volumes- Clean up containers and volumes (deletes Ollama models!)"
	@echo "  make clean-all    - Clean up everything (containers, images, volumes)"
	@echo "  make pull-models  - Pull all Ollama models"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linters"
	@echo ""

# Install Python dependencies
install:
	@echo "Installing dependencies..."
	uv sync
	@echo "✓ Dependencies installed"

# Build container image
build:
	@echo "Building container image..."
	@./scripts/build.sh

# Run locally without container
run:
	@echo "Starting application locally..."
	streamlit run app.py

# Deploy with Podman
deploy:
	@./scripts/deploy.sh

# Development mode with hot reload
dev:
	@./scripts/dev.sh

# Stop containers
stop:
	@./scripts/stop.sh

# Restart containers
restart: stop deploy

# View logs
logs:
	@./scripts/logs.sh

# Open shell in container
shell:
	@echo "Opening shell in container..."
	podman exec -it rag-chatbot /bin/bash

# Clean up - containers only
clean:
	@./scripts/clean.sh

# Clean up - containers and images
clean-images:
	@./scripts/clean.sh --images

# Clean up - containers and volumes
clean-volumes:
	@./scripts/clean.sh --volumes

# Clean up - everything
clean-all:
	@./scripts/clean.sh --all

# Pull Ollama models
pull-models:
	@./scripts/pull-ollama-models.sh --all

# Run tests
test:
	@echo "Running tests..."
	pytest tests/ -v

# Lint code
lint:
	@echo "Running linters..."
	@command -v ruff >/dev/null 2>&1 || { echo "ruff not found, install with: pip install ruff"; exit 1; }
	ruff check src/ app.py
	@echo "✓ Linting complete"

# Check system requirements
check:
	@echo "Checking system requirements..."
	@command -v podman >/dev/null 2>&1 && echo "✓ Podman installed" || echo "✗ Podman not found"
	@command -v python3 >/dev/null 2>&1 && echo "✓ Python3 installed" || echo "✗ Python3 not found"
	@[ -f .env ] && echo "✓ .env file exists" || echo "⚠ .env file not found"
	@echo ""
