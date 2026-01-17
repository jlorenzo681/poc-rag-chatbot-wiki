# Deployment Scripts

This directory contains scripts for managing the RAG Chatbot deployment with Docker/Podman.

## Available Scripts

### `deploy.sh`
Deploy the RAG Chatbot service using docker-compose.

```bash
# Basic deployment
./scripts/deploy.sh
```

### `dev.sh`
Start development mode with hot reload for code changes.

```bash
# Basic dev mode
./scripts/dev.sh
```

**Features:**
- Hot reload for `app.py`, `src/`, `config/`, and `.streamlit/`
- Development container connected to compose network

### `build.sh`
Build the RAG Chatbot container image.

```bash
./scripts/build.sh
```

### `stop.sh`
Stop all services.

```bash
./scripts/stop.sh
```

### `logs.sh`
View container logs.

```bash
# View rag-chatbot logs (default)
./scripts/logs.sh
```

### `clean.sh`
Clean up containers, images, volumes, and networks.

```bash
# Remove containers only
./scripts/clean.sh

# Remove containers and images
./scripts/clean.sh --images

# Remove containers and volumes
./scripts/clean.sh --volumes

# Remove everything
./scripts/clean.sh --all
```

**Options:**
- `--images` - Also remove container images
- `--volumes` - Also remove volumes
- `--all` - Remove everything (images + volumes)

## Quick Start

1. **First time setup:**
   ```bash
   # Create .env file
   cp .env.example .env
   # Edit .env and add your configuration

   # Deploy
   ./scripts/deploy.sh
   ```

2. **Development:**
   ```bash
   # Start dev mode with hot reload
   ./scripts/dev.sh
   ```

3. **View logs:**
   ```bash
   # Watch application logs
   ./scripts/logs.sh
   ```

4. **Stop services:**
   ```bash
   ./scripts/stop.sh
   ```

## Architecture

Scripts use `docker-compose.yml` to manage the services.
- **rag-chatbot**: Streamlit web application
- **backend**: FastAPI backend service
- **celery_worker**: Asynchronous task worker
- **redis**: Message broker and cache
- **neo4j**: Graph database

## Troubleshooting

**Network issues:**
- Always use `./scripts/deploy.sh` or `docker-compose up -d` to start services
- Never start containers individually with `docker run`
