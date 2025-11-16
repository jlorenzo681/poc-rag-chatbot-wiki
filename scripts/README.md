# Deployment Scripts

This directory contains shell scripts for easy deployment and management of the RAG Chatbot using Podman.

## Available Scripts

### deploy.sh
**Full deployment script with checks and validation**

```bash
./scripts/deploy.sh
```

Features:
- Checks for Podman installation
- Validates .env file and API keys
- Creates necessary directories
- Builds container image
- Stops existing containers
- Starts new container with podman-compose
- Verifies deployment success
- Shows useful management commands

### build.sh
**Build the container image**

```bash
./scripts/build.sh
```

Builds the container image using the Containerfile. Useful when you want to rebuild after code changes.

### stop.sh
**Stop the running application**

```bash
./scripts/stop.sh
```

Stops and removes the RAG Chatbot containers using podman-compose or manual commands.

### logs.sh
**View container logs in real-time**

```bash
./scripts/logs.sh
```

Follows the container logs. Press Ctrl+C to exit.

## Usage Examples

### First-time deployment
```bash
# Set up environment
cp .env.example .env
nano .env  # Add your GROQ_API_KEY

# Deploy
./scripts/deploy.sh
```

### After code changes
```bash
# Rebuild and redeploy
./scripts/build.sh
./scripts/stop.sh
./scripts/deploy.sh
```

### Check application logs
```bash
./scripts/logs.sh
```

### Stop the application
```bash
./scripts/stop.sh
```

## Prerequisites

All scripts require:
- Podman installed
- .env file with GROQ_API_KEY
- Executable permissions (already set)

## Troubleshooting

### Permission Denied
```bash
# Make scripts executable
chmod +x scripts/*.sh
```

### podman-compose Not Found
The deploy script will automatically install it if not found, or you can install manually:
```bash
pip install podman-compose
```

### Scripts Don't Find Podman
Ensure Podman is in your PATH:
```bash
which podman
```

If not found, install Podman from: https://podman.io/getting-started/installation

## Alternative: Using Makefile

You can also use the Makefile in the project root for the same operations:

```bash
make deploy    # Same as ./scripts/deploy.sh
make build     # Same as ./scripts/build.sh
make stop      # Same as ./scripts/stop.sh
make logs      # Same as ./scripts/logs.sh
```

## Notes

- Scripts are designed to work from the project root directory
- They will create necessary directories if they don't exist
- All scripts provide colored output for better readability
- Error handling is built-in with helpful messages
