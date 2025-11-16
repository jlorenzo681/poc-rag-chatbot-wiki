# Systemd Service Setup

This directory contains systemd service files for running the RAG Chatbot as a system service.

## Installation

### 1. Update the Service File

Edit `rag-chatbot.service` and replace `/path/to/your/project` with the actual path to your project directory.

### 2. Copy Service File

For user service (recommended):
```bash
mkdir -p ~/.config/systemd/user
cp deployment/systemd/rag-chatbot.service ~/.config/systemd/user/
```

For system-wide service (requires root):
```bash
sudo cp deployment/systemd/rag-chatbot.service /etc/systemd/system/
```

### 3. Reload Systemd

For user service:
```bash
systemctl --user daemon-reload
```

For system service:
```bash
sudo systemctl daemon-reload
```

### 4. Enable and Start Service

For user service:
```bash
systemctl --user enable rag-chatbot
systemctl --user start rag-chatbot
```

For system service:
```bash
sudo systemctl enable rag-chatbot
sudo systemctl start rag-chatbot
```

## Management Commands

### Check Status
```bash
systemctl --user status rag-chatbot
```

### View Logs
```bash
journalctl --user -u rag-chatbot -f
```

### Restart Service
```bash
systemctl --user restart rag-chatbot
```

### Stop Service
```bash
systemctl --user stop rag-chatbot
```

### Disable Service
```bash
systemctl --user disable rag-chatbot
```

## Auto-start on Boot

For user services to start on boot without login:
```bash
loginctl enable-linger $USER
```

## Troubleshooting

If the service fails to start:

1. Check the logs:
   ```bash
   journalctl --user -u rag-chatbot -n 50
   ```

2. Verify the image exists:
   ```bash
   podman images | grep rag-chatbot
   ```

3. Test manual start:
   ```bash
   podman run --rm -p 8501:8501 --env-file .env rag-chatbot:latest
   ```

4. Check paths in the service file are correct
5. Ensure .env file has proper permissions and API keys
