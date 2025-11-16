"""
Configuration settings for the RAG Chatbot application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_STORES_DIR = DATA_DIR / "vector_stores"
LOGS_DIR = PROJECT_ROOT / "logs"

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Document Processing Settings
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Embedding Settings
DEFAULT_EMBEDDING_TYPE = "huggingface"  # or "openai"
HUGGINGFACE_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# LLM Settings
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant8"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 500

# Retrieval Settings
DEFAULT_RETRIEVAL_K = 4

# Memory Settings
DEFAULT_MEMORY_TYPE = "buffer"  # or "window"
DEFAULT_WINDOW_SIZE = 5

# Streamlit Settings
STREAMLIT_PAGE_TITLE = "Document Q&A Chatbot"
STREAMLIT_PAGE_ICON = "📚"
STREAMLIT_LAYOUT = "wide"

# Supported file types
SUPPORTED_FILE_TYPES = ["pdf", "txt", "md"]

# Available Groq models
GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "compound"
]

# Create directories if they don't exist
for directory in [DATA_DIR, DOCUMENTS_DIR, VECTOR_STORES_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
