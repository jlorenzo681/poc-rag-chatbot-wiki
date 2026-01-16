"""
Configuration settings for the RAG Chatbot application.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT: Path = Path(__file__).parent.parent

# Data directories
DATA_DIR: Path = PROJECT_ROOT / "data"
DOCUMENTS_DIR: Path = DATA_DIR / "documents"
VECTOR_STORES_DIR: Path = DATA_DIR / "vector_stores"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# API Keys
# OPENAI_API_KEY removed

# Document Processing Settings
DEFAULT_CHUNK_SIZE: int = 1000
DEFAULT_CHUNK_OVERLAP: int = 200

# Embedding Settings
DEFAULT_EMBEDDING_TYPE: str = "ollama"  # Changed from huggingface to ollama

# Dynamic Embedding Models
EMBEDDING_MODEL_EN: str = "nomic-embed-text" # High quality English model (Ollama)
EMBEDDING_MODEL_MULTILINGUAL: str = "bge-m3"   # High quality Multilingual model (Ollama)
FASTTEXT_MODEL_PATH: str = "data/models/lid.176.ftz"

# LLM Settings
DEFAULT_LLM_MODEL: str = "llama3.1"
DEFAULT_TEMPERATURE: float = 0.3
DEFAULT_MAX_TOKENS: int = 500

# Retrieval Settings
DEFAULT_RETRIEVAL_K: int = 4

# Memory Settings
DEFAULT_MEMORY_TYPE: str = "buffer"  # or "window"
DEFAULT_WINDOW_SIZE: int = 5

# Streamlit Settings
STREAMLIT_PAGE_TITLE: str = "Document Q&A Chatbot"
STREAMLIT_PAGE_ICON: str = "📚"
STREAMLIT_LAYOUT: str = "wide"

# Supported file types
SUPPORTED_FILE_TYPES: List[str] = ["pdf", "txt", "md"]



# Create directories if they don't exist
for directory in [DATA_DIR, DOCUMENTS_DIR, VECTOR_STORES_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
