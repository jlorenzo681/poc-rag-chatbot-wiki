# Project Structure Documentation

## Directory Layout

### Current Structure (After Reorganization)

```
poc-rag-chatbot-wiki/
│
├── 📁 src/                          # Source code package
│   ├── __init__.py                 # Package root
│   └── 📁 chatbot/                 # Main chatbot package
│       ├── __init__.py             # Exports: DocumentProcessor, VectorStoreManager, RAGChain, RAGChatbot
│       ├── 📁 core/                # Core functionality modules
│       │   ├── __init__.py
│       │   ├── document_processor.py      # Document loading and text chunking
│       │   ├── vector_store_manager.py    # Embeddings and FAISS vector storage
│       │   └── rag_chain.py              # RAG implementation with memory
│       └── 📁 utils/               # Utility functions (extensible)
│           └── __init__.py
│
├── 📁 config/                      # Configuration management
│   ├── __init__.py                # Config exports
│   └── settings.py                # Application settings, constants, paths
│
├── 📁 data/                        # Runtime data (gitignored except .gitkeep)
│   ├── README.md                  # Data directory documentation
│   ├── 📁 documents/              # Uploaded documents storage
│   │   └── .gitkeep
│   └── 📁 vector_stores/          # Saved FAISS indices
│       └── .gitkeep
│
├── 📁 logs/                        # Application logs (gitignored)
│   └── .gitkeep
│
├── 📁 tests/                       # Test files (ready for pytest)
│
├── 📄 app.py                      # Streamlit web interface
├── 📄 example_usage.py            # CLI usage examples
├── 📄 requirements.txt            # Python dependencies
├── 📄 .env.example               # Environment variables template
├── 📄 .env                       # Actual environment variables (gitignored)
├── 📄 .gitignore                 # Git ignore rules
├── 📄 README.md                  # Main documentation
├── 📄 SETUP.md                   # Setup and installation guide
└── 📄 STRUCTURE.md               # This file

```

## Module Organization

### Core Modules (`src/chatbot/core/`)

#### document_processor.py
- **Purpose**: Load and process documents into chunks
- **Key Classes**: `DocumentProcessor`
- **Dependencies**: langchain, PyPDF2
- **Features**: PDF, TXT, MD support with configurable chunking

#### vector_store_manager.py
- **Purpose**: Manage embeddings and vector storage
- **Key Classes**: `VectorStoreManager`
- **Dependencies**: langchain, faiss, sentence-transformers, openai
- **Features**:
  - OpenAI or HuggingFace embeddings
  - FAISS vector store
  - Save/load functionality

#### rag_chain.py
- **Purpose**: Implement RAG with conversation memory
- **Key Classes**: `RAGChain`, `RAGChatbot`
- **Dependencies**: langchain, groq
- **Features**:
  - Retrieval-augmented generation
  - Conversation memory (buffer/window)
  - Source citations

### Configuration (`config/`)

#### settings.py
- **Purpose**: Centralized configuration
- **Exports**: All configuration constants
- **Features**:
  - Environment variable loading
  - Path management
  - Default settings
  - Directory creation

## Import Patterns

### Old Pattern (Before Restructuring)
```python
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from rag_chain import RAGChain, RAGChatbot
```

### New Pattern (After Restructuring)
```python
from src.chatbot.core.document_processor import DocumentProcessor
from src.chatbot.core.vector_store_manager import VectorStoreManager
from src.chatbot.core.rag_chain import RAGChain, RAGChatbot

# Or use package-level imports
from src.chatbot import DocumentProcessor, VectorStoreManager, RAGChain, RAGChatbot
```

### Using Configuration
```python
from config import (
    GROQ_API_KEY,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DOCUMENTS_DIR,
    VECTOR_STORES_DIR
)
```

## Data Flow

```
1. Document Upload
   └─> data/documents/
       └─> DocumentProcessor (src/chatbot/core/document_processor.py)
           └─> Text Chunks

2. Embedding & Vectorization
   └─> VectorStoreManager (src/chatbot/core/vector_store_manager.py)
       └─> FAISS Index
           └─> data/vector_stores/ (saved for reuse)

3. Query Processing
   └─> RAGChain (src/chatbot/core/rag_chain.py)
       ├─> Retrieve relevant chunks
       ├─> Generate response with Groq LLM
       └─> Return answer + sources

4. Web Interface
   └─> app.py (Streamlit)
       └─> Orchestrates all components
```

## Benefits of New Structure

### 1. **Modularity**
   - Clear separation of concerns
   - Easy to locate and modify specific functionality
   - Reusable components

### 2. **Scalability**
   - Easy to add new modules in `core/` or `utils/`
   - Structured for future growth
   - Test-ready architecture

### 3. **Configuration Management**
   - Centralized settings in `config/`
   - Easy to modify defaults
   - Environment-aware setup

### 4. **Data Organization**
   - Separate directories for different data types
   - Clear gitignore strategy
   - Production-ready data handling

### 5. **Professional Standards**
   - Follows Python package conventions
   - Proper `__init__.py` files
   - Import hierarchy
   - Documentation structure

## File Purposes

| File/Directory | Purpose |
|----------------|---------|
| `src/chatbot/core/` | Core business logic |
| `src/chatbot/utils/` | Helper functions (extensible) |
| `config/` | Configuration and settings |
| `data/documents/` | Temporary document storage |
| `data/vector_stores/` | Persistent vector indices |
| `logs/` | Application logs |
| `tests/` | Unit and integration tests |
| `app.py` | Main web interface entry point |
| `example_usage.py` | CLI examples and demos |
| `README.md` | Project overview and documentation |
| `SETUP.md` | Installation and setup instructions |
| `STRUCTURE.md` | This architecture documentation |

## Future Expansion Ideas

### Potential Additions

```
src/chatbot/
├── core/                    # Existing core modules
├── utils/                   # Utilities (can add)
│   ├── file_utils.py       # File handling helpers
│   ├── logging_utils.py    # Logging configuration
│   └── validation.py       # Input validation
├── api/                     # Future API layer
│   ├── routes.py           # API endpoints
│   └── schemas.py          # Request/response schemas
└── models/                  # Future custom models
    └── custom_embeddings.py # Custom embedding implementations
```

## Getting Started

1. **Installation**: See [SETUP.md](SETUP.md)
2. **Usage**: See [README.md](README.md)
3. **Examples**: See [example_usage.py](example_usage.py)
4. **Configuration**: Edit [config/settings.py](config/settings.py)

## Development Guidelines

### Adding a New Module

1. Place in appropriate directory (`core/` for main logic, `utils/` for helpers)
2. Update corresponding `__init__.py` to export classes/functions
3. Import from package level in your code
4. Add tests in `tests/` directory

### Modifying Configuration

1. Edit `config/settings.py` for defaults
2. Use `.env` for sensitive data
3. Never commit `.env` to git

### Data Management

1. Use `data/documents/` for temporary uploads
2. Save vector stores to `data/vector_stores/`
3. Both directories are gitignored
4. Use `.gitkeep` files to preserve structure
