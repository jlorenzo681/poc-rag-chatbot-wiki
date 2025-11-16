# Document Q&A Chatbot (RAG System)

A production-ready Retrieval-Augmented Generation (RAG) chatbot that enables intelligent Q&A over your documents using LangChain, Groq, and Streamlit.

## Features

- **Multiple Document Formats**: Support for PDF, TXT, and Markdown files
- **Conversational Memory**: Multi-turn conversations with context retention
- **Source Citations**: View which document chunks informed each answer
- **Flexible Embeddings**: Choose between OpenAI embeddings or free local HuggingFace models
- **Interactive Web Interface**: Built with Streamlit for easy deployment
- **Modular Architecture**: Clean separation of concerns for easy customization

## Architecture

The system consists of four main components:

1. **Document Processor** (`document_processor.py`): Loads and chunks documents into manageable pieces
2. **Vector Store Manager** (`vector_store_manager.py`): Creates and manages embeddings and vector search
3. **RAG Chain** (`rag_chain.py`): Implements retrieval and generation logic with conversation memory
4. **Web Interface** (`app.py`): Streamlit-based user interface

## Installation

### Prerequisites

- Python 3.8 or higher
- Groq API key (get it from https://console.groq.com/keys)
- Optional: OpenAI API key (only for OpenAI embeddings)
- Optional: Podman (for containerized deployment)

### Quick Start

#### Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd poc-rag-chatbot-wiki
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file:
```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=your-api-key-here
```

#### Containerized Deployment (Podman)

For production deployment with Podman:

```bash
# 1. Set up environment
cp .env.example .env
nano .env  # Add your GROQ_API_KEY

# 2. Deploy with one command
make deploy
```

**See [DEPLOYMENT.md](DEPLOYMENT.md) for the complete deployment guide.**

## Usage

### Running the Web Interface

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Web Interface

1. **Enter API Key**: Paste your Groq API key in the sidebar
2. **Upload Document**: Choose a PDF, TXT, or Markdown file
3. **Configure Settings**: Select your preferred model and temperature
4. **Process Document**: Click the "Process Document" button
5. **Ask Questions**: Start chatting with your document!

### Using the CLI Example

For command-line usage, see `example_usage.py`:

```bash
python example_usage.py
```

## Configuration Options

### Embedding Models

- **OpenAI** (`text-embedding-3-small`): High quality, requires API key
- **HuggingFace** (`all-MiniLM-L6-v2`): Free, runs locally, lower quality


### Parameters

- **Chunk Size**: 1000 characters (adjustable in `document_processor.py`)
- **Chunk Overlap**: 200 characters (preserves context at boundaries)
- **Retrieval K**: 4 documents (top results to retrieve)
- **Temperature**: 0.3 (lower = more focused, higher = more creative)

## Module Documentation

### DocumentProcessor

Handles document loading and text chunking:

```python
from document_processor import DocumentProcessor

processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
chunks = processor.process_document("path/to/document.pdf")
```

**Supported formats**: PDF, TXT, MD, URLs

### VectorStoreManager

Manages embeddings and vector storage:

```python
from vector_store_manager import VectorStoreManager

# Using OpenAI embeddings
manager = VectorStoreManager(
    embedding_type="openai",
    openai_api_key="your-key"
)

# Or using free HuggingFace embeddings
manager = VectorStoreManager(
    embedding_type="huggingface",
    model_name="all-MiniLM-L6-v2"
)

# Create and save vector store
manager.create_vector_store(chunks)
manager.save_vector_store("faiss_index")
```

### RAGChain

Implements retrieval and generation:

```python
from rag_chain import RAGChain, RAGChatbot

# Create RAG chain
rag_chain = RAGChain(
    retriever=retriever,
    groq_api_key="your-key",
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)

# Create conversational chain with memory
chain = rag_chain.create_conversational_chain(memory_type="buffer")

# Create chatbot interface
chatbot = RAGChatbot(chain, return_sources=True)
response = chatbot.ask("What is this document about?")
```

## Example Use Cases

### Research Assistant
Upload academic papers and ask questions about methodology, results, and conclusions.

### Technical Documentation Helper
Load API documentation or technical manuals and query for specific implementation details.

### Legal Document Analysis
Upload contracts or legal documents and extract key terms and clauses.

### Coffee Equipment Knowledge Base
*(As mentioned in the tutorial)*
Upload all your coffee equipment specs, brewing guides, and recipes, then ask:
- "What's the optimal brew ratio for my Comandante C40?"
- "Compare thermal properties of different drippers"
- "What water temperature should I use with my grinder?"

## Advanced Features

### Custom System Prompts

Modify the system prompt in `rag_chain.py` or pass a custom one:

```python
custom_prompt = """You are an expert coffee consultant...
Context: {context}"""

rag_chain = RAGChain(
    retriever=retriever,
    groq_api_key=api_key,
    system_prompt=custom_prompt
)
```

### Different Memory Types

```python
# Buffer memory (full history)
chain = rag_chain.create_conversational_chain(memory_type="buffer")

# Window memory (last N turns only)
chain = rag_chain.create_conversational_chain(
    memory_type="window",
    window_size=5
)
```

### Loading Existing Vector Stores

```python
# For OpenAI embeddings (if used)
manager = VectorStoreManager(embedding_type="openai", openai_api_key=key)
manager.load_vector_store("faiss_index")

# For HuggingFace embeddings (free)
manager = VectorStoreManager(embedding_type="huggingface")
manager.load_vector_store("faiss_index")
```

## Project Structure

```
poc-rag-chatbot-wiki/
├── src/                       # Source code package
│   └── chatbot/              # Main chatbot package
│       ├── core/             # Core functionality modules
│       │   ├── __init__.py
│       │   ├── document_processor.py     # Document loading and chunking
│       │   ├── vector_store_manager.py   # Embeddings and vector storage
│       │   └── rag_chain.py             # RAG chain implementation
│       ├── utils/            # Utility functions
│       │   └── __init__.py
│       └── __init__.py
├── config/                   # Configuration files
│   ├── __init__.py
│   └── settings.py          # Application settings and constants
├── data/                    # Data directory (gitignored)
│   ├── documents/          # Uploaded documents storage
│   └── vector_stores/      # Saved vector store indices
├── logs/                   # Application logs (gitignored)
├── tests/                  # Test files (future)
├── app.py                 # Streamlit web interface
├── example_usage.py       # CLI usage examples
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## Troubleshooting

### Common Issues

**API Key Errors**
- Ensure your Groq API key is valid and active
- Get your free Groq API key at https://console.groq.com/keys

**Memory Issues with Large Documents**
- Reduce chunk size
- Use HuggingFace embeddings instead of OpenAI
- Process documents in smaller batches

**Slow Performance**
- Use `llama3-8b-8192` for faster responses
- Reduce the retrieval K value
- Use HuggingFace embeddings for local, free processing

## Performance Tips

1. **Chunking Strategy**: The default 1000/200 split balances quality and performance
2. **Retrieval K**: 4 documents is usually optimal; more isn't always better
3. **Temperature**: Use 0.1-0.3 for factual Q&A, 0.5-0.7 for creative tasks
4. **Embeddings**: OpenAI embeddings are higher quality but cost per token; HuggingFace is free but slower on CPU

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Powered by [Groq](https://groq.com/) for ultra-fast LLM inference
- UI created with [Streamlit](https://streamlit.io/)
- Vector search using [FAISS](https://github.com/facebookresearch/faiss)

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Groq Documentation](https://console.groq.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

## Contact

For questions or feedback, please open an issue on GitHub.
