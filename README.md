# Document Q&A Chatbot (RAG System)

A production-ready Retrieval-Augmented Generation (RAG) chatbot that enables intelligent Q&A over your documents using LangChain, OpenAI, and Streamlit.

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
- OpenAI API key (for GPT models and OpenAI embeddings)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd poc-rag-chatbot-wiki
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file (optional):
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

### Running the Web Interface

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Web Interface

1. **Enter API Key**: Paste your OpenAI API key in the sidebar
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

### LLM Models

- `gpt-4-turbo`: Most capable, higher cost
- `gpt-4`: Very capable, moderate cost
- `gpt-3.5-turbo`: Fast and economical

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
    openai_api_key="your-key",
    model_name="gpt-4-turbo",
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
    openai_api_key=api_key,
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
manager = VectorStoreManager(embedding_type="openai", openai_api_key=key)
manager.load_vector_store("faiss_index")
```

## Project Structure

```
poc-rag-chatbot-wiki/
├── app.py                      # Streamlit web interface
├── document_processor.py       # Document loading and chunking
├── vector_store_manager.py     # Embeddings and vector storage
├── rag_chain.py               # RAG chain implementation
├── example_usage.py           # CLI usage examples
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

## Troubleshooting

### Common Issues

**API Key Errors**
- Ensure your OpenAI API key is valid
- Check that you have sufficient API credits

**Memory Issues with Large Documents**
- Reduce chunk size
- Use HuggingFace embeddings instead of OpenAI
- Process documents in smaller batches

**Slow Performance**
- Use `gpt-3.5-turbo` instead of `gpt-4`
- Reduce the retrieval K value
- Consider using a more powerful machine for local embeddings

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
- Powered by [OpenAI](https://openai.com/)
- UI created with [Streamlit](https://streamlit.io/)
- Vector search using [FAISS](https://github.com/facebookresearch/faiss)

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)

## Contact

For questions or feedback, please open an issue on GitHub.
