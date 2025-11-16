"""
Streamlit Web Interface for Document Q&A Chatbot
A RAG-based chatbot with interactive document upload and chat interface.
"""

import streamlit as st
import os
import sys
import tempfile

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.chatbot.core.document_processor import DocumentProcessor
from src.chatbot.core.vector_store_manager import VectorStoreManager
from src.chatbot.core.rag_chain import RAGChain, RAGChatbot


# Page configuration
st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = None
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False


def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temporary location.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Path to saved file
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def process_document(file_path: str, api_key: str, embedding_type: str) -> bool:
    """
    Process uploaded document and create vector store.

    Args:
        file_path: Path to document
        api_key: API key for embeddings (if using OpenAI)
        embedding_type: Type of embeddings to use

    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize vector store manager first (needed for hash calculation)
        with st.spinner(f"🔧 Initializing {embedding_type} embeddings..."):
            if embedding_type == "OpenAI":
                vector_manager = VectorStoreManager(
                    embedding_type="openai",
                    openai_api_key=api_key,
                    model_name="text-embedding-3-small"
                )
            else:  # HuggingFace
                vector_manager = VectorStoreManager(
                    embedding_type="huggingface",
                    model_name="all-MiniLM-L6-v2"
                )

        # Calculate file hash for caching
        file_hash = vector_manager.get_file_hash(file_path)
        cache_path = f"data/vector_stores/{file_hash}"

        # Check if cached vector store exists
        if os.path.exists(cache_path):
            with st.spinner("📦 Loading cached vector store..."):
                vector_manager.load_vector_store(cache_path)
                st.success("✓ Loaded from cache (document already processed)")
                st.session_state.vector_store_manager = vector_manager
                return True

        # If not cached, process document
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

        # Process document
        with st.spinner("📄 Loading and chunking document..."):
            chunks = processor.process_document(file_path)
            st.success(f"✓ Created {len(chunks)} chunks")

        # Create vector store with caching
        with st.spinner("🔄 Creating vector store..."):
            vector_manager.create_vector_store(chunks, cache_key=file_hash)
            st.success("✓ Vector store created and cached")

        # Store in session state
        st.session_state.vector_store_manager = vector_manager

        return True

    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return False


def initialize_chatbot(llm_provider: str, groq_api_key: str, ollama_url: str, model_name: str, temperature: float):
    """
    Initialize the RAG chatbot.

    Args:
        llm_provider: LLM provider ('Groq' or 'Ollama')
        groq_api_key: Groq API key (if using Groq)
        ollama_url: Ollama server URL (if using Ollama)
        model_name: LLM model name
        temperature: Temperature for generation
    """
    try:
        # Get retriever
        retriever = st.session_state.vector_store_manager.get_retriever(k=4)

        # Create RAG chain
        with st.spinner("🤖 Initializing chatbot..."):
            provider_lower = llm_provider.lower()

            rag_chain = RAGChain(
                retriever=retriever,
                llm_provider=provider_lower,
                groq_api_key=groq_api_key if provider_lower == "groq" else None,
                ollama_base_url=ollama_url,
                model_name=model_name,
                temperature=temperature,
                max_tokens=500
            )

            # Create conversational chain
            conversational_chain = rag_chain.create_conversational_chain(
                memory_type="buffer"
            )

            # Create chatbot
            chatbot = RAGChatbot(
                chain=conversational_chain,
                return_sources=True
            )

            st.session_state.chatbot = chatbot
            st.success("✓ Chatbot ready!")

    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")


def display_chat_message(role: str, content: str, sources=None):
    """
    Display a chat message with optional sources.

    Args:
        role: Message role (user or assistant)
        content: Message content
        sources: Optional source documents
    """
    with st.chat_message(role):
        st.write(content)

        # Display sources if available
        if sources and role == "assistant":
            with st.expander("📚 View Sources"):
                for source in sources:
                    st.markdown(f"**Source {source['index']}:**")
                    st.markdown(f"```\n{source['content']}\n```")
                    if source['metadata']:
                        st.caption(f"Metadata: {source['metadata']}")
                    st.divider()


def main():
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown('<p class="main-header">📚 Document Q&A Chatbot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload documents and ask questions using RAG technology</p>', unsafe_allow_html=True)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # LLM Provider Selection
        st.subheader("LLM Provider")
        llm_provider = st.radio(
            "Choose Provider",
            ["Groq", "Ollama"],
            help="Groq: Cloud API (fast), Ollama: Local (containerized)"
        )

        st.divider()

        # Provider-specific configuration
        if llm_provider == "Groq":
            env_api_key = os.getenv("GROQ_API_KEY", "")
            if env_api_key:
                api_key = env_api_key
                st.success("✓ Using GROQ_API_KEY from environment")
            else:
                api_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    help="Enter your Groq API key"
                )

            model_name = st.selectbox(
                "Model",
                ["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                help="Select the Groq model to use"
            )
            ollama_url = ""
        else:  # Ollama
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
            st.text_input(
                "Ollama Server URL",
                value=ollama_url,
                disabled=True,
                help="Ollama server URL (from environment)"
            )

            model_name = st.selectbox(
                "Model",
                ["llama3.1:8b", "llama3.1:70b", "mistral:latest", "mixtral:latest"],
                help="Select the Ollama model (pull it first if needed)"
            )
            api_key = ""

        st.divider()

        # Embedding options
        st.subheader("Embeddings")
        embedding_type = st.radio(
            "Embedding Model",
            ["HuggingFace (Free)", "OpenAI"],
            index=0,
            help="HuggingFace runs locally, OpenAI requires API key"
        )

        st.divider()

        # LLM settings
        st.subheader("LLM Settings")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )

        st.divider()

        # Document upload
        st.subheader("📁 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF or Text File",
            type=["pdf", "txt", "md"],
            help="Upload a document to create a knowledge base"
        )

        # Process button
        can_process = uploaded_file and ((llm_provider == "Groq" and api_key) or (llm_provider == "Ollama"))

        if uploaded_file and can_process:
            if st.button(
                "🚀 Process Document",
                use_container_width=True,
                disabled=st.session_state.document_processed
            ):
                # Save uploaded file
                file_path = save_uploaded_file(uploaded_file)

                # Process document
                success = process_document(file_path, api_key if llm_provider == "Groq" else "", embedding_type)

                if success:
                    # Initialize chatbot
                    initialize_chatbot(llm_provider, api_key, ollama_url, model_name, temperature)
                    st.session_state.document_processed = True

                    # Clean up temp file
                    os.unlink(file_path)

        elif uploaded_file and llm_provider == "Groq" and not api_key:
            st.warning("⚠️ Please enter your Groq API key")

        st.divider()

        # Reset button
        if st.session_state.document_processed:
            if st.button("🔄 Reset Chat", use_container_width=True):
                if st.session_state.chatbot:
                    st.session_state.chatbot.reset_conversation()
                st.session_state.messages = []
                st.rerun()

        # Info section
        st.divider()
        st.caption("💡 **How to use:**")
        st.caption("1. Enter your Groq API key")
        st.caption("2. Upload a document")
        st.caption("3. Click 'Process Document'")
        st.caption("4. Start asking questions!")

    # Main chat interface
    if not st.session_state.document_processed:
        # Welcome message
        st.info("👈 Please upload a document and configure settings in the sidebar to get started")

        # Example use cases
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✨ Example Use Cases")
            st.markdown("""
            - **Research Papers**: Ask questions about scientific papers
            - **Technical Documentation**: Query API docs and manuals
            - **Legal Documents**: Extract information from contracts
            - **Books & Articles**: Discuss content and themes
            """)

        with col2:
            st.subheader("🔧 Features")
            st.markdown("""
            - **Conversational Memory**: Multi-turn conversations
            - **Source Citations**: See where answers come from
            - **Multiple Formats**: PDF, TXT, Markdown support
            - **Flexible Models**: Groq LLMs with local/cloud embeddings
            """)

    else:
        # Display chat messages
        for message in st.session_state.messages:
            display_chat_message(
                message["role"],
                message["content"],
                message.get("sources")
            )

        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            display_chat_message("user", prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.ask(prompt)

                    # Display answer
                    answer = response["answer"]
                    st.write(answer)

                    # Display sources
                    sources = response.get("sources", [])
                    if sources:
                        with st.expander("📚 View Sources"):
                            for source in sources:
                                st.markdown(f"**Source {source['index']}:**")
                                st.markdown(f"```\n{source['content']}\n```")
                                if source['metadata']:
                                    st.caption(f"Metadata: {source['metadata']}")
                                st.divider()

                    # Add to message history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })


if __name__ == "__main__":
    main()
