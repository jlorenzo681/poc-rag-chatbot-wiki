"""
Streamlit Web Interface for Document Q&A Chatbot
A RAG-based chatbot with interactive document upload and chat interface.
"""

import os
import re
import sys
from typing import Literal, List, Dict, Any, Optional

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import DOCUMENTS_DIR
from src.chatbot.core.document_processor import DocumentProcessor
from src.chatbot.core.rag_chain import RAGChain, RAGChatbot
from src.chatbot.core.vector_store_manager import VectorStoreManager
from src.chatbot.core.event_bus import EventBus, Event, DocumentUploadEvent, ProcessingCompleteEvent, ChatResponseEvent, ErrorEvent

# Page configuration
st.set_page_config(
    page_title="Document Q&A Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)


def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = None
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "last_uploaded_filename" not in st.session_state:
        st.session_state.last_uploaded_filename = None
    
    if "event_bus" not in st.session_state:
        st.session_state.event_bus = EventBus()
        # Register a simple console logger for demonstration
        def log_event(event: Event):
            print(f"🔔 EVENT: {type(event).__name__}: {event}")
            
        st.session_state.event_bus.subscribe(Event, log_event)
        st.session_state.event_bus.subscribe(DocumentUploadEvent, log_event)
        st.session_state.event_bus.subscribe(ProcessingCompleteEvent, log_event)
        st.session_state.event_bus.subscribe(ChatResponseEvent, log_event)
        st.session_state.event_bus.subscribe(ErrorEvent, log_event)


def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to documents directory with sanitized filename.
    Files with the same name will be overwritten, allowing cache reuse.

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        Path to saved file
    """
    # Ensure documents directory exists
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Sanitize filename: remove special characters, keep alphanumeric, dots, hyphens, underscores
    original_name = uploaded_file.name
    name_parts = original_name.rsplit(".", 1)
    base_name = name_parts[0]
    extension = name_parts[1] if len(name_parts) > 1 else ""

    # Remove or replace invalid characters
    sanitized_base = re.sub(r"[^\w\s\-\.]", "_", base_name)
    # Replace multiple spaces/underscores with single underscore
    sanitized_base = re.sub(r"[\s_]+", "_", sanitized_base)
    # Remove leading/trailing underscores
    sanitized_base = sanitized_base.strip("_")

    # Construct final filename without timestamp
    if sanitized_base:
        final_filename = (
            f"{sanitized_base}.{extension}" if extension else sanitized_base
        )
    else:
        final_filename = f"document.{extension}" if extension else "document"

    file_path = DOCUMENTS_DIR / final_filename

    # Overwrite if file already exists
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Emit upload event
    if "event_bus" in st.session_state:
        st.session_state.event_bus.publish(DocumentUploadEvent(
            filename=original_name,
            file_path=str(file_path)
        ))

    return str(file_path)


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
                    model_name="text-embedding-3-small",
                    event_bus=st.session_state.event_bus
                )
            else:  # HuggingFace
                vector_manager = VectorStoreManager(
                    embedding_type="huggingface", 
                    model_name="all-MiniLM-L6-v2",
                    event_bus=st.session_state.event_bus
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
        processor = DocumentProcessor(
            chunk_size=1000, 
            chunk_overlap=200,
            event_bus=st.session_state.event_bus
        )

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


LLMProvider = Literal["Groq", "Ollama", "LM Studio"]
LLMProviderLower = Literal["groq", "ollama", "lmstudio"]

PROVIDER_MAP: dict[LLMProvider, LLMProviderLower] = {
    "Groq": "groq",
    "Ollama": "ollama",
    "LM Studio": "lmstudio",
}


def initialize_chatbot(
    llm_provider: LLMProvider,
    groq_api_key: str,
    ollama_url: str,
    lmstudio_url: str,
    model_name: str,
    temperature: float,
):
    """
    Initialize the RAG chatbot.

    Args:
        llm_provider: LLM provider ('Groq', 'Ollama', or 'LM Studio')
        groq_api_key: Groq API key (if using Groq)
        ollama_url: Ollama server URL (if using Ollama)
        lmstudio_url: LM Studio server URL (if using LM Studio)
        model_name: LLM model name
        temperature: Temperature for generation
    """
    try:
        # Get retriever
        retriever = st.session_state.vector_store_manager.get_retriever(k=4)

        # Create RAG chain
        with st.spinner("🤖 Initializing chatbot..."):
            provider_lower = PROVIDER_MAP[llm_provider]

            rag_chain = RAGChain(
                retriever=retriever,
                llm_provider=provider_lower,
                groq_api_key=groq_api_key if provider_lower == "groq" else None,
                ollama_base_url=ollama_url,
                lmstudio_base_url=lmstudio_url,
                model_name=model_name,
                temperature=temperature,
                max_tokens=500,
            )

            # Create conversational chain
            conversational_chain = rag_chain.create_conversational_chain(
                memory_type="buffer"
            )

            # Create chatbot
            chatbot = RAGChatbot(
                chain=conversational_chain, 
                return_sources=True,
                event_bus=st.session_state.event_bus,
                event_metadata={
                    "llm_provider": llm_provider,
                    "model_name": model_name
                }
            )

            st.session_state.chatbot = chatbot
            st.success("✓ Chatbot ready!")

    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")


def display_chat_message(role: str, content: str, sources: Optional[List[Dict[str, Any]]] = None) -> None:
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
                    if source["metadata"]:
                        st.caption(f"Metadata: {source['metadata']}")
                    st.divider()


def main() -> None:
    """Main application function."""
    initialize_session_state()

    # Header
    st.markdown(
        '<p class="main-header">📚 Document Q&A Chatbot</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Upload documents and ask questions using RAG technology</p>',
        unsafe_allow_html=True,
    )

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # LLM Provider Selection
        st.subheader("LLM Provider")
        llm_provider = st.radio(
            "Choose Provider",
            ["Groq", "Ollama", "LM Studio"],
            help="Groq: Cloud API (fast), Ollama: Local (containerized), LM Studio: Local (GPU accelerated)",
        )

        st.divider()

        # Provider-specific configuration
        ollama_url = ""
        lmstudio_url = ""

        if llm_provider == "Groq":
            env_api_key = os.getenv("GROQ_API_KEY", "")
            if env_api_key:
                api_key = env_api_key
                st.success("✓ Using GROQ_API_KEY from environment")
            else:
                api_key = st.text_input(
                    "Groq API Key", type="password", help="Enter your Groq API key"
                )

            model_name = st.selectbox(
                "Model",
                ["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
                help="Select the Groq model to use",
            )
        elif llm_provider == "Ollama":
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
            st.text_input(
                "Ollama Server URL",
                value=ollama_url,
                disabled=True,
                help="Ollama server URL (from environment)",
            )

            model_name = st.selectbox(
                "Model",
                ["llama3.2:3b"],
                help="Select the Ollama model (pull it first if needed)",
            )
            api_key = ""
        else:  # LM Studio
            st.warning(
                "⚠️ Make sure LM Studio is running in server mode (Local Server tab)"
            )
            lmstudio_url = os.getenv(
                "LMSTUDIO_BASE_URL", "http://host.docker.internal:1234/v1"
            )
            lmstudio_url = st.text_input(
                "LM Studio Server URL",
                value=lmstudio_url,
                help="LM Studio server URL (default: http://host.docker.internal:1234/v1)",
            )

            model_name = st.text_input(
                "Model Name",
                value="local-model",
                help="Enter the model name loaded in LM Studio (or use 'local-model')",
            )
            api_key = ""

        st.divider()

        # Embedding options
        st.subheader("Embeddings")
        embedding_type = st.radio(
            "Embedding Model",
            ["HuggingFace (Free)", "OpenAI"],
            index=0,
            help="HuggingFace runs locally, OpenAI requires API key",
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
            help="Lower = more focused, Higher = more creative",
        )

        st.divider()

        # Document upload
        st.subheader("📁 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF or Text File",
            type=["pdf", "txt", "md"],
            help="Upload a document to create a knowledge base",
        )

        # Check if file has changed
        if uploaded_file:
            if uploaded_file.name != st.session_state.last_uploaded_filename:
                # File changed - reset state
                st.session_state.document_processed = False
                st.session_state.chatbot = None
                st.session_state.vector_store_manager = None
                st.session_state.messages = []
                st.session_state.last_uploaded_filename = uploaded_file.name
        else:
            # File removed
            if st.session_state.last_uploaded_filename is not None:
                st.session_state.document_processed = False
                st.session_state.chatbot = None
                st.session_state.vector_store_manager = None
                st.session_state.messages = []
                st.session_state.last_uploaded_filename = None

        # Process button
        can_process = uploaded_file and (
            (llm_provider == "Groq" and api_key)
            or llm_provider == "Ollama"
            or (llm_provider == "LM Studio" and lmstudio_url)
        )

        if uploaded_file and can_process:
            if st.button(
                "🚀 Process Document",
                use_container_width=True,
                disabled=st.session_state.document_processed,
            ):
                # Save uploaded file
                file_path = save_uploaded_file(uploaded_file)

                # Process document
                success = process_document(
                    file_path, api_key if llm_provider == "Groq" else "", embedding_type
                )

                if success:
                    # Initialize chatbot
                    initialize_chatbot(
                        llm_provider,
                        api_key,
                        ollama_url,
                        lmstudio_url,
                        model_name,
                        temperature,
                    )
                    st.session_state.document_processed = True

                    # Note: File is kept in data/documents/ directory for reference

        elif uploaded_file and llm_provider == "Groq" and not api_key:
            st.warning("⚠️ Please enter your Groq API key")
        elif uploaded_file and llm_provider == "LM Studio" and not lmstudio_url:
            st.warning("⚠️ Please enter the LM Studio server URL")

        st.divider()

        # Reset button
        if st.session_state.document_processed:
            if st.button("🔄 Start Over", use_container_width=True):
                st.session_state.document_processed = False
                st.session_state.chatbot = None
                st.session_state.vector_store_manager = None
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
        st.info(
            "👈 Please upload a document and configure settings in the sidebar to get started"
        )

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
                message["role"], message["content"], message.get("sources")
            )

        # Chat input
        if prompt := st.chat_input("Ask a question about your document..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
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
                                if source["metadata"]:
                                    st.caption(f"Metadata: {source['metadata']}")
                                st.divider()

                    # Add to message history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )


if __name__ == "__main__":
    main()
