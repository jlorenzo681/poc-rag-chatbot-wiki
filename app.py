"""
Streamlit Web Interface for Document Q&A Chatbot
A RAG-based chatbot with interactive document upload and chat interface.
"""

import os
import re
import sys
import time
from typing import Literal, List, Dict, Any, Optional

import streamlit as st

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import DOCUMENTS_DIR
# DocumentProcessor is now used by backend only
from src.chatbot.core.rag_chain import RAGChain, RAGChatbot
from src.chatbot.core.vector_store_manager import VectorStoreManager
from src.chatbot.core.event_bus import EventBus, Event, DocumentUploadEvent, ProcessingCompleteEvent, ChatResponseEvent, ErrorEvent

import requests
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

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





def process_document(uploaded_file, api_key: str) -> tuple[bool, Optional[str]]:
    """
    Process uploaded document via Backend API.
    
    Args:
        uploaded_file: Streamlit file object
        api_key: API key
        
    Returns:
        Tuple (Success, File Hash)
    """
    # Hardcoded to local embeddings
    embedding_type = "huggingface"
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    data = {"api_key": api_key, "embedding_type": embedding_type}
    
    try:
        # 1. Upload and Start Task
        with st.spinner("🚀 Uploading to backend..."):
            response = requests.post(f"{BACKEND_URL}/upload", files=files, data=data)
            response.raise_for_status()
            task_info = response.json()
            task_id = task_info["task_id"]
            
        # 2. Poll Status
        progress_bar = st.progress(0, text="Starting processing...")
        status = "PENDING"
        
        while status not in ["SUCCESS", "FAILURE"]:
            time.sleep(0.5)
            try:
                res = requests.get(f"{BACKEND_URL}/tasks/{task_id}")
                res.raise_for_status()
                task_status = res.json()
                status = task_status["status"]
                
                if status == "PROGRESS":
                    info = task_status.get("result", {})
                    msg = info.get("status", "Processing...")
                    progress_bar.progress(50, text=f"⏳ {msg}")
            except Exception as e:
                # Tolerate transient polling errors
                print(f"Polling error: {e}")
                continue
            
        if status == "SUCCESS":
            progress_bar.progress(100, text="✅ Processing complete!")
            time.sleep(0.5) 
            progress_bar.empty()
            
            result = task_status["result"]
            file_hash = result.get("file_hash")
            
            # 3. Load Vector Store Locally (Read-Only)
            # We need to initialize the manager to access the cached store
            with st.spinner("📦 Loading vector store..."):
                vector_manager = VectorStoreManager(
                    model_name="all-MiniLM-L6-v2",
                    event_bus=st.session_state.event_bus
                )
                
                cache_path = f"data/vector_stores/{file_hash}"
                vector_manager.load_vector_store(cache_path)
                st.session_state.vector_store_manager = vector_manager
                
            return True, file_hash
            
        else:
            error_msg = task_status.get("error", "Unknown error")
            st.error(f"Task failed: {error_msg}")
            return False, None

    except Exception as e:
        st.error(f"Backend communication error: {str(e)}")
        return False, None


LLMProvider = Literal["Ollama", "LM Studio"]
LLMProviderLower = Literal["ollama", "lmstudio"]

PROVIDER_MAP: dict[LLMProvider, LLMProviderLower] = {
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
            ["Ollama", "LM Studio"],
            help="Ollama: Local (containerized), LM Studio: Local (GPU accelerated)",
        )

        st.divider()

        # Provider-specific configuration
        ollama_url = ""
        lmstudio_url = ""

        if llm_provider == "Ollama":
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
            st.text_input(
                "Ollama Server URL",
                value=ollama_url,
                disabled=True,
                help="Ollama server URL (from environment)",
            )

            model_name = st.selectbox(
                "Model",
                ["llama3.2:3b", "llama3.2:1b"],
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
            llm_provider == "Ollama"
            or (llm_provider == "LM Studio" and lmstudio_url)
        )

        if uploaded_file and can_process:
            if st.button(
                "🚀 Process Document",
                use_container_width=True,
                disabled=st.session_state.document_processed,
            ):
                # Process document via API (No local save needed here, API handles it)
                success, file_hash = process_document(
                    uploaded_file, ""
                )

                if success:
                    # Initialize chatbot
                    initialize_chatbot(
                        llm_provider,
                        "",
                        ollama_url,
                        lmstudio_url,
                        model_name,
                        temperature,
                    )
                    st.session_state.document_processed = True

                    # Note: File is kept in data/documents/ directory for reference

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
        st.caption("1. Upload a document")
        st.caption("2. Click 'Process Document'")
        st.caption("3. Start asking questions!")

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
            - **Flexible Models**: Use local LLMs via Ollama or LM Studio
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
