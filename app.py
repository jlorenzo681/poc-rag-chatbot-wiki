"""
Streamlit Web Interface for Document Q&A Chatbot
A RAG-based chatbot with interactive document upload and chat interface.
"""

import streamlit as st
import os
import tempfile
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from rag_chain import RAGChain, RAGChatbot


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
        api_key: OpenAI API key
        embedding_type: Type of embeddings to use

    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize document processor
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

        # Process document
        with st.spinner("📄 Loading and chunking document..."):
            chunks = processor.process_document(file_path)
            st.success(f"✓ Created {len(chunks)} chunks")

        # Initialize vector store manager
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

        # Create vector store
        with st.spinner("🔄 Creating vector store..."):
            vector_manager.create_vector_store(chunks)
            st.success("✓ Vector store created")

        # Store in session state
        st.session_state.vector_store_manager = vector_manager

        return True

    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return False


def initialize_chatbot(api_key: str, model_name: str, temperature: float):
    """
    Initialize the RAG chatbot.

    Args:
        api_key: OpenAI API key
        model_name: LLM model name
        temperature: Temperature for generation
    """
    try:
        # Get retriever
        retriever = st.session_state.vector_store_manager.get_retriever(k=4)

        # Create RAG chain
        with st.spinner("🤖 Initializing chatbot..."):
            rag_chain = RAGChain(
                retriever=retriever,
                openai_api_key=api_key,
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

        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to use GPT models"
        )

        st.divider()

        # Embedding options
        st.subheader("Embeddings")
        embedding_type = st.radio(
            "Embedding Model",
            ["OpenAI", "HuggingFace (Free)"],
            help="OpenAI requires API key, HuggingFace runs locally"
        )

        st.divider()

        # LLM settings
        st.subheader("LLM Settings")
        model_name = st.selectbox(
            "Model",
            ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            help="Select the GPT model to use"
        )

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
        if uploaded_file and api_key:
            if st.button("🚀 Process Document", use_container_width=True):
                # Save uploaded file
                file_path = save_uploaded_file(uploaded_file)

                # Process document
                success = process_document(file_path, api_key, embedding_type)

                if success:
                    # Initialize chatbot
                    initialize_chatbot(api_key, model_name, temperature)
                    st.session_state.document_processed = True

                    # Clean up temp file
                    os.unlink(file_path)

        elif uploaded_file and not api_key and embedding_type == "OpenAI":
            st.warning("⚠️ Please enter your OpenAI API key")

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
        st.caption("1. Enter your OpenAI API key")
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
            - **Flexible Models**: OpenAI or local embeddings
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
