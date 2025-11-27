"""
Example CLI Usage of the RAG Chatbot
Demonstrates how to use the modules programmatically.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.chatbot.core.document_processor import DocumentProcessor
from src.chatbot.core.vector_store_manager import VectorStoreManager
from src.chatbot.core.rag_chain import RAGChain, RAGChatbot


def example_basic_usage() -> None:
    """
    Example 1: Basic document Q&A without saving vector store.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Document Q&A")
    print("=" * 60)

    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️  Please set GROQ_API_KEY environment variable")
        return

    # Step 1: Process document
    print("\n📄 Step 1: Processing document...")
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)

    # You'll need to provide your own document
    # For demo purposes, we'll create a sample text file
    sample_doc = "sample_document.txt"
    with open(sample_doc, "w") as f:
        f.write("""
        Coffee Brewing Guide

        The Comandante C40 is a premium hand grinder known for its consistency.
        For pour-over coffee, use a medium-fine grind setting (around 20-24 clicks).

        Water temperature is crucial for extraction. For light roasts, use 200-205°F.
        For medium roasts, 195-200°F works well. Dark roasts benefit from 185-195°F.

        The optimal brew ratio is typically 1:16 (coffee to water). For example,
        20g of coffee to 320g of water. Adjust based on taste preferences.

        The Graycano dripper features a flat-bottom design that promotes even extraction.
        It works well with both conical and flat-bottom filters.
        """)

    chunks = processor.process_document(sample_doc)

    # Step 2: Create vector store with HuggingFace embeddings (free)
    print("\n🔧 Step 2: Creating vector store...")
    vector_manager = VectorStoreManager(
        embedding_type="huggingface",
        model_name="all-MiniLM-L6-v2"
    )
    vector_manager.create_vector_store(chunks)

    # Step 3: Create RAG chain
    print("\n🤖 Step 3: Initializing chatbot...")
    retriever = vector_manager.get_retriever(k=3)
    rag_chain = RAGChain(
        retriever=retriever,
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.3
    )

    # Create conversational chain
    chain = rag_chain.create_conversational_chain(memory_type="buffer")
    chatbot = RAGChatbot(chain, return_sources=True)

    # Step 4: Ask questions
    print("\n💬 Step 4: Asking questions...")
    questions = [
        "What grind setting should I use for the Comandante C40?",
        "What water temperature is best for light roasts?",
        "What is the optimal brew ratio?"
    ]

    for question in questions:
        print(f"\n❓ Q: {question}")
        response = chatbot.ask(question)
        print(f"🤖 A: {response['answer']}")

        if response.get('sources'):
            print(f"\n📚 Sources: {len(response['sources'])} documents retrieved")

    # Cleanup
    os.unlink(sample_doc)
    print("\n✓ Example completed!")


def example_save_and_load() -> None:
    """
    Example 2: Save vector store and load it later.
    """
    print("\n" + "=" * 60)
    print("Example 2: Save and Load Vector Store")
    print("=" * 60)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⚠️  Please set GROQ_API_KEY environment variable")
        return

    # Create sample document
    sample_doc = "sample_doc_2.txt"
    with open(sample_doc, "w") as f:
        f.write("""
        Photography Equipment Guide

        The Sony A7IV is a full-frame mirrorless camera with 33MP resolution.
        It excels in both photo and video work, with 4K 60fps capability.

        For portraits, a 85mm f/1.8 lens provides beautiful bokeh and sharpness.
        The compression at 85mm is flattering for facial features.

        Landscape photography benefits from ultra-wide lenses like 16-35mm.
        Shoot at f/8-f/11 for optimal sharpness across the frame.
        """)

    # Process and save
    print("\n📄 Creating and saving vector store...")
    processor = DocumentProcessor()
    chunks = processor.process_document(sample_doc)

    vector_manager = VectorStoreManager(
        embedding_type="huggingface",
        model_name="all-MiniLM-L6-v2"
    )
    vector_manager.create_vector_store(chunks)
    vector_manager.save_vector_store("photography_index")

    print("✓ Vector store saved to 'photography_index'")

    # Later: Load the vector store
    print("\n📂 Loading vector store from disk...")
    new_manager = VectorStoreManager(
        embedding_type="huggingface",
        model_name="all-MiniLM-L6-v2"
    )
    new_manager.load_vector_store("photography_index")

    # Use the loaded vector store
    retriever = new_manager.get_retriever(k=2)
    rag_chain = RAGChain(
        retriever=retriever,
        groq_api_key=api_key,
        model_name="llama-3.1-8b-instant"
    )

    chain = rag_chain.create_basic_chain()
    chatbot = RAGChatbot(chain, return_sources=False)

    question = "What resolution does the Sony A7IV have?"
    print(f"\n❓ Q: {question}")
    response = chatbot.ask(question)
    print(f"🤖 A: {response['answer']}")

    # Cleanup
    os.unlink(sample_doc)
    import shutil
    shutil.rmtree("photography_index")
    print("\n✓ Example completed!")


def example_with_openai_embeddings() -> None:
    """
    Example 3: Using OpenAI embeddings for higher quality (still requires OpenAI key for embeddings).
    """
    print("\n" + "=" * 60)
    print("Example 3: OpenAI Embeddings (Higher Quality)")
    print("=" * 60)

    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not groq_api_key:
        print("⚠️  Please set GROQ_API_KEY environment variable")
        return
    if not openai_api_key:
        print("⚠️  Please set OPENAI_API_KEY environment variable for embeddings")
        return

    # Create sample document
    sample_doc = "sample_doc_3.txt"
    with open(sample_doc, "w") as f:
        f.write("""
        Machine Learning Concepts

        Neural networks are computational models inspired by biological neurons.
        They consist of layers: input layer, hidden layers, and output layer.

        Backpropagation is the key algorithm for training neural networks.
        It calculates gradients using the chain rule from calculus.

        Overfitting occurs when a model learns training data too well,
        including noise and outliers. Regularization techniques help prevent this.
        """)

    print("\n📄 Processing document...")
    processor = DocumentProcessor()
    chunks = processor.process_document(sample_doc)

    # Use OpenAI embeddings
    print("\n🔧 Creating vector store with OpenAI embeddings...")
    vector_manager = VectorStoreManager(
        embedding_type="openai",
        openai_api_key=openai_api_key,
        model_name="text-embedding-3-small"
    )
    vector_manager.create_vector_store(chunks)

    # Create chatbot
    retriever = vector_manager.get_retriever(k=2)
    rag_chain = RAGChain(
        retriever=retriever,
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    chain = rag_chain.create_conversational_chain(memory_type="window", window_size=3)
    chatbot = RAGChatbot(chain, return_sources=True)

    # Multi-turn conversation
    print("\n💬 Multi-turn conversation:")
    questions = [
        "What is backpropagation?",
        "How does it relate to neural networks?",
        "What's the solution to overfitting?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n🔄 Turn {i}")
        print(f"❓ Q: {question}")
        response = chatbot.ask(question)
        print(f"🤖 A: {response['answer']}")

    # Cleanup
    os.unlink(sample_doc)
    print("\n✓ Example completed!")


def main() -> None:
    """
    Run all examples.
    """
    print("\n" + "=" * 60)
    print("RAG Chatbot - Example Usage Scripts")
    print("=" * 60)

    print("\nThis script demonstrates three usage patterns:")
    print("1. Basic usage with HuggingFace embeddings (free)")
    print("2. Saving and loading vector stores")
    print("3. Using OpenAI embeddings for higher quality")

    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n⚠️  Warning: GROQ_API_KEY not found in environment")
        print("Please set it with: export GROQ_API_KEY='your-key-here'")
        print("Or create a .env file with GROQ_API_KEY=your-key-here")
        return

    # Run examples
    try:
        example_basic_usage()
        example_save_and_load()
        example_with_openai_embeddings()

        print("\n" + "=" * 60)
        print("All examples completed successfully! 🎉")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
