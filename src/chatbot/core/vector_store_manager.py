"""
Vector Store Manager Module
Handles embeddings generation and vector store operations.
"""

from typing import List, Optional, Literal
from langchain.schema import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import hashlib
from .event_bus import EventBus, VectorStoreUpdateEvent

# Optional: OpenAI embeddings (requires langchain-openai package)
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class VectorStoreManager:
    """
    Manages vector store creation, loading, and retrieval operations.
    """

    def __init__(
        self,
        embedding_type: Literal["openai", "huggingface"] = "huggingface",
        openai_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the vector store manager.

        Args:
            embedding_type: Type of embeddings to use ('openai' or 'huggingface')
            openai_api_key: OpenAI API key (required if embedding_type='openai')
            model_name: Name of the embedding model
        """
        self.embedding_type = embedding_type
        self.embeddings = self._initialize_embeddings(
            embedding_type,
            openai_api_key,
            model_name
        )
        self.vector_store = None
        self.event_bus = event_bus

    def _initialize_embeddings(
        self,
        embedding_type: str,
        api_key: Optional[str],
        model_name: Optional[str]
    ) -> Embeddings:
        """
        Initialize the embeddings model.

        Args:
            embedding_type: Type of embeddings ('openai' or 'huggingface')
            api_key: API key for OpenAI
            model_name: Name of the model

        Returns:
            Embeddings instance
        """
        if embedding_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI embeddings requested but langchain-openai is not installed. "
                    "Install it with: pip install langchain-openai"
                )
            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAI embeddings")

            model = model_name or "text-embedding-3-small"
            print(f"🔧 Initializing OpenAI embeddings (model: {model})")
            return OpenAIEmbeddings(
                model=model,
                openai_api_key=api_key
            )

        elif embedding_type == "huggingface":
            model = model_name or "all-MiniLM-L6-v2"
            print(f"🔧 Initializing HuggingFace embeddings (model: {model})")
            return HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={},
                encode_kwargs={'normalize_embeddings': True}
            )

        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

    def get_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            Hex string of the file hash
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def create_vector_store(self, documents: List[Document], cache_key: Optional[str] = None) -> FAISS:
        """
        Create a new vector store from documents, with optional caching.

        Args:
            documents: List of Document objects to embed
            cache_key: Optional cache key (file hash) to save/load from cache

        Returns:
            FAISS vector store
        """
        # If cache_key provided, check if cached version exists
        if cache_key:
            cache_path = f"data/vector_stores/{cache_key}"
            if os.path.exists(cache_path):
                print(f"\n📦 Loading cached vector store for hash: {cache_key[:12]}...")
                return self.load_vector_store(cache_path)

        print(f"\n🔄 Creating vector store from {len(documents)} documents...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        print("✓ Vector store created successfully")

        # Save to cache if cache_key provided
        if cache_key:
            cache_path = f"data/vector_stores/{cache_key}"
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self.save_vector_store(cache_path)
            print(f"✓ Vector store cached with key: {cache_key[:12]}...")

        
        if self.event_bus:
            self.event_bus.publish(VectorStoreUpdateEvent(
                operation="create",
                document_count=len(documents)
            ))

        return self.vector_store

    def save_vector_store(self, path: str = "faiss_index"):
        """
        Save the vector store to disk.

        Args:
            path: Directory path to save the vector store
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")

        self.vector_store.save_local(path)
        print(f"✓ Vector store saved to: {path}")

    def load_vector_store(self, path: str = "faiss_index") -> FAISS:
        """
        Load a vector store from disk.

        Args:
            path: Directory path to load the vector store from

        Returns:
            FAISS vector store
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at: {path}")

        print(f"📂 Loading vector store from: {path}")
        self.vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✓ Vector store loaded successfully")

        if self.event_bus and self.vector_store:
            # Note: We can't easily distinguish document count on load with FAISS without accessing index
            # accessing self.vector_store.index.ntotal if available
            count = 0
            if hasattr(self.vector_store, 'index'):
                count = self.vector_store.index.ntotal
                
            self.event_bus.publish(VectorStoreUpdateEvent(
                operation="load",
                document_count=count
            ))

        return self.vector_store

    def add_documents(self, documents: List[Document]):
        """
        Add new documents to an existing vector store.

        Args:
            documents: List of Document objects to add
        """
        if self.vector_store is None:
            raise ValueError("No vector store exists. Create one first.")

        print(f"\n➕ Adding {len(documents)} documents to vector store...")
        self.vector_store.add_documents(documents)
        print("✓ Documents added successfully")
        
        if self.event_bus:
            self.event_bus.publish(VectorStoreUpdateEvent(
                operation="add",
                document_count=len(documents)
            ))

    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Query text
            k: Number of documents to return

        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            raise ValueError("No vector store exists. Create or load one first.")

        results = self.vector_store.similarity_search(query, k=k)
        return results

    def get_retriever(self, k: int = 4) -> VectorStoreRetriever:
        """
        Get a retriever for the vector store.

        Args:
            k: Number of documents to retrieve

        Returns:
            Retriever instance
        """
        if self.vector_store is None:
            raise ValueError("No vector store exists. Create or load one first.")

        return self.vector_store.as_retriever(
            search_kwargs={"k": k}
        )


if __name__ == "__main__":
    # Example usage
    print("Vector Store Manager initialized successfully!")
    print("\nSupported embedding types:")
    print("  - openai: text-embedding-3-small (default)")
    print("  - huggingface: all-MiniLM-L6-v2 (default, free)")
