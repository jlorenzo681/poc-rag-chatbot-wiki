"""
Vector Store Manager Module
Handles embeddings generation and vector store operations.
"""

from typing import List, Optional, Literal, Any
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
import os
import hashlib
import fasttext
import sys
from .event_bus import EventBus, VectorStoreUpdateEvent
import config.settings as settings


class VectorStoreManager:
    """
    Manages vector store creation, loading, and retrieval operations with dynamic embedding selection.
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the vector store manager.
        """
        self.vector_store = None
        self.event_bus = event_bus
        self.ft_model = self._load_fasttext_model()
        self.current_embedding_model = None # Track which model is currently loaded

    def _load_fasttext_model(self):
        """Load the FastText language identification model."""
        model_path = settings.FASTTEXT_MODEL_PATH
        if not os.path.exists(model_path):
            print(f"⚠ FastText model not found at {model_path}. Language detection disabled.")
            return None
        
        # Suppress fasttext warning about load_model
        fasttext.FastText.eprint = lambda x: None
        try:
            return fasttext.load_model(str(model_path))
        except Exception as e:
            print(f"❌ Error loading FastText model: {e}")
            return None

    def detect_language(self, text: str) -> str:
        """
        Detect language of the text using FastText.
        
        Args:
            text: Text sample to analyze
            
        Returns:
            Language code (e.g., 'en', 'es') or 'en' if detection fails
        """
        if not self.ft_model:
            return 'en'
            
        try:
            # Clean text specifically for fasttext
            clean_text = text.replace('\n', ' ')[:1000] # Analyze first 1000 chars
            pred = self.ft_model.predict(clean_text, k=1)
            language_code = pred[0][0].replace('__label__', '')
            confidence = pred[1][0]
            print(f"🔍 Language detected: {language_code} ({confidence:.1%})")
            return language_code
        except Exception as e:
            print(f"⚠ Language detection failed: {e}")
            return 'en'

    def _get_embeddings_for_language(self, language_code: str) -> Embeddings:
        """
        Get the appropriate embedding model for the detected language.
        
        Args:
            language_code: Detected language code
            
        Returns:
            Embeddings instance
        """
        if language_code == 'en':
            model_name = settings.EMBEDDING_MODEL_EN
            print(f"🔧 Selecting English embedding model: {model_name}")
        else:
            model_name = settings.EMBEDDING_MODEL_MULTILINGUAL
            print(f"🔧 Selecting Multilingual embedding model ({language_code}): {model_name}")

        return OllamaEmbeddings(
            model=model_name
        )

    def get_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA256 hash of a file.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def create_vector_store(self, documents: List[Document], cache_key: Optional[str] = None) -> FAISS:
        """
        Create a new vector store from documents with dynamic embedding selection.
        """
        # Detect language from a sample of documents
        sample_text = " ".join([d.page_content for d in documents[:5]])
        language = self.detect_language(sample_text)
        
        # Initialize appropriate embeddings
        self.embeddings = self._get_embeddings_for_language(language)

        # If cache_key provided, check if cached version exists
        # NOTE: Using a simple cache key might be risky if we switch models for the same file.
        # Ideally, cache key should include model name, but for now we follow existing pattern.
        if cache_key:
            # Append language/model specific suffix to avoid mixing incompatible embeddings
            cache_path = f"data/vector_stores/{cache_key}_{language}"
            if os.path.exists(cache_path):
                print(f"\n📦 Loading cached vector store: {cache_path}...")
                return self.load_vector_store(cache_path)
            
            # Legacy fallback
            old_cache_path = f"data/vector_stores/{cache_key}"
            if os.path.exists(old_cache_path):
                 print(f"\n⚠ Found legacy cache at {old_cache_path}, but ignoring to ensure correct embedding model.")

        print(f"\n🔄 Creating vector store from {len(documents)} documents...")
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        print("✓ Vector store created successfully")

        # Save to cache if cache_key provided
        if cache_key:
            cache_path = f"data/vector_stores/{cache_key}_{language}"
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            self.save_vector_store(cache_path)
            print(f"✓ Vector store cached at: {cache_path}")

        if self.event_bus:
            self.event_bus.publish(VectorStoreUpdateEvent(
                operation="create",
                document_count=len(documents)
            ))

        return self.vector_store

    def save_vector_store(self, path: str = "faiss_index"):
        """Save the vector store to disk."""
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")

        self.vector_store.save_local(path)
        print(f"✓ Vector store saved to: {path}")

    def load_vector_store(self, path: str = "faiss_index") -> FAISS:
        """
        Load a vector store from disk.
        WARNING: This assumes the currently initialized embeddings match the stored index.
        For dynamic loading, create_vector_store handles the flow better.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at: {path}")

        # If embeddings not set (e.g. manual load), default to English or Multilingual?
        # Ideally, we should detect or store metadata. For now, if not set, default to Multilingual as safest.
        if not hasattr(self, 'embeddings') or self.embeddings is None:
             print("⚠ Embeddings not initialized, defaulting to Multilingual for load.")
             self.embeddings = self._get_embeddings_for_language('es') # Fallback

        print(f"📂 Loading vector store from: {path}")
        self.vector_store = FAISS.load_local(
            path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✓ Vector store loaded successfully")

        if self.event_bus and self.vector_store:
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
    print("  - huggingface: all-MiniLM-L6-v2 (default, free)")
