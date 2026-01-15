from .celery_config import celery_app
from src.chatbot.core.document_processor import DocumentProcessor
from src.chatbot.core.vector_store_manager import VectorStoreManager
import os

@celery_app.task(bind=True)
def process_document_task(self, file_path: str, api_key: str, embedding_type: str):
    """
    Celery task to process a document.
    """
    try:
        self.update_state(state='PROGRESS', meta={'status': 'Initializing...'})
        
        # 1. Initialize Vector Manager
        # Note: We pass None for event_bus to avoid overhead, relying on Celery state for feedback
        # 1. Initialize Vector Manager
        # Note: We pass None for event_bus to avoid overhead, relying on Celery state for feedback
        # Enforcing local embeddings
        vector_manager = VectorStoreManager(
            model_name="all-MiniLM-L6-v2"
        )

        # 2. Check Cache
        self.update_state(state='PROGRESS', meta={'status': 'Checking cache...'})
        file_hash = vector_manager.get_file_hash(file_path)
        cache_path = f"data/vector_stores/{file_hash}"
        
        if os.path.exists(cache_path):
            return {"status": "cached", "file_hash": file_hash, "message": "Loaded from cache"}

        # 3. Process
        self.update_state(state='PROGRESS', meta={'status': 'Chunking document...'})
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        chunks = processor.process_document(file_path)
        
        # 4. Create Vector Store
        self.update_state(state='PROGRESS', meta={'status': f'Embedding {len(chunks)} chunks...'})
        vector_manager.create_vector_store(chunks, cache_key=file_hash)
        
        return {
            "status": "completed", 
            "chunks": len(chunks), 
            "file_hash": file_hash,
            "message": "Processing complete"
        }
        
    except Exception as e:
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        raise e
