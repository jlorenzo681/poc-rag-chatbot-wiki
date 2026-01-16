
import os
from typing import List, Optional
from langchain_community.graphs import Neo4jGraph
from .simple_graph_transformer import SimpleGraphTransformer
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
import config.settings as settings

class GraphStoreManager:
    """
    Manages Neo4j graph database interactions and document transformation.
    """
    def __init__(self):
        self.graph = None
        self.llm_transformer = None
        
        if getattr(settings, "ENABLE_GRAPHRAG", False):
            self._initialize_graph()
            self._initialize_transformer()
        else:
            print("ℹ️ GraphRAG is disabled in settings.")

    def _initialize_graph(self):
        """Initialize connection to Neo4j."""
        try:
            self.graph = Neo4jGraph(
                url=settings.NEO4J_URI,
                username=settings.NEO4J_USERNAME,
                password=settings.NEO4J_PASSWORD
            )
            # Create fulltext index for hybrid search if needed, but langchain might handle it
            print("✓ Connected to Neo4j graph database")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            self.graph = None

    def _initialize_transformer(self):
        """Initialize LLM graph transformer."""
        try:
            # Use Llama 3.1 for extraction
            # Use Llama 3.1 for extraction
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            print(f"DEBUG: OLLAMA_BASE_URL={base_url}")
            print(f"DEBUG: DEFAULT_LLM_MODEL={settings.DEFAULT_LLM_MODEL}")
            
            llm = ChatOllama(
                model=settings.DEFAULT_LLM_MODEL,
                temperature=0, # Deterministic for extraction
                base_url=base_url
            )
            # Use our robust custom transformer
            self.llm_transformer = SimpleGraphTransformer(llm=llm)
            print("✓ Initialized Simple Graph Transformer")
        except Exception as e:
            print(f"❌ Failed to initialize Graph Transformer: {e}")

    def add_documents_to_graph(self, documents: List[Document]):
        """
        Extract entities/relationships and add to Neo4j.
        """
        if not self.graph or not self.llm_transformer:
            if getattr(settings, "ENABLE_GRAPHRAG", False):
                print("⚠ Graph integration failed initialization. Skipping.")
            return

        print(f"\n🕸️ Extracting graph data from {len(documents)} documents...")
        print("  (This requires heavy LLM processing and may take time...)")
        
        # Process documents individually to prevent one failure from stopping the whole batch
        valid_graph_documents = []
        
        from tqdm import tqdm
        print(f"Processing {len(documents)} chunks individually for robustness...")
        
        for i, doc in enumerate(documents):
            try:
                # Process single document
                # Note: convert_to_graph_documents expects a list
                chunk_graph_docs = self.llm_transformer.convert_to_graph_documents([doc])
                if chunk_graph_docs:
                    valid_graph_documents.extend(chunk_graph_docs)
                    print(f"  ✓ Chunk {i+1}/{len(documents)} processed successfully")
                else:
                    print(f"  ⚠ Chunk {i+1}/{len(documents)} yielded no graph data")
                    
            except Exception as e:
                print(f"  ❌ Chunk {i+1}/{len(documents)} failed graph extraction: {str(e)}")
                # Continue to next chunk
                continue

        if not valid_graph_documents:
            print("⚠ No valid graph data extracted from any chunks.")
            return

        print(f"✓ Extracted {len(valid_graph_documents)} valid graph segments")
        
        try:
            # Store in Neo4j
            self.graph.add_graph_documents(valid_graph_documents)
            print("✓ Graph data successfully stored in Neo4j")
        except Exception as e:
            print(f"❌ Error storing graph documents in Neo4j: {e}")

    
    def query_graph(self, query: str, llm=None) -> str:
        """
        Query the graph using GraphCypherQAChain.
        """
        if not self.graph:
            return ""
            
        try:
            from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
            
            # Use provided LLM or create a default one
            if llm is None:
                llm = ChatOllama(
                    model=settings.DEFAULT_LLM_MODEL,
                    temperature=0,
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                )

            chain = GraphCypherQAChain.from_llm(
                llm=llm, 
                graph=self.graph, 
                verbose=True,
                allow_dangerous_requests=True # Required for Neo4j
            )
            
            result = chain.invoke({"query": query})
            return result.get("result", "")
            
        except Exception as e:
            print(f"❌ Graph Query Error: {e}")
            return ""

    def refresh_schema(self):
        """Refresh graph schema."""
        if self.graph:
            self.graph.refresh_schema()

