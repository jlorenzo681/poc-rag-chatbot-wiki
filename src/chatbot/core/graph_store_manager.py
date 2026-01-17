
import os
from typing import List, Optional, Dict, Any
from langchain_community.graphs import Neo4jGraph
from .simple_graph_transformer import SimpleGraphTransformer
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
import config.settings as settings

class GraphStoreManager:
    """
    Manages Neo4j graph database interactions and document transformation.
    """
    def __init__(self, model_name: Optional[str] = None):
        self.graph = None
        self.llm_transformer = None
        self.model_name = model_name
        
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
            llm = None
            if settings.DEFAULT_LLM_PROVIDER in ["lmstudio", "mlx"]:
                print(f"🔧 Using {settings.DEFAULT_LLM_PROVIDER.upper()} for Graph Extraction: {settings.LLM_BASE_URL}")
                
                # Auto-detect model if "local-model" is set (default)
                # Prioritize instance override model_name
                model_name = self.model_name or settings.DEFAULT_LLM_MODEL
                
                if model_name == "local-model":
                    try:
                        import requests
                        models_url = f"{settings.LLM_BASE_URL.rstrip('/v1')}/v1/models"
                        response = requests.get(models_url, timeout=2)
                        if response.status_code == 200:
                            data = response.json()
                            if data.get("data"):
                                model_name = data["data"][0]["id"]
                                print(f"🔍 Detected loaded model in LM Studio: {model_name}")
                    except Exception as e:
                        print(f"⚠ Failed to auto-detect model from LM Studio: {e}")
                        
                llm = ChatOpenAI(
                    base_url=settings.LLM_BASE_URL,
                    api_key="lm-studio",
                    model=model_name,
                    temperature=0
                )
            else:
                # Use Ollama
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                print(f"🔧 Using Ollama for Graph Extraction: {base_url}")
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
            # Use provided LLM or create a default one
            if llm is None:
                if settings.DEFAULT_LLM_PROVIDER in ["lmstudio", "mlx"]:
                    # Auto-detect model if "local-model" is set
                    model_name = self.model_name or settings.DEFAULT_LLM_MODEL
                    if model_name == "local-model":
                        try:
                            import requests
                            # Note: In query_graph we are likely inside a request, so prints might not show in main stdout easily, but useful for debug
                            models_url = f"{settings.LLM_BASE_URL.rstrip('/v1')}/v1/models"
                            response = requests.get(models_url, timeout=2)
                            if response.status_code == 200:
                                data = response.json()
                                if data.get("data"):
                                    # Filter out embedding models to avoid selecting 'text-embedding-bge-m3' as LLM
                                    for model_info in data["data"]:
                                        model_id = model_info["id"]
                                        if "embedding" not in model_id.lower():
                                            model_name = model_id
                                            print(f"🔍 Detected LLM model in LM Studio: {model_name}")
                                            break
                                    else:
                                        # If all look like embeddings, just take the first one as fallback
                                        model_name = data["data"][0]["id"]
                        except Exception:
                            pass # Fallback to default if detection fails

                    llm = ChatOpenAI(
                        base_url=settings.LLM_BASE_URL,
                        api_key="lm-studio",
                        model=model_name,
                        temperature=0
                    )
                else:
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

    def get_visual_graph(self, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve nodes and relationships for visualization.
        Returns raw data to be formatted by the frontend.
        """
        if not self.graph:
            return {"nodes": [], "edges": []}

        try:
            # Query to get a subgraph
            query = f"""
            MATCH (n)-[r]->(m)
            RETURN n, r, m
            LIMIT {limit}
            """
            
            # Use the graph's query method directly
            data = self.graph.query(query)
            
            nodes = {}
            edges = []
            
            for record in data:
                # Process source node
                source = record['n']
                source_id = source.get('id') or source.get('name') or str(source.id)
                nodes[source_id] = {
                    "id": source_id,
                    "label": source.get('name', source_id),
                    "type": list(source.labels)[0] if source.labels else "Node"
                }
                
                # Process target node
                target = record['m']
                target_id = target.get('id') or target.get('name') or str(target.id)
                nodes[target_id] = {
                    "id": target_id,
                    "label": target.get('name', target_id),
                    "type": list(target.labels)[0] if target.labels else "Node"
                }
                
                # Process relationship
                rel = record['r']
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "label": type(rel).__name__
                })
                
            return {
                "nodes": list(nodes.values()),
                "edges": edges
            }
            
        except Exception as e:
            print(f"❌ Error fetching visual graph: {e}")
            return {"nodes": [], "edges": []}


    def refresh_schema(self):
        """Refresh graph schema."""
        if self.graph:
            self.graph.refresh_schema()

    def _get_marker_path(self, file_hash: str) -> str:
        """Generate the marker file path safely."""
        # Make marker model-aware to handle switching models
        # Prioritize instance model name, then default from settings
        model_name = self.model_name or settings.DEFAULT_LLM_MODEL
        safe_llm_model = model_name.replace("/", "_").replace(":", "_")
        return f"data/vector_stores/{file_hash}_{safe_llm_model}_graph.done"

    def check_cache(self, file_hash: str) -> bool:
        """
        Check if graph data has already been extracted for this file and model.
        Returns True if cached.
        """
        if not getattr(settings, "ENABLE_GRAPHRAG", False):
            return False
            
        marker_path = self._get_marker_path(file_hash)
        if os.path.exists(marker_path):
            print(f"✓ Graph marker found at {marker_path}")
            return True
        return False

    def mark_as_completed(self, file_hash: str):
        """
        Create a marker file indicating graph extraction accomplishment.
        """
        if not getattr(settings, "ENABLE_GRAPHRAG", False):
            return

        marker_path = self._get_marker_path(file_hash)
        try:
            with open(marker_path, "w") as f:
                f.write("done")
            print(f"✓ Created graph marker at {marker_path}")
        except Exception as e:
            print(f"❌ Failed to create graph marker: {e}")

