# Local RAG Chatbot with GraphRAG & Decoupled Architecture

A production-ready, fully local Retrieval-Augmented Generation (RAG) system featuring a decoupled microservices architecture, visual knowledge graph, and event-driven communication.

## 🚀 Features

- **100% Local Privacy**: Runs entirely on your machine using [LM Studio](https://lmstudio.ai/). No API keys required.
- **Microservices Architecture**:
  - **Frontend**: Streamlit-based interactive UI.
  - **Backend**: FastAPI for robust API endpoints.
  - **Worker**: Celery (with Redis) for asynchronous document processing.
- **GraphRAG**: Visual knowledge graph construction and visualization using Neo4j and Streamlit-Agraph.
- **Event-Driven**: Internal Event Bus for state management and decoupled component communication.
- **Hybrid Support**: Switch between Vector Search and Graph Exploration.
- **Observability**: Built-in support for multiple embedding models and hot-swappable LLM providers.

## 🏗 Architecture

The system consists of the following containerized services:

1.  **rag-chatbot** (Frontend): Streamlit UI for uploading documents, chatting, and viewing the knowledge graph.
2.  **backend** (API): FastAPI service handling file uploads, retrieval logic, and task orchestration.
3.  **celery_worker** (Processing): Background worker for parsing PDF/TXT/MD files and generating embeddings.
4.  **redis** (Message Broker & Cache): Handles Celery task queues and caches vector stores.
5.  **neo4j** (Graph Database): Stores extracted entities and relationships for knowledge graph visualization.

## 🛠 Prerequisites

- **Docker Desktop** (must be running)
- **LM Studio** (installed on your host machine)
- **Python 3.10+** (for local development)
- **Make** (optional, for easy commands)

## ⚡ Quick Start (Docker)

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd poc-rag-chatbot-wiki
    ```

2.  **Configure Environment**
    ```bash
    cp .env.example .env
    # The default settings allow connecting to host-based LM Studio.
    ```

3.  **Start LLM Server**
    - **LM Studio**: Start the Local Inference Server (default port 1234).

4.  **Deploy Application**
    ```bash
    make deploy
    # Or: ./scripts/deploy.sh
    ```

5.  **Access the App**
    Open your browser to: http://localhost:8501

## 🖥 Local Development

For developing individual components without Docker containers:

1.  **Install Dependencies**
    ```bash
    make install
    # or
    pip install -r requirements.txt
    ```

2.  **Start Infrastructure (Redis & Neo4j)**
    You still need Redis and Neo4j running. You can use Docker for just these:
    ```bash
    docker run -d -p 6379:6379 redis:7-alpine
    docker run -d -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password -e NEO4J_PLUGINS='["apoc"]' neo4j:5.11
    ```

3.  **Start Components** (in separate terminals)

    **Terminal 1: Backend**
    ```bash
    uvicorn src.backend.main:app --reload --port 8000
    ```

    **Terminal 2: Worker**
    ```bash
    celery -A src.backend.celery_config worker --loglevel=info
    ```

    **Terminal 3: Frontend**
    ```bash
    streamlit run app.py
    ```

## ⚙️ Configuration

Key settings in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|

| `LLM_BASE_URL` | URL for LM Studio | `http://host.docker.internal:1234/v1` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` (Docker) |
| `NEO4J_URI` | Neo4j Bolt URI | `bolt://neo4j:7687` |
| `ENABLE_GRAPHRAG`| Enable Graph features | `True` |

## 📁 Project Structure

```
poc-rag-chatbot-wiki/
├── src/
│   ├── backend/              # FastAPI Application & Celery Tasks
│   └── chatbot/              # Core Logic (RAG, Events, Graph)
├── config/                   # Configuration settings
├── data/                     # Data storage (gitignored)
├── scripts/                  # Helper scripts (build, deploy, clean)
├── app.py                    # Streamlit Frontend
├── docker-compose.yml        # Docker Services Definition
├── Makefile                  # Setup & Run Commands
└── requirements.txt          # Python Dependencies
```

## 🔧 Troubleshooting

**"Connection refused" to LM Studio?**
Ensure your local LLM server is allowing external connections or listen on `0.0.0.0`. For Docker on Mac/Windows, `host.docker.internal` is used to reach the host.

**Graph not showing?**
Ensure Neo4j is running and the `ENABLE_GRAPHRAG` setting is True. You may need to "Start Over" and re-process the document to populate the graph.

**Worker not picking up tasks?**
Check Redis connectivity and ensure the `celery_worker` container is healthy.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
