"""
RAG Chain Module
Implements the retrieval-augmented generation chain with conversation memory.
"""

from typing import Optional, Dict, Any, Literal, List
from langchain.schema import Document
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
from .event_bus import EventBus, ChatQueryEvent, ChatResponseEvent, ErrorEvent


class RAGChain:
    """
    Creates and manages RAG (Retrieval-Augmented Generation) chains.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant answering questions based on provided documents.
Use the context provided to answer questions accurately and comprehensively.

Guidelines:
- Base your answers on the provided context
- If the answer is not in the context, say "I don't have information about this in the documents."
- Cite specific details from the context when possible
- Be concise but thorough
- If you're uncertain, acknowledge it

Context: {context}"""

    def __init__(
        self,
        retriever,
        llm_provider: Literal["ollama", "lmstudio"] = "ollama",
        ollama_base_url: str = "http://ollama:11434",
        lmstudio_base_url: str = "http://localhost:1234/v1",
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.3,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the RAG chain.

        Args:
            retriever: Vector store retriever
            llm_provider: LLM provider to use ('ollama', or 'lmstudio')
            ollama_base_url: Ollama server URL (used if llm_provider='ollama')
            lmstudio_base_url: LM Studio server URL (used if llm_provider='lmstudio')
            model_name: Name of the LLM model
            temperature: Temperature for response generation (0-1)
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt
        """
        self.retriever = retriever
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.llm_provider = llm_provider

        # Initialize LLM based on provider
        if llm_provider == "ollama":
            self.llm = ChatOllama(
                base_url=ollama_base_url,
                model=model_name,
                temperature=temperature,
                num_predict=max_tokens,
                timeout=300.0,  # 5 minutes for CPU-only inference
            )
            print(f"🤖 Initialized Ollama LLM: {model_name}")
            print(f"   Base URL: {ollama_base_url}")
            print("   ⚠ Using CPU inference - responses may be slow")

        elif llm_provider == "lmstudio":
            self.llm = ChatOpenAI(
                base_url=lmstudio_base_url,
                api_key="lm-studio",  # LM Studio doesn't require a real API key
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=300.0,
            )
            print(f"🤖 Initialized LM Studio LLM: {model_name}")
            print(f"   Base URL: {lmstudio_base_url}")
            print("   Using Metal GPU acceleration")

        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        print(f"   Temperature: {temperature}")
        print(f"   Max tokens: {max_tokens}")

    def create_basic_chain(self) -> Any:
        """
        Create a basic RAG chain without conversation memory. Example purpose.

        Returns:
            Retrieval chain
        """
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{input}")
        ])

        # Create document combination chain
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)

        # Create full retrieval chain
        rag_chain = create_retrieval_chain(
            self.retriever,
            question_answer_chain
        )

        print("✓ Basic RAG chain created")
        return rag_chain

    def create_conversational_chain(
        self,
        memory_type: str = "buffer",
        window_size: int = 5
    ) -> Any:
        """
        Create a conversational RAG chain with memory.

        Args:
            memory_type: Type of memory ('buffer' or 'window')
            window_size: Number of conversation turns to remember (for window memory)

        Returns:
            Conversational retrieval chain
        """
        # Initialize memory
        if memory_type == "buffer":
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            print("💭 Using ConversationBufferMemory (full history)")
        elif memory_type == "window":
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer",
                k=window_size
            )
            print(f"💭 Using ConversationBufferWindowMemory (last {window_size} turns)")
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")

        # Create conversational chain
        conversational_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )

        print("✓ Conversational RAG chain created")
        return conversational_chain


class RAGChatbot:
    """
    High-level interface for the RAG chatbot.
    """

    def __init__(
        self,
        chain,
        return_sources: bool = True,
        event_bus: Optional[EventBus] = None,
        event_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the RAG chatbot.

        Args:
            chain: RAG chain instance
            return_sources: Whether to return source documents
            event_bus: Event bus for publishing events
            event_metadata: Metadata for events (llm_provider, model_name, etc.)
        """
        self.chain = chain
        self.return_sources = return_sources
        self.event_bus = event_bus
        self.event_metadata = event_metadata or {}

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the chatbot.

        Args:
            question: Question to ask

        Returns:
            Dictionary containing answer and optional source documents
        """
        try:
            start_time = time.time()
            
            # Emit query event
            if self.event_bus:
                self.event_bus.publish(ChatQueryEvent(
                    question=question,
                    llm_provider=self.event_metadata.get("llm_provider", "unknown"),
                    model_name=self.event_metadata.get("model_name", "unknown")
                ))

            # Check if it's a conversational chain
            if hasattr(self.chain, 'memory'):
                response = self.chain({"question": question})
            else:
                response = self.chain.invoke({"input": question})

            # Format response
            result = {
                "question": question,
                "answer": response.get("answer", ""),
            }

            # Add source documents if available and requested
            if self.return_sources:
                # Handle different response formats
                sources = response.get("source_documents") or response.get("context", [])
                if sources:
                    result["sources"] = self._format_sources(sources)

            # Emit response event
            if self.event_bus:
                self.event_bus.publish(ChatResponseEvent(
                    question=question,
                    answer=result["answer"],
                    source_count=len(result.get("sources", [])),
                    duration_seconds=time.time() - start_time
                ))

            return result

        except Exception as e:
            if self.event_bus:
                self.event_bus.publish(ErrorEvent(
                    error_type=type(e).__name__,
                    message=str(e),
                    context={"question": question}
                ))
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in RAG chain: {error_details}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}\n\nDetails: {type(e).__name__}",
                "error": True
            }

    def _format_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Format source documents for display.

        Args:
            documents: List of source documents

        Returns:
            List of formatted source information
        """
        sources = []
        for i, doc in enumerate(documents, 1):
            source_info = {
                "index": i,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        return sources

    def reset_conversation(self):
        """
        Reset conversation history (for conversational chains).
        """
        if hasattr(self.chain, 'memory'):
            self.chain.memory.clear()
            print("✓ Conversation history cleared")
        else:
            print("⚠ This chain doesn't have conversation memory")


if __name__ == "__main__":
    print("RAG Chain module initialized successfully!")
    print("\nSupported features:")
    print("  - Basic RAG chain")
    print("  - Conversational chain with memory")
    print("  - Source document tracking")
    print("  - Customizable system prompts")
