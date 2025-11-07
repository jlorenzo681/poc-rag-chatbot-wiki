"""
RAG Chain Module
Implements the retrieval-augmented generation chain with conversation memory.
"""

from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


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
        openai_api_key: str,
        model_name: str = "gpt-4-turbo",
        temperature: float = 0.3,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the RAG chain.

        Args:
            retriever: Vector store retriever
            openai_api_key: OpenAI API key
            model_name: Name of the LLM model
            temperature: Temperature for response generation (0-1)
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt
        """
        self.retriever = retriever
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )

        print(f"🤖 Initialized LLM: {model_name}")
        print(f"   Temperature: {temperature}")
        print(f"   Max tokens: {max_tokens}")

    def create_basic_chain(self):
        """
        Create a basic RAG chain without conversation memory.

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
    ):
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
        return_sources: bool = True
    ):
        """
        Initialize the RAG chatbot.

        Args:
            chain: RAG chain instance
            return_sources: Whether to return source documents
        """
        self.chain = chain
        self.return_sources = return_sources

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question to the chatbot.

        Args:
            question: Question to ask

        Returns:
            Dictionary containing answer and optional source documents
        """
        try:
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

            return result

        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "error": True
            }

    def _format_sources(self, documents) -> list:
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
