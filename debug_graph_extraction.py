import os
import sys
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

# Setup
print("Initializing LLM and Transformer...")
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    base_url="http://ollama:11434"
)

transformer = LLMGraphTransformer(llm=llm)

# Test Document
text = "Alice works at Google. She lives in New York."
doc = Document(page_content=text)

print(f"\nProcessing text: '{text}'")

try:
    # Try to convert
    graph_docs = transformer.convert_to_graph_documents([doc])
    print("\n✓ Extraction Successful!")
    for g in graph_docs:
        print(g)
except Exception as e:
    print(f"\n❌ Extraction Failed: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Raw LLM Output Debugging ---")
# To see raw output, we might need to invoke the chain directly if transformer hides it.
# LLMGraphTransformer uses a chain internally.
try:
    chain = transformer.chain
    print("Invoking internal chain to see raw output...")
    raw_response = chain.invoke({"input": text})
    print(f"\nRaw Response Type: {type(raw_response)}")
    print(f"Raw Response Content:\n{raw_response}")
except Exception as e:
    print(f"Could not invoke chain directly: {e}")
