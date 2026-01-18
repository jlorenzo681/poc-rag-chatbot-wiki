
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot.core.simple_graph_transformer import SimpleGraphTransformer
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from langchain_core.runnables import RunnableLambda

def mock_invoke(input):
    # Simulate LLM returning a list for 'head' which causes unhashable type error
    json_content = """
    {
        "relationships": [
            {
                "head": ["Entity1", "Entity1_Alias"],
                "head_type": "Person",
                "relation": "KNOWS",
                "tail": "Entity2",
                "tail_type": "Person"
            }
        ]
    }
    """
    return AIMessage(content=json_content)

def test_reproduction():
    # Wrap function in RunnableLambda to satisfy type checks for | operator
    mock_llm = RunnableLambda(mock_invoke)
    transformer = SimpleGraphTransformer(mock_llm)
    
    doc = Document(page_content="Test content")
    
    print("Attempting to convert document with malformed LLM output...")
    try:
        results = transformer.convert_to_graph_documents([doc])
        if results is None or len(results) == 0:
            print("✓ SUCCESS: handled gracefully (returned None/empty)")
        else:
            print("✓ SUCCESS: Results returned")
    except Exception as e:
        print(f"❌ CAUGHT EXPECTED ERROR: {e}")

if __name__ == "__main__":
    test_reproduction()
