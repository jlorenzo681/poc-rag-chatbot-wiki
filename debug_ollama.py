from langchain_ollama import OllamaEmbeddings
import os
import sys

print(f"Testing OllamaEmbeddings initialization...")
try:
    # Try 1: Standard base_url
    print("Attempt 1: base_url='http://ollama:11434'")
    emb = OllamaEmbeddings(model="llama3", base_url="http://ollama:11434")
    print("✓ Success")
except Exception as e:
    print(f"✗ Failed: {e}")

try:
    # Try 2: Without base_url
    print("\nAttempt 2: No base_url")
    emb = OllamaEmbeddings(model="llama3")
    print("✓ Success")
except Exception as e:
    print(f"✗ Failed: {e}")

try:
    # Try 3: Check introspection
    import pydantic
    print(f"\nPydantic version: {pydantic.VERSION}")
    print(f"OllamaEmbeddings fields: {OllamaEmbeddings.__fields__.keys()}")
except Exception as e:
    print(f"✗ Failed to inspect fields: {e}")
