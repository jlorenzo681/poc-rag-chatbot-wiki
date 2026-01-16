
import os
from langchain_ollama import ChatOllama
import requests

base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
model = "llama3.2:1b"

print(f"Testing connectivity to {base_url}...")
try:
    resp = requests.get(f"{base_url}/api/tags")
    print(f"Connectivity check: {resp.status_code}")
    print(f"Models: {resp.json()}")
except Exception as e:
    print(f"Connectivity failed: {e}")

print(f"\nTesting ChatOllama with base_url='{base_url}', model='{model}'...")
try:
    llm = ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0
    )
    result = llm.invoke("Hello, are you there?")
    print(f"Result: {result.content}")
except Exception as e:
    print(f"ChatOllama failed: {e}")
