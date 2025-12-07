
import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.vector_store import initialize_vector_store
from src.rag_pipeline import create_rag_chain, query_rag

# Load environment variables
from pathlib import Path
env_path = Path(__file__).parent / ".env"
print(f"DEBUG: Looking for .env at {env_path}, exists={env_path.exists()}")
load_dotenv(dotenv_path=env_path)

def verify_rag():
    api_key = os.getenv("OPENAI_API_KEY")
    print(f"DEBUG: OPENAI_API_KEY set? {bool(api_key)}")
    if api_key:
        print(f"DEBUG: Key starts with {api_key[:8]}...")
    else:
        print("DEBUG: OPENAI_API_KEY is missing!")

    print("🚀 Starting RAG Verification...")

    # 1. Force Reprocess to use new PDFPlumberLoader
    print("\n📦 Rebuilding Vector Database (this might take a minute)...")
    try:
        vector_store = initialize_vector_store(force_reprocess=True)
    except Exception as e:
        print(f"❌ Failed to initialize vector store: {e}")
        return

    # 2. Create Chain
    chain = create_rag_chain(vector_store)

    # 3. Test Questions
    questions = [
        "what's the shear force for HBS diameter 6mm?",
        "How many screws in HBS8200 box?",
        "What are these minimum distances for HBS 10mm?"
    ]

    print("\n🔍 Running Test Queries...")
    
    print("\n🧐 Debugging Retrieval for specific keywords...")
    
    keywords = ["HBS8200"]
    for keyword in keywords:
        print(f"\n--- Searching for '{keyword}' ---")
        docs = vector_store.similarity_search(keyword, k=10)
        found = False
        for doc in docs:
            if keyword.lower() in doc.page_content.lower():
                print(f"✅ Found in Source: {doc.metadata.get('source')} Page: {doc.metadata.get('page')}")
                print(f"Preview: {doc.page_content[:150]}...")
                found = True
        if not found:
            print(f"❌ '{keyword}' NOT found in top 5 results")

    # Skip questions for now to focus on retrieval debug
    

    for q in questions:
        print(f"\n❓ Question: {q}")
        try:
            result = query_rag(chain, q)
            print(f"✅ Answer:\n{result['answer']}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Error answering question: {e}")

if __name__ == "__main__":
    verify_rag()
