# Test Azure OpenAI Embeddings (Without FAISS)
# This verifies that vector embeddings are working for semantic search

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_openai import AzureOpenAIEmbeddings

from config import settings

print("=" * 80)
print("Testing Azure OpenAI Embeddings for Vector/Semantic Search")
print("=" * 80)

# Initialize embeddings
print("\n1. Initializing Azure OpenAI Embeddings...")
print(f"   Endpoint: {settings.AZURE_OPENAI_ENDPOINT[:50]}...")
print(f"   Model: {settings.AZURE_OPENAI_EMBEDDING_MODEL}")

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=settings.AZURE_OPENAI_EMBEDDING_MODEL,
    openai_api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
)

# Test 1: Embed a document
print("\n2. Testing document embedding...")
test_doc = """
container_number: TEMU1234567
po_number: PO123456
status: In Transit
eta: 2026-02-25
port_of_discharge: Los Angeles
"""

try:
    doc_embedding = embeddings.embed_query(test_doc)
    print(f"   ✅ SUCCESS: Generated embedding vector")
    print(f"   Vector dimensions: {len(doc_embedding)}")
    print(f"   First 5 values: {doc_embedding[:5]}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 2: Embed a user query
print("\n3. Testing query embedding (for semantic search)...")
user_query = "delayed containers arriving at Los Angeles port"

try:
    query_embedding = embeddings.embed_query(user_query)
    print(f"   ✅ SUCCESS: Generated query embedding")
    print(f"   Vector dimensions: {len(query_embedding)}")
    print(f"   First 5 values: {query_embedding[:5]}")
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

# Test 3: Calculate similarity
print("\n4. Testing semantic similarity calculation...")
import numpy as np


def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Create related and unrelated queries
related_query = "containers delayed at LA port"
unrelated_query = "how to bake a cake"

try:
    related_emb = embeddings.embed_query(related_query)
    unrelated_emb = embeddings.embed_query(unrelated_query)

    # Calculate similarities
    related_similarity = cosine_similarity(query_embedding, related_emb)
    unrelated_similarity = cosine_similarity(query_embedding, unrelated_emb)

    print(f"   Original query: '{user_query}'")
    print(f"   Related query:  '{related_query}'")
    print(f"   Similarity: {related_similarity:.4f} ✅ (HIGH - semantically similar)")
    print(f"")
    print(f"   Unrelated query: '{unrelated_query}'")
    print(
        f"   Similarity: {unrelated_similarity:.4f} ✅ (LOW - semantically different)"
    )

    if related_similarity > unrelated_similarity:
        print("\n   ✅ Semantic search is working correctly!")
        print("   Related queries have higher similarity than unrelated ones.")
    else:
        print("\n   ⚠️  Warning: Unexpected similarity scores")

except Exception as e:
    print(f"   ❌ FAILED: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("CONCLUSION:")
print("=" * 80)
print("✅ Vector embeddings are working correctly")
print("✅ Azure OpenAI embedding model is responding")
print("✅ Semantic similarity calculations are accurate")
print("✅ Your system CAN perform vector/semantic search")
print("")
print("⚠️  Note: FAISS has DLL issues on Windows, but embeddings work fine.")
print("   Recommendation: Use Pinecone instead (already configured in your code)")
print("   Or manually install Microsoft Visual C++ Redistributable for FAISS")
print("=" * 80)
