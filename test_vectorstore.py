# Test Vector Store Initialization
# This script tests the vector store setup and creates the FAISS index

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from services.vectorstore import (_build_index, get_vectorstore,
                                  search_with_fallback)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_faiss_index():
    """Test FAISS index creation and search"""
    print("\n" + "=" * 80)
    print("Testing FAISS Vector Store")
    print("=" * 80)

    try:
        # Check if index exists
        index_path = Path("faiss_index")
        if index_path.exists():
            print(f"✓ FAISS index directory exists at: {index_path}")
        else:
            print(f"✗ FAISS index directory not found. Will create new index...")

        # Get or create vector store
        print("\nInitializing FAISS vector store...")
        vectorstore = get_vectorstore()

        if vectorstore:
            print("✓ FAISS vector store initialized successfully!")

            # Test search
            print("\nTesting vector search with query: 'delayed containers'")
            results = vectorstore.similarity_search("delayed containers", k=3)

            if results:
                print(f"✓ Found {len(results)} results from FAISS")
                print("\nSample result (first 200 chars):")
                print("-" * 80)
                print(results[0].page_content[:200] + "...")
                print("-" * 80)
            else:
                print("✗ No results found. Index might be empty.")

        else:
            print("✗ Failed to initialize FAISS vector store")
            return False

    except Exception as e:
        print(f"✗ Error testing FAISS: {e}")
        logger.error(f"FAISS test failed", exc_info=True)
        return False

    return True


def test_search_fallback():
    """Test the search with fallback mechanism"""
    print("\n" + "=" * 80)
    print("Testing Search with Fallback")
    print("=" * 80)

    try:
        test_queries = [
            "delayed containers",
            "containers arriving at Los Angeles",
            "shipments from Shanghai",
        ]

        for query in test_queries:
            print(f"\nSearching for: '{query}'")
            results = search_with_fallback(query, k=2)

            if results:
                print(f"  ✓ Found {len(results)} results")
            else:
                print(f"  ✗ No results found")

    except Exception as e:
        print(f"✗ Error in search fallback: {e}")
        logger.error(f"Search fallback test failed", exc_info=True)
        return False

    return True


def check_configuration():
    """Check configuration settings"""
    print("\n" + "=" * 80)
    print("Checking Configuration")
    print("=" * 80)

    print(f"\nVector Store Type: {getattr(settings, 'VECTOR_STORE_TYPE', 'faiss')}")
    print(f"Azure OpenAI Endpoint: {settings.AZURE_OPENAI_ENDPOINT[:50]}...")
    print(f"Azure OpenAI Deployment: {settings.AZURE_OPENAI_DEPLOYMENT}")
    print(f"Embedding Model: {settings.AZURE_OPENAI_EMBEDDING_MODEL}")

    # Check Pinecone config
    has_pinecone = hasattr(settings, "PINECONE_API_KEY") and settings.PINECONE_API_KEY
    print(f"Pinecone Configured: {'Yes' if has_pinecone else 'No'}")

    if has_pinecone:
        print(f"Pinecone Environment: {settings.PINECONE_ENVIRONMENT}")
        print(f"Pinecone Index Name: {settings.PINECONE_INDEX_NAME}")


def rebuild_index():
    """Force rebuild the FAISS index"""
    print("\n" + "=" * 80)
    print("Rebuilding FAISS Index")
    print("=" * 80)

    try:
        import shutil

        index_path = Path("faiss_index")

        if index_path.exists():
            print(f"Removing existing index at: {index_path}")
            shutil.rmtree(index_path)

        print("\nBuilding new FAISS index (this may take a few minutes)...")
        vectorstore = _build_index()

        if vectorstore:
            print("✓ FAISS index rebuilt successfully!")
            return True
        else:
            print("✗ Failed to rebuild FAISS index")
            return False

    except Exception as e:
        print(f"✗ Error rebuilding index: {e}")
        logger.error(f"Index rebuild failed", exc_info=True)
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("VECTOR STORE DIAGNOSTICS")
    print("=" * 80)

    # Check configuration
    check_configuration()

    # Ask user if they want to rebuild
    index_exists = Path("faiss_index").exists()
    if index_exists:
        print("\n" + "=" * 80)
        response = (
            input("\nFAISS index already exists. Rebuild it? (y/n): ").strip().lower()
        )
        if response == "y":
            if not rebuild_index():
                print("\n✗ Rebuild failed. Exiting.")
                sys.exit(1)
    else:
        print("\nNo FAISS index found. Building new index...")
        if not rebuild_index():
            print("\n✗ Index creation failed. Exiting.")
            sys.exit(1)

    # Test FAISS
    if not test_faiss_index():
        print("\n✗ FAISS test failed")
        sys.exit(1)

    # Test search fallback
    if not test_search_fallback():
        print("\n✗ Search fallback test failed")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nVector store is working correctly.")
    print("You can now use the vector search functionality in your chatbot.")
    print("\nTo test in your chatbot, try queries like:")
    print("  - 'Search for delayed containers'")
    print("  - 'Find information about container ABCD1234567'")
    print("  - 'What shipments are going to Los Angeles?'")
    print("=" * 80 + "\n")
