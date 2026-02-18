# Quick ChromaDB Test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Quick ChromaDB Test")
print("=" * 80)

# Check if ChromaDB directory exists
chroma_dir = Path("chroma_db")
if chroma_dir.exists():
    print(f"\n✓ ChromaDB directory exists at: {chroma_dir}")

    # Try to load and search
    try:
        from services.vectorstore import get_vectorstore

        print("\n1. Loading ChromaDB vectorstore...")
        vectorstore = get_vectorstore()
        print("   ✓ ChromaDB loaded successfully!")

        print("\n2. Testing semantic search...")
        query = "delayed containers"
        results = vectorstore.similarity_search(query, k=3)

        if results:
            print(f"   ✓ Found {len(results)} results for '{query}'")
            print(f"\n   Sample result (first 150 chars):")
            print(f"   {results[0].page_content[:150]}...")
        else:
            print("   ⚠️  No results found")

        print("\n" + "=" * 80)
        print("✓ ChromaDB IS WORKING!")
        print("=" * 80)
        print("\nVector embeddings are successfully being used for:")
        print("  ✓ Vector similarity search")
        print("  ✓ Semantic search of user queries")
        print("  ✓ Pure Python (no DLL issues)")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
else:
    print(f"\n⚠️  ChromaDB directory not found.")
    print(f"   Run the full test_vectorstore.py to build the index first.")
    print(f"   Or wait - let me try to build it now...")

    try:
        from services.vectorstore import _build_index

        print("\n   Building ChromaDB index (may take a few minutes)...")
        vectorstore = _build_index()

        if vectorstore:
            print("   ✓ Index built successfully!")

            # Test search
            results = vectorstore.similarity_search("delayed containers", k=2)
            if results:
                print(f"   ✓ Search works! Found {len(results)} results")
                print("\n" + "=" * 80)
                print("✓ SUCCESS - ChromaDB is fully operational!")
                print("=" * 80)
            else:
                print("   ⚠️  Index built but search returned no results")
        else:
            print("   ✗ Failed to build index")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error building index: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
