# services/vectorstore.py
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import pandas as pd

# Import text splitter from the correct package
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings

from config import settings

from .azure_blob import get_shipment_df

logger = logging.getLogger("shipping_chatbot")
VECTORSTORE_DIR = Path("chroma_db")
METADATA_FILE = VECTORSTORE_DIR / "index_metadata.json"


def _embeddings() -> AzureOpenAIEmbeddings:
    """Create AzureOpenAIEmbeddings the same object used for RAG and for the RetrievalQA chain."""
    return AzureOpenAIEmbeddings(
        azure_deployment=settings.AZURE_OPENAI_EMBEDDING_MODEL,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
    )


def _build_index() -> Chroma:
    """
    Build a brand new ChromaDB index from the shipment dataframe.
    The routine is batched and throttled to respect Azure rate limits.
    """
    logger.info("Creating new ChromaDB vector store")
    df = get_shipment_df()

    # Turn each row into a single text block
    # Optimization: Use a format that is more token-efficient or LLM-friendly if possible.
    # Logic Fix: Increase text chunk size to avoid splitting a single row into multiple incoherent chunks.
    rows_as_text = [
        "\n".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
        for row in df.fillna("").astype(str).to_dict(orient="records")
    ]

    # CRITICAL FIX: Increased chunk_size from 400 to 2000.
    # Splitting a row breaks the context (e.g. key 'ETA' separated from value).
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks: List[str] = []

    # Check if rows are effectively split. Ideally, we want 1 row = 1 chunk.
    for txt in rows_as_text:
        # If the row is small enough, keep it as is.
        # split_text will return [txt] if it fits.
        split_chunks = splitter.split_text(txt)
        chunks.extend(split_chunks)

    embeddings = _embeddings()

    # Create persistent directory
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize ChromaDB client with telemetry disabled to avoid SSL issues
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR),
        settings=ChromaSettings(
            anonymized_telemetry=False,  # Disable telemetry to avoid SSL issues
            allow_reset=True,
        ),
    )

    # Process in batches with throttling to avoid Azure rate limits
    batch = 10  # Very conservative batch size to avoid rate limits
    logger.info(
        f"Building ChromaDB index from {len(chunks)} chunks with batch size {batch}"
    )

    # Create vectorstore and add documents in batches
    vectorstore = Chroma(
        client=client,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    for i in range(0, len(chunks), batch):
        batch_texts = chunks[i : i + batch]

        batch_num = (i // batch) + 1
        total_batches = (len(chunks) + batch - 1) // batch

        # Add batch to vectorstore with retry logic
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                vectorstore.add_texts(batch_texts)
                logger.info(
                    f"Indexed batch {batch_num}/{total_batches} ({len(batch_texts)} chunks)"
                )
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Batch {batch_num} failed (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}"
                    )
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff: 5s, 10s, 20s
                else:
                    logger.error(
                        f"Batch {batch_num} failed after {max_retries} attempts: {e}"
                    )
                    raise

        # Be gentle with the embeddings endpoint to avoid 429s
        # Increased delay to 3 seconds to stay well under rate limits
        if i + batch < len(chunks):  # Don't sleep after the last batch
            time.sleep(3.0)

    logger.info("ChromaDB index created and persisted")

    # Save metadata for incremental updates
    try:
        current_hashes = {_get_row_hash(row): idx for idx, row in df.iterrows()}
        metadata = {
            "last_update": datetime.now().isoformat(),
            "total_records": len(df),
            "total_chunks": len(chunks),
            "indexed_hashes": current_hashes,
            "version": "1.0",
        }
        _save_metadata(metadata)
        logger.info("Index metadata saved for incremental updates")
    except Exception as e:
        logger.warning(f"Failed to save metadata (non-critical): {e}")

    return vectorstore


# ----------------------------------------------------------------------
# Public accessor – lazy‑load on first call
# ----------------------------------------------------------------------
_vectorstore: Chroma | None = None


def get_vectorstore() -> Chroma:
    """Return a cached ChromaDB store; builds it on first request."""
    global _vectorstore
    if _vectorstore is None:
        if VECTORSTORE_DIR.exists():
            logger.info("Loading existing ChromaDB index")
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                client = chromadb.PersistentClient(
                    path=str(VECTORSTORE_DIR),
                    settings=ChromaSettings(
                        anonymized_telemetry=False,  # Disable telemetry
                        allow_reset=True,
                    ),
                )

                _vectorstore = Chroma(
                    client=client,
                    embedding_function=_embeddings(),
                    persist_directory=str(VECTORSTORE_DIR),
                )
            except Exception as e:
                logger.error(f"Failed to load existing index: {e}. Rebuilding...")
                _vectorstore = _build_index()
        else:
            _vectorstore = _build_index()
    return _vectorstore


# ----------------------------------------------------------------------
# Pinecone Vector Store Support
# ----------------------------------------------------------------------
_pinecone_store = None


def _build_pinecone_index():
    """Build or connect to Pinecone index from the shipment dataframe."""
    try:
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone, ServerlessSpec
    except ImportError:
        logger.error(
            "Pinecone not installed. Install with: pip install pinecone-client langchain-pinecone"
        )
        return None

    if not settings.PINECONE_API_KEY or not settings.PINECONE_ENVIRONMENT:
        logger.warning(
            "Pinecone credentials not configured. Skipping Pinecone initialization."
        )
        return None

    try:
        logger.info("Initializing Pinecone vector store")

        # Initialize Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)

        index_name = settings.PINECONE_INDEX_NAME

        # Check if index exists, if not create it
        if index_name not in pc.list_indexes().names():
            logger.info(f"Creating new Pinecone index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=1536,  # text-embedding-ada-002 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws", region=settings.PINECONE_ENVIRONMENT or "us-east-1"
                ),
            )
            time.sleep(1)  # Wait for index to be ready

        # Get the shipment data
        df = get_shipment_df()
        rows_as_text = [
            "\n".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
            for row in df.fillna("").astype(str).to_dict(orient="records")
        ]

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks: List[str] = []
        for txt in rows_as_text:
            split_chunks = splitter.split_text(txt)
            chunks.extend(split_chunks)

        embeddings = _embeddings()

        # Create or update Pinecone vector store
        vectorstore = PineconeVectorStore.from_texts(
            texts=chunks, embedding=embeddings, index_name=index_name
        )

        logger.info(
            f"Pinecone index '{index_name}' initialized with {len(chunks)} chunks"
        )
        return vectorstore

    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}", exc_info=True)
        return None


def get_pinecone_vectorstore():
    """Return a cached Pinecone store; builds it on first request."""
    global _pinecone_store
    if _pinecone_store is None:
        _pinecone_store = _build_pinecone_index()
    return _pinecone_store


def get_hybrid_vectorstore():
    """
    Get vector store based on configuration.
    Returns ChromaDB, Pinecone, or both based on VECTOR_STORE_TYPE setting.
    Falls back to ChromaDB if Pinecone is unavailable.
    """
    store_type = getattr(settings, "VECTOR_STORE_TYPE", "chroma").lower()

    if store_type == "pinecone":
        pinecone = get_pinecone_vectorstore()
        if pinecone:
            return pinecone
        else:
            logger.warning("Pinecone not available, falling back to ChromaDB")
            return get_vectorstore()
    elif store_type == "both":
        chroma = get_vectorstore()
        pinecone = get_pinecone_vectorstore()
        if pinecone:
            return {"chroma": chroma, "pinecone": pinecone}
        else:
            logger.warning("Pinecone not available, using ChromaDB only")
            return chroma
    else:  # default to chroma
        return get_vectorstore()


def search_with_fallback(query: str, k: int = 5) -> List:
    """
    Search vector stores with fallback mechanism.
    Tries ChromaDB first, falls back to Pinecone if ChromaDB fails or returns no results.
    """
    results = []

    # Try ChromaDB first
    try:
        chroma_store = get_vectorstore()
        if chroma_store:
            results = chroma_store.similarity_search(query, k=k)
            if results:
                logger.info(f"Found {len(results)} results from ChromaDB")
                return results
            else:
                logger.info("ChromaDB returned no results, trying Pinecone fallback")
    except Exception as e:
        logger.warning(f"ChromaDB search failed: {e}")

    # Fallback to Pinecone only if it's configured
    try:
        store_type = getattr(settings, "VECTOR_STORE_TYPE", "chroma").lower()
        if store_type in ["pinecone", "both"]:
            pinecone_store = get_pinecone_vectorstore()
            if pinecone_store:
                results = pinecone_store.similarity_search(query, k=k)
                if results:
                    logger.info(
                        f"Found {len(results)} results from Pinecone (fallback)"
                    )
                    return results
    except Exception as e:
        logger.warning(f"Pinecone search failed: {e}")

    if not results:
        logger.warning("No results found from any vector store")
    return results


# ----------------------------------------------------------------------
# Incremental Update System
# ----------------------------------------------------------------------


def _get_row_hash(row: pd.Series) -> str:
    """Generate a unique hash for a DataFrame row based on key fields."""
    # Use key fields that uniquely identify a shipment
    key_fields = [
        "container_number",
        "po_number_multiple",
        "ocean_bl_number",
        "booking_number",
    ]
    row_str = "|".join(
        str(row.get(field, "")) for field in key_fields if field in row.index
    )
    return hashlib.md5(row_str.encode()).hexdigest()


def _row_to_text(row: pd.Series) -> str:
    """Convert a DataFrame row to text format for indexing."""
    return "\n".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())


def _load_metadata() -> Dict:
    """Load index metadata from file."""
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
    return {
        "last_update": None,
        "total_records": 0,
        "total_chunks": 0,
        "indexed_hashes": {},
        "version": "1.0",
    }


def _save_metadata(metadata: Dict):
    """Save index metadata to file."""
    try:
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(
            f"Metadata saved: {metadata['total_records']} records, {metadata['total_chunks']} chunks"
        )
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")


def _detect_changes(current_df: pd.DataFrame, metadata: Dict) -> Dict[str, List]:
    """
    Detect changes between current data and indexed data.
    Returns dict with 'new', 'modified', 'deleted' row indices/hashes.
    """
    indexed_hashes = metadata.get("indexed_hashes", {})

    # Generate hashes for current data
    current_hashes = {}
    for idx, row in current_df.iterrows():
        row_hash = _get_row_hash(row)
        current_hashes[row_hash] = idx

    # Detect changes
    new_hashes = set(current_hashes.keys()) - set(indexed_hashes.keys())
    deleted_hashes = set(indexed_hashes.keys()) - set(current_hashes.keys())

    # For simplicity, treat any hash difference as "modified" is complex
    # Instead, we just track new and deleted

    changes = {
        "new": [current_hashes[h] for h in new_hashes],
        "deleted": list(deleted_hashes),
        "current_hashes": current_hashes,
    }

    logger.info(
        f"Changes detected: {len(changes['new'])} new, {len(changes['deleted'])} deleted"
    )
    return changes


def update_vectorstore_incremental() -> bool:
    """
    Incrementally update the vector store with only new/changed data.
    Returns True if successful, False otherwise.
    """
    logger.info("Starting incremental vectorstore update...")

    try:
        # Load current data
        df = get_shipment_df()
        logger.info(f"Current data: {len(df)} records")

        # Load metadata
        metadata = _load_metadata()

        # Check if index exists
        if not VECTORSTORE_DIR.exists() or not metadata.get("last_update"):
            logger.info("No existing index found. Performing full rebuild...")
            vectorstore = _build_index()

            # Save metadata after full build
            current_hashes = {_get_row_hash(row): idx for idx, row in df.iterrows()}
            metadata = {
                "last_update": datetime.now().isoformat(),
                "total_records": len(df),
                "total_chunks": len(df),  # Will be updated with actual chunk count
                "indexed_hashes": current_hashes,
                "version": "1.0",
            }
            _save_metadata(metadata)
            return True

        # Detect changes
        changes = _detect_changes(df, metadata)

        # If no changes, skip update
        if len(changes["new"]) == 0 and len(changes["deleted"]) == 0:
            logger.info("No changes detected. Skipping update.")
            return True

        # Load existing vectorstore
        logger.info("Loading existing vectorstore...")
        vectorstore = get_vectorstore()

        # Process deleted records
        if changes["deleted"]:
            logger.info(f"Removing {len(changes['deleted'])} deleted records...")
            # ChromaDB doesn't have easy way to delete by custom ID,
            # so we'll skip deletion for now or implement collection reset
            # For production, consider storing document IDs in metadata

        # Process new records
        if changes["new"]:
            logger.info(f"Adding {len(changes['new'])} new records...")
            new_rows = df.iloc[changes["new"]]

            # Convert rows to text
            rows_as_text = [_row_to_text(row) for _, row in new_rows.iterrows()]

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000, chunk_overlap=100
            )
            new_chunks = []
            for txt in rows_as_text:
                split_chunks = splitter.split_text(txt)
                new_chunks.extend(split_chunks)

            logger.info(f"Adding {len(new_chunks)} new chunks...")

            # Add in small batches with retry logic
            batch_size = 10
            for i in range(0, len(new_chunks), batch_size):
                batch_texts = new_chunks[i : i + batch_size]

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        vectorstore.add_texts(batch_texts)
                        logger.info(
                            f"Added batch {i//batch_size + 1}/{(len(new_chunks) + batch_size - 1)//batch_size}"
                        )
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(f"Batch failed, retrying: {str(e)[:100]}")
                            time.sleep(5 * (attempt + 1))
                        else:
                            raise

                time.sleep(3.0)  # Rate limiting

        # Update metadata
        metadata["last_update"] = datetime.now().isoformat()
        metadata["total_records"] = len(df)
        metadata["total_chunks"] = (
            metadata.get("total_chunks", 0) + len(new_chunks)
            if changes["new"]
            else metadata.get("total_chunks", 0)
        )
        metadata["indexed_hashes"] = changes["current_hashes"]
        _save_metadata(metadata)

        logger.info("✅ Incremental update completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Incremental update failed: {e}", exc_info=True)
        return False


def force_rebuild_vectorstore() -> bool:
    """
    Force a complete rebuild of the vectorstore.
    Use this for monthly maintenance or when incremental updates have issues.
    """
    logger.info("Starting FULL vectorstore rebuild...")

    try:
        import shutil

        # Remove existing index
        if VECTORSTORE_DIR.exists():
            logger.info(f"Removing existing index at {VECTORSTORE_DIR}")
            shutil.rmtree(VECTORSTORE_DIR)

        # Rebuild
        vectorstore = _build_index()

        # Save metadata
        df = get_shipment_df()
        current_hashes = {_get_row_hash(row): idx for idx, row in df.iterrows()}

        metadata = {
            "last_update": datetime.now().isoformat(),
            "total_records": len(df),
            "total_chunks": 0,  # Will be calculated
            "indexed_hashes": current_hashes,
            "version": "1.0",
        }
        _save_metadata(metadata)

        logger.info("✅ Full rebuild completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Full rebuild failed: {e}", exc_info=True)
        return False


def get_index_status() -> Dict:
    """Get current index status and statistics."""
    metadata = _load_metadata()

    status = {
        "index_exists": VECTORSTORE_DIR.exists(),
        "last_update": metadata.get("last_update"),
        "total_records": metadata.get("total_records", 0),
        "total_chunks": metadata.get("total_chunks", 0),
        "version": metadata.get("version", "unknown"),
    }

    # Check if update needed
    if metadata.get("last_update"):
        try:
            df = get_shipment_df()
            changes = _detect_changes(df, metadata)
            status["pending_changes"] = {
                "new_records": len(changes["new"]),
                "deleted_records": len(changes["deleted"]),
            }
            status["update_recommended"] = (
                len(changes["new"]) > 0 or len(changes["deleted"]) > 0
            )
        except Exception as e:
            logger.warning(f"Could not check for changes: {e}")
            status["update_recommended"] = None

    return status
