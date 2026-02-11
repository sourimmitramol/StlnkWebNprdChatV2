# services/vectorstore.py
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

# Import text splitter from the correct package
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

from config import settings

from .azure_blob import get_shipment_df

logger = logging.getLogger("shipping_chatbot")
VECTORSTORE_DIR = Path("faiss_index")


def _embeddings() -> AzureOpenAIEmbeddings:
    """Create AzureOpenAIEmbeddings the same object used for RAG and for the RetrievalQA chain."""
    return AzureOpenAIEmbeddings(
        azure_deployment=settings.AZURE_OPENAI_EMBEDDING_MODEL,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
    )


def _build_index() -> FAISS:
    """
    Build a brand new FAISS index from the shipment dataframe.
    The routine is batched and throttled to respect Azure rate limits.
    """
    logger.info("Creating new FAISS vector store")
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

    # Use a conservative batch size and delay to avoid hitting
    # Azure OpenAI embeddings rate limits.
    # 20 chunks per batch with a 1s pause between batches greatly
    # reduces the chance of 429s for this one‑time indexing job.
    batch = 20
    logger.info(
        f"Building FAISS index from {len(chunks)} chunks with batch size {batch}"
    )

    # Initialize vectorstore with the first batch
    first_batch = chunks[:batch]
    vectorstore = FAISS.from_texts(first_batch, embeddings)

    # Process remaining batches with throttling
    for i in range(batch, len(chunks), batch):
        batch_texts = chunks[i : i + batch]
        vectorstore.add_texts(batch_texts)

        batch_num = (i // batch) + 1
        total_batches = (len(chunks) + batch - 1) // batch
        logger.info(
            f"Indexed batch {batch_num}/{total_batches} ({len(batch_texts)} chunks)"
        )

        # Be gentle with the embeddings endpoint to avoid 429s
        time.sleep(1.0)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    logger.info("FAISS index persisted")
    return vectorstore


# ----------------------------------------------------------------------
# Public accessor – lazy‑load on first call
# ----------------------------------------------------------------------
_vectorstore: FAISS | None = None


def get_vectorstore() -> FAISS:
    """Return a cached FAISS store; builds it on first request."""
    global _vectorstore
    if _vectorstore is None:
        if VECTORSTORE_DIR.exists():
            logger.info("Loading existing FAISS index")
            try:
                _vectorstore = FAISS.load_local(
                    str(VECTORSTORE_DIR),
                    _embeddings(),
                    allow_dangerous_deserialization=True,
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
    Returns FAISS, Pinecone, or both based on VECTOR_STORE_TYPE setting.
    Falls back to FAISS if Pinecone is unavailable.
    """
    store_type = getattr(settings, "VECTOR_STORE_TYPE", "faiss").lower()

    if store_type == "pinecone":
        pinecone = get_pinecone_vectorstore()
        if pinecone:
            return pinecone
        else:
            logger.warning("Pinecone not available, falling back to FAISS")
            return get_vectorstore()
    elif store_type == "both":
        faiss = get_vectorstore()
        pinecone = get_pinecone_vectorstore()
        if pinecone:
            return {"faiss": faiss, "pinecone": pinecone}
        else:
            logger.warning("Pinecone not available, using FAISS only")
            return faiss
    else:  # default to faiss
        return get_vectorstore()


def search_with_fallback(query: str, k: int = 5) -> List:
    """
    Search vector stores with fallback mechanism.
    Tries FAISS first, falls back to Pinecone if FAISS fails or returns no results.
    """
    results = []

    # Try FAISS first
    try:
        faiss_store = get_vectorstore()
        if faiss_store:
            results = faiss_store.similarity_search(query, k=k)
            if results:
                logger.info(f"Found {len(results)} results from FAISS")
                return results
            else:
                logger.info("FAISS returned no results, trying Pinecone fallback")
    except Exception as e:
        logger.warning(f"FAISS search failed: {e}")

    # Fallback to Pinecone only if it's configured
    try:
        store_type = getattr(settings, "VECTOR_STORE_TYPE", "faiss").lower()
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
