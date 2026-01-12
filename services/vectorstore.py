# services/vectorstore.py
import logging
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
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
    The routine is batched (50 docs per batch) and respects Azure rate limits.
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

    # Optimization: Increase batch size slightly if Azure permits (usually safe up to 16k tokens/req).
    # 50 rows * 500 chars ~= 25k chars ~= 6k tokens. Safe.
    batch = 50 
    vectorstore = FAISS.from_texts(chunks[:batch], embeddings)

    for i in range(batch, len(chunks), batch):
        vectorstore.add_texts(chunks[i:i + batch])
        if i % (batch * 5) == 0:
            logger.debug(f"Indexed batch {i // batch + 1}/{len(chunks)//batch}")
         
        # 0.2s should be sufficient.
        time.sleep(0.2) 

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
