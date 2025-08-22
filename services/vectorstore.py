# services/vectorstore.py
import logging
import os
import time
from pathlib import Path
from typing import List

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

from config import settings
from .azure_blob import get_shipment_df

logger = logging.getLogger("shipping_chatbot")
VECTORSTORE_DIR = Path("faiss_index")


def _embeddings() -> AzureOpenAIEmbeddings:
    """Create AzureOpenAIEmbeddings – the same object used for RAG and for
    the RetrievalQA chain."""
    return AzureOpenAIEmbeddings(
        azure_deployment=settings.AZURE_OPENAI_EMBEDDING_MODEL,
        openai_api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
    )


def _build_index() -> FAISS:
    """
    Build a brand‑new FAISS index from the shipment dataframe.
    The routine is **batched** (25 docs per batch) and respects Azure rate limits.
    """
    logger.info("Creating new FAISS vector store")
    df = get_shipment_df()

    # Turn each row into a single text block
    rows_as_text = [
        "\n".join(f"{k}: {v}" for k, v in row.items() if str(v).strip())
        for row in df.fillna("").astype(str).to_dict(orient="records")
    ]

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks: List[str] = []
    for txt in rows_as_text:
        chunks.extend(splitter.split_text(txt))

    embeddings = _embeddings()

    batch = 25
    vectorstore = FAISS.from_texts(chunks[:batch], embeddings)

    for i in range(batch, len(chunks), batch):
        vectorstore.add_texts(chunks[i:i + batch])
        logger.debug(f"Indexed batch {i // batch + 1}")
        time.sleep(1.2)         # small pause for rate‑limit safety

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
            _vectorstore = FAISS.load_local(
                str(VECTORSTORE_DIR),
                _embeddings(),
                allow_dangerous_deserialization=True,
            )
        else:
            _vectorstore = _build_index()
    return _vectorstore
