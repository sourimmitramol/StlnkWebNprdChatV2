# services/azure_blob.py
import logging
from io import StringIO

import pandas as pd
from azure.storage.blob import BlobServiceClient
from langchain_community.chat_models import AzureChatOpenAI

from config import settings
from .preprocess import preprocess_data

logger = logging.getLogger("shipping_chatbot")


def _client() -> BlobServiceClient:
    """Helper to build the BlobServiceClient from Settings."""
    return BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)


def download_shipment_csv() -> pd.DataFrame:
    """Download the raw CSV from Azure Blob and return a *raw* DataFrame."""
    logger.info("Downloading shipment CSV from Azure Blob Storage")
    blob_client = _client().get_blob_client(
        container=settings.AZURE_CONTAINER_NAME,
        blob=settings.AZURE_BLOB_NAME,
    )
    raw_bytes = blob_client.download_blob().readall()
    df = pd.read_csv(StringIO(raw_bytes.decode("utf-8")))
    logger.info(f"Fetched {len(df)} rows")
    return df


# ----------------------------------------------------------------------
# Cached, pre‑processed DataFrame (singleton for the whole app)
# ----------------------------------------------------------------------
_cached_df: pd.DataFrame | None = None


def get_shipment_df() -> pd.DataFrame:
    """Public accessor – returns a **pre‑processed** copy of the shipment data."""
    global _cached_df
    if _cached_df is None:
        raw = download_shipment_csv()
        _cached_df = preprocess_data(raw)
    # Return a shallow copy so callers cannot mutate the cached object
    return _cached_df.copy()


llm = AzureChatOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION,
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
)
