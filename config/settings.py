# config/settings.py
import os
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---------- Azure Blob ----------
    AZURE_STORAGE_CONNECTION_STRING: str = Field(
        ..., env="AZURE_STORAGE_CONNECTION_STRING"
    )
    AZURE_CONTAINER_NAME: str = Field(..., env="AZURE_CONTAINER_NAME")
    AZURE_BLOB_NAME: str = Field(..., env="AZURE_BLOB_NAME")
    AZURE_BLOB_API_VERSION: str = Field(
        default="2021-04-10", env="AZURE_BLOB_API_VERSION"
    )

    # ---------- Azure OpenAI ----------
    AZURE_OPENAI_ENDPOINT: str = Field(..., env="AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_KEY: str = Field(..., env="AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_API_VERSION: str = Field(
        default="2025-01-01-preview", env="AZURE_OPENAI_API_VERSION"
    )
    AZURE_OPENAI_DEPLOYMENT: str = Field(..., env="AZURE_OPENAI_DEPLOYMENT")
    AZURE_OPENAI_EMBEDDING_MODEL: str = Field(
        default="text-embedding-ada-002", env="AZURE_OPENAI_EMBEDDING_MODEL"
    )

    # ---------- Azure Search (optional) ----------
    AZURE_SEARCH_ENDPOINT: Optional[str] = Field(
        default=None, env="AZURE_SEARCH_ENDPOINT"
    )
    AZURE_SEARCH_API_KEY: Optional[str] = Field(
        default=None, env="AZURE_SEARCH_API_KEY"
    )
    AZURE_SEARCH_INDEX_NAME: Optional[str] = Field(
        default=None, env="AZURE_SEARCH_INDEX_NAME"
    )
    AZURE_SEARCH_SEMANTIC_CONFIG: Optional[str] = Field(
        default=None, env="AZURE_SEARCH_SEMANTIC_CONFIG"
    )
    AZURE_SEARCH_API_VERSION: Optional[str] = Field(
        default="2021-04-30", env="AZURE_SEARCH_API_VERSION"
    )

    # ---------- Logging ----------
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="app.log", env="LOG_FILE")
    azure_service_name: str = Field(..., description="Azure service name")
    azure_openai_type: str = Field(..., description="Azure OpenAI type")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @validator("LOG_LEVEL")
    def _ensure_valid_log_level(cls, v: str) -> str:
        ok = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in ok:
            raise ValueError(f"LOG_LEVEL must be one of {ok}")
        return v.upper()


# Create a singleton that the rest of the project can import
settings = Settings()
