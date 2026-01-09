# services/azure_helpers.py
"""
Thin compatibility layer – the original notebook imported
`setup_azure_vectorstore`, `setup_azure_vectorstore_with_batching`,
`initialize_azure_agent` and `get_azure_embeddings` from a module called
`azure_helpers`.  To keep the public import path unchanged we simply re‑export
the implementations that live in `services.vectorstore` and `agents.azure_agent`.
"""

from .vectorstore import get_vectorstore as setup_azure_vectorstore
from .vectorstore import get_vectorstore as setup_azure_vectorstore_with_batching

from agents.azure_agent import initialize_azure_agent, get_azure_embeddings
