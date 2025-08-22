shipping_chatbot/
│
├─ config/
│   ├─ __init__.py
│   └─ settings.py                         # now includes BLOB_REFRESH_INTERVAL_SECONDS
│
├─ utils/
│   ├─ __init__.py
│   ├─ logger.py
│   ├─ container.py
│   └─ misc.py
│
├─ services/
│   ├─ __init__.py
│   ├─ azure_blob.py                      # cache + ETag refresh logic
│   ├─ preprocess.py
│   ├─ vectorstore.py
│   ├─ azure_helpers.py
│   └─ auto_updater.py                    # background poller (new)
│
├─ agents/
│   ├─ __init__.py
│   ├─ tools.py
│   └─ azure_agent.py
│
├─ cli/
│   ├─ __init__.py
│   └─ run.py                             # now launches auto‑updater asynchronously (optional)
│
├─ api/
│   ├─ __init__.py
│   ├─ schemas.py
│   └─ main.py                            # FastAPI startup / shutdown starts BlobAutoUpdater
│
├─ streamlit_app/
│   ├─ __init__.py
│   └─ app.py
│
└─ tests/
    ├─ __init__.py
    ├─ conftest.py
    ├─ test_preprocess.py
    ├─ test_vectorstore.py
    └─ test_agent.py



"""
Single source of truth for secrets	All environment variables are read via pydantic.BaseSettings → type‑checked, validated, and auto‑documented.
Cache‑aware heavy resources	- Blob CSV is downloaded once and cached (services.azure_blob._cached_df).
- FAISS index is persisted on disk (faiss_index/) and lazily loaded.
Separation of concerns	Data access (services), business logic (agents/tools), LLM wiring (agents/azure_agent), UI/API layers (cli, api, streamlit_app).
Logging & observability	Central logger with file + console, configurable log level, every module uses logger.
Retry & rate‑limit safety	Vectorstore building uses a small batch size & time.sleep; embeddings have a tenacity‑wrapped retry (get_azure_embeddings_with_retry in the original code – you can reuse it if needed).
Testability	All pure functions receive a DataFrame as an argument; the get_shipment_df cache can be overridden in tests (see tests/conftest.py).
Packaging	pyproject.toml pins exact versions that are known to be compatible with Python 3.11.4.
CI‑ready	Example GitHub Action (not shown in the tree but easy to add) would run ruff, black --check, mypy, and pytest.
Dockerised deployment	Dockerfile builds a slim python:3.11-slim image, copies the code, runs poetry install --no-dev, then starts uvicorn.
Extensible	Adding a new tool = write a plain function + append a Tool entry in agents/tools.py. No need to touch the API or UI.
"""


File	What is added
config/settings.py	a new optional setting BLOB_REFRESH_INTERVAL_SECONDS (default 300 s).
services/azure_blob.py	a thread‑safe cache that stores the blob’s ETag (or Last‑Modified) and reloads the CSV only when the remote blob changes.
services/auto_updater.py (new)	an optional background asyncio task that polls the blob every BLOB_REFRESH_INTERVAL_SECONDS and forces a refresh. The FastAPI app starts this task automatically; the CLI can also start it on demand.
api/main.py & cli/run.py	a tiny hook that kicks the background refresher on startup (FastAPI) or when the CLI is launched.