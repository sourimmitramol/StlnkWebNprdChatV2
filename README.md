# Shipping Chatbot - AI-Powered Logistics Assistant

## 🚀 Project Overview

An enterprise-grade AI chatbot for shipping and logistics management, built with Azure OpenAI, LangChain, and FastAPI. This intelligent assistant provides real-time insights into container tracking, purchase orders, vessel information, and comprehensive shipment analytics.

### Key Features

- **🤖 Natural Language Processing**: Query shipping data using conversational language
- **📦 Container Tracking**: Real-time status updates, milestones, and location tracking
- **🚢 Vessel Information**: Comprehensive vessel and voyage details
- **📊 Analytics**: Transit time analysis, delay detection, and performance metrics
- **🔍 Smart Search**: Semantic vector search and SQL query capabilities
- **🔐 Multi-tenant Support**: Consignee-based authorization and data filtering
- **⚡ Real-time Data**: Automatic blob storage sync with ETag-based caching
- **🌐 Multiple Interfaces**: REST API, CLI, and Streamlit web interface

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Tools & Capabilities](#tools--capabilities)
- [Deployment](#deployment)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   FastAPI    │  │  Streamlit   │  │     CLI      │      │
│  │   REST API   │  │   Web App    │  │   Interface  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┴──────────────────┘
                             │
          ┌──────────────────▼─────────────────────┐
          │      LangChain Agent Orchestrator      │
          │   (Azure OpenAI GPT-4 Integration)     │
          └──────────────────┬─────────────────────┘
                             │
          ┌──────────────────▼─────────────────────┐
          │          Tool Ecosystem (60+)          │
          │  ┌────────────────────────────────┐    │
          │  │  Container Tracking Tools      │    │
          │  │  • Milestones • Status • ETA   │    │
          │  ├────────────────────────────────┤    │
          │  │  Analytics Tools               │    │
          │  │  • Transit • Delay • Hot       │    │
          │  ├────────────────────────────────┤    │
          │  │  Search Tools                  │    │
          │  │  • Vector • SQL • Semantic     │    │
          │  ├────────────────────────────────┤    │
          │  │  Data Lookup Tools             │    │
          │  │  • PO • Booking • BL • Vessel  │    │
          │  └────────────────────────────────┘    │
          └──────────────────┬─────────────────────┘
                             │
          ┌──────────────────▼─────────────────────┐
          │         Data Layer                     │
          │  ┌────────────┐  ┌──────────────┐     │
          │  │   Azure    │  │    FAISS     │     │
          │  │   Blob     │  │   Vector     │     │
          │  │  Storage   │  │    Store     │     │
          │  └────────────┘  └──────────────┘     │
          │  ┌────────────┐  ┌──────────────┐     │
          │  │  Pandas    │  │   SQLite     │     │
          │  │ DataFrame  │  │   Engine     │     │
          │  └────────────┘  └──────────────┘     │
          └────────────────────────────────────────┘
```

### Design Principles

1. **Single Source of Truth**: All configuration via `pydantic.BaseSettings`
2. **Cache-First Architecture**: Intelligent caching for heavy resources (blob data, FAISS index)
3. **Separation of Concerns**: Clear boundaries between data, business logic, and presentation
4. **Observability**: Comprehensive logging with file and console outputs
5. **Resilience**: Retry logic, rate limiting, and graceful degradation
6. **Extensibility**: Plugin-like tool architecture for easy feature additions

---

## 💻 Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.11.4+ | Core runtime |
| **LLM Framework** | LangChain | 0.2.0+ | Agent orchestration |
| **LLM Provider** | Azure OpenAI | GPT-4 | Natural language understanding |
| **Web Framework** | FastAPI | 0.115.0+ | REST API server |
| **Data Processing** | Pandas | 2.2.2+ | DataFrame operations |
| **Vector Search** | FAISS | 1.8.0+ | Semantic similarity search |
| **Cloud Storage** | Azure Blob | 12.22.0+ | Data persistence |
| **Embeddings** | Azure OpenAI | text-embedding-ada-002 | Vector generation |

### Supporting Libraries

- **Web Server**: Uvicorn (ASGI server)
- **UI Framework**: Streamlit (web interface)
- **String Matching**: FuzzyWuzzy, python-Levenshtein
- **Configuration**: python-dotenv, pydantic-settings
- **Retry Logic**: tenacity
- **Testing**: pytest, pytest-cov
- **Code Quality**: ruff, black, isort, mypy

---

## 📁 Project Structure

```
StlnkWebNprdChatV2/
│
├── 📁 config/                    # Configuration management
│   ├── __init__.py
│   └── settings.py               # Environment variables & settings
│
├── 📁 utils/                     # Utility functions
│   ├── __init__.py
│   ├── logger.py                 # Centralized logging
│   ├── container.py              # Container number extraction
│   └── misc.py                   # Miscellaneous helpers
│
├── 📁 services/                  # Business services layer
│   ├── __init__.py
│   ├── azure_blob.py             # Blob storage with ETag caching
│   ├── azure_helpers.py          # Azure service helpers
│   ├── preprocess.py             # Data cleaning & normalization
│   ├── vectorstore.py            # FAISS vector database
│   └── auto_updater.py           # Background data refresh service
│
├── 📁 agents/                    # AI agent components
│   ├── __init__.py
│   ├── azure_agent.py            # LangChain agent initialization
│   ├── prompts.py                # System prompts & date parsing
│   ├── router.py                 # Query routing logic
│   └── tools.py                  # 60+ tool implementations
│
├── 📁 api/                       # REST API layer
│   ├── __init__.py
│   ├── main.py                   # FastAPI application
│   └── schemas.py                # Pydantic request/response models
│
├── 📁 cli/                       # Command-line interface
│   ├── __init__.py
│   └── run.py                    # CLI entry point
│
├── 📁 streamlit_app/             # Web interface
│   ├── __init__.py
│   └── app.py                    # Streamlit application
│
├── 📁 tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py               # Test fixtures
│   ├── test_preprocess.py        # Data processing tests
│   ├── test_vectorstore.py       # Vector search tests
│   └── test_agent.py             # Agent behavior tests
│
├── 📁 .github/workflows/         # CI/CD pipelines
│   ├── main_stlnkwebchatbotnprdv2.yml    # Production deployment
│   └── dev_stlnkwebchatbotnprdv1.yml     # Development deployment
│
├── 📄 .env                       # Environment variables (not in git)
├── 📄 .gitignore                 # Git ignore rules
├── 📄 requirements.txt           # Python dependencies
├── 📄 pyproject.toml             # Poetry configuration
├── 📄 docker-compose.yml         # Docker services
├── 📄 Dockerfile                 # Container image
├── 📄 README.md                  # This file
└── 📄 GIT_WORKFLOW.md            # Git branching strategy
```

### Key Files Explained

| File | Purpose |
|------|---------|
| **config/settings.py** | Centralized configuration using Pydantic BaseSettings. Loads all Azure credentials, API keys, and application settings from environment variables. |
| **services/azure_blob.py** | Manages blob storage interactions with intelligent ETag-based caching. Only downloads data when remote blob changes. |
| **services/auto_updater.py** | Background service that polls Azure Blob Storage at configurable intervals (default: 300s) for data updates. |
| **agents/tools.py** | Contains 60+ specialized tools for container tracking, analytics, search, and data retrieval. Each tool is a standalone function. |
| **agents/prompts.py** | System prompts, date parsing logic (handles "next 7 days", "last month", "October 2025"), and column synonym mappings. |
| **api/main.py** | FastAPI application with health checks, CORS, and the main `/ask` endpoint. Handles conversational queries and agent initialization. |

---

## 🔧 Installation

### Prerequisites

- Python 3.11.4 or higher
- pip or Poetry package manager
- Azure account with:
  - Azure OpenAI service
  - Azure Blob Storage
  - Azure AI Search (optional)

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/sourimmitramol/StlnkWebNprdChatV2.git
cd StlnkWebNprdChatV2

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using Poetry

```bash
# Clone the repository
git clone https://github.com/sourimmitramol/StlnkWebNprdChatV2.git
cd StlnkWebNprdChatV2

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Option 3: Using Docker

```bash
# Build the Docker image
docker build -t shipping-chatbot .

# Run the container
docker run -p 8000:8000 --env-file .env shipping-chatbot
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
AZURE_CONTAINER_NAME="production-data"
AZURE_BLOB_NAME="shipment-pdata.csv"
AZURE_BLOB_API_VERSION="2021-04-10"

# Azure OpenAI
AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY="your_api_key"
AZURE_OPENAI_API_VERSION="2025-01-01-preview"
AZURE_OPENAI_DEPLOYMENT="gpt-4"
AZURE_OPENAI_EMBEDDING_MODEL="text-embedding-ada-002"

# Azure AI Search (Optional)
AZURE_SERVICE_NAME="your-search-service"
AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net"
AZURE_SEARCH_API_KEY="your_search_key"
AZURE_SEARCH_INDEX_NAME="shipment-p-index"

# Application Settings
BLOB_REFRESH_INTERVAL_SECONDS="300"  # 5 minutes
LOG_LEVEL="INFO"
```

### Configuration Class

The [`Settings`](config/settings.py) class in [`config/settings.py`](config/settings.py) handles all configuration:

```python
from config.settings import settings

# Access configuration
print(settings.AZURE_OPENAI_ENDPOINT)
print(settings.BLOB_REFRESH_INTERVAL_SECONDS)
```

---

## 🚀 Usage

### 1. REST API (FastAPI)

#### Start the Server

```bash
# Development mode (with auto-reload)
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### API Endpoints

**Health Check**
```bash
GET http://localhost:8000/health
```

**Ask Question**
```bash
POST http://localhost:8000/ask
Content-Type: application/json

{
  "question": "Where is container MSBU4522691?",
  "consignee_code": "0000866"
}
```

**Response Format**
```json
{
  "response": "Container MSBU4522691 status...",
  "observation": ["Raw tool output"],
  "table": [/* Structured data */],
  "mode": "agent"
}
```

### 2. Command-Line Interface

```bash
# Interactive mode
python -m cli.run

# Single query
python -m cli.run --query "Show hot containers arriving next week" --consignee "0000866"
```

### 3. Streamlit Web Interface

```bash
# Start Streamlit app
streamlit run streamlit_app/app.py

# Access at http://localhost:8501
```

### Example Queries

#### Container Tracking
```
- Where is container MSBU4522691?
- Track container TCLU4521258
- Get milestones for container MSKU4343533
- When will ABCD1234567 arrive at discharge port?
```

#### Purchase Orders
```
- Show status of PO 5300009636
- Which containers have PO 6300134648?
- Is PO 5302816722 delayed?
- Transit analysis for PO 6300134648
```

#### Delay Analysis
```
- Show delayed containers
- Containers delayed by more than 7 days
- Delayed POs at Shanghai
- Which containers are delayed by 5 days arriving at NLRTM?
```

#### Upcoming Arrivals
```
- Containers arriving next 7 days
- Hot containers arriving this week
- Upcoming arrivals at Los Angeles
- Containers going to ENFIELD in October
```

#### Analytics
```
- Average transit time for containers from Shanghai
- Show delay statistics for last month
- Transit analysis for carrier Maersk
- Count containers arrived in December 2025
```

---

## 📚 API Documentation

### POST `/ask`

Main endpoint for conversational queries.

**Request Schema** ([`QueryWithConsigneeBody`](api/schemas.py))

```typescript
{
  question: string;      // Natural language query
  consignee_code: string; // Comma-separated consignee codes (e.g., "0000866,0000867")
}
```

**Response Schema**

```typescript
{
  response: string;       // Human-readable answer
  observation: string[];  // Raw tool observations
  table: object[];        // Structured data (if available)
  mode: "agent" | "direct" | "fallback"; // Processing mode
}
```

**HTTP Status Codes**

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid request (missing question or consignee_code) |
| 500 | Internal server error |

---

## 🛠️ Tools & Capabilities

The agent has access to 60+ specialized tools organized by category:

### 📦 Container Tracking Tools

| Tool | Description | Example Query |
|------|-------------|---------------|
| **Get Container Milestones** | Complete journey timeline | "Track container MSBU4522691" |
| **Get Vessel Info** | Vessel and voyage details | "Which vessel is carrying TCLU4521258?" |
| **Check Arrival Status** | Delivery confirmation | "Has container ABCD1234567 arrived?" |
| **Get Container ETA** | Estimated arrival time | "When will MSKU4343533 arrive?" |

### 📋 Purchase Order Tools

| Tool | Description | Example Query |
|------|-------------|---------------|
| **Get Containers For PO** | PO-to-container mapping | "Which containers have PO 5300009636?" |
| **Get PO Transit Analysis** | Transit metrics for all containers in PO | "Transit analysis for PO 6300134648" |
| **Is PO Hot** | Check if PO is marked hot | "Is PO 5302816722 hot?" |
| **Get Upcoming POs** | POs arriving in timeframe | "POs arriving next 15 days" |

### ⏱️ Delay & Analytics Tools

| Tool | Description | Example Query |
|------|-------------|---------------|
| **Get Delayed Containers** | Containers with delays | "Containers delayed by more than 7 days" |
| **Get Delayed POs** | Purchase orders with delays | "POs delayed by 5 days" |
| **Get Bulk Transit Analysis** | Aggregate transit statistics | "Average delay for Maersk in December" |
| **Analyze Data With Pandas** | Complex analytical queries | "Count containers by port in last month" |

### 🚢 Vessel & Booking Tools

| Tool | Description | Example Query |
|------|-------------|---------------|
| **Get Booking Details** | Booking-to-container mapping | "Show containers for booking GT3000512" |
| **Get ETA For Booking** | Booking arrival time | "ETA for booking CN9140225" |
| **Get Carrier For BL** | Bill of lading carrier info | "Who is carrier for BL MOLWMNL2400017" |
| **Get Containers For BL** | BL-to-container mapping | "Containers for ocean BL MEDUKE520904" |

### 🔍 Search Tools

| Tool | Description | Example Query |
|------|-------------|---------------|
| **Vector Search** | Semantic similarity search | "Find shipments similar to..." |
| **SQL Query Tool** | Natural language to SQL | "Show me containers from Singapore" |
| **Lookup Keyword** | General keyword search | "Search for supplier ACME" |

### 📅 Date & Time Handling

The system has sophisticated date parsing via [`parse_time_period`](agents/prompts.py):

```python
# Relative dates
"next 7 days"          → Today to today+6
"last 15 days"         → Today-15 to today
"yesterday"            → Previous day

# Month names (with tense detection)
"arrived in October"   → October 2025 (past tense)
"arriving in October"  → October 2026 (future tense)
"October 2025"         → October 1-31, 2025

# Week/month references
"this week"            → Current Monday-Sunday
"this month"           → Current month
"last month"           → Previous calendar month

# Ranges
"between Jan and Mar"  → January 1 to March 31
"Q1 2026"              → January-March 2026
```

---

## 🚀 Deployment

### Azure Web App Deployment

The project includes GitHub Actions workflows for automated deployment:

#### Production (`main` branch)
- **Workflow**: [`.github/workflows/main_stlnkwebchatbotnprdv2.yml`](.github/workflows/main_stlnkwebchatbotnprdv2.yml)
- **Deployment Target**: `stlnkwebchatbotnprdv2`
- **Trigger**: Push to `main`

#### Development (`dev` branch)
- **Workflow**: [`.github/workflows/dev_stlnkwebchatbotnprdv1.yml`](.github/workflows/dev_stlnkwebchatbotnprdv1.yml)
- **Deployment Target**: `stlnkwebchatbotnprdv1`
- **Trigger**: Push to `dev`

### Manual Deployment

```bash
# Login to Azure
az login

# Deploy to Azure Web App
az webapp up --name your-app-name --resource-group your-rg --runtime PYTHON:3.11

# Set environment variables
az webapp config appsettings set --name your-app-name --resource-group your-rg --settings @.env
```

### Docker Deployment

```bash
# Build image
docker build -t shipping-chatbot:latest .

# Run container
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name shipping-chatbot \
  shipping-chatbot:latest

# Or use docker-compose
docker-compose up -d
```

---

## 👨‍💻 Development

### Development Setup

```bash
# Install dev dependencies
poetry install --with dev

# Or with pip
pip install -r requirements.txt
pip install pytest pytest-cov ruff black isort mypy pre-commit
```

### Code Quality Tools

```bash
# Format code
black .
isort .

# Lint code
ruff check .

# Type checking
mypy .

# Run all checks
pre-commit run --all-files
```

### Adding a New Tool

1. **Create tool function** in [`agents/tools.py`](agents/tools.py):

```python
def my_custom_tool(query: str) -> str:
    """
    Description of what this tool does.
    
    Args:
        query: Natural language query
        
    Returns:
        Result string or structured data
    """
    df = _df()  # Get data
    # ... implement logic
    return result
```

2. **Register tool** in the `build_tools()` function:

```python
Tool(
    name="My Custom Tool",
    func=my_custom_tool,
    description="When to use this tool and what it returns"
),
```

3. **Test the tool**:

```python
# In tests/test_agent.py
def test_my_custom_tool():
    result = my_custom_tool("test query")
    assert result is not None
```

### Branch Strategy

See [GIT_WORKFLOW.md](GIT_WORKFLOW.md) for detailed branching strategy:

```
main            → Production (stlnkwebchatbotnprdv2)
  └── dev       → Development (stlnkwebchatbotnprdv1)
       └── feature/* → Feature branches
```

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_agent.py

# Run with verbose output
pytest -v

# Run with logging
pytest -s
```

### Test Structure

```python
# tests/conftest.py - Test fixtures
@pytest.fixture
def sample_df():
    return pd.DataFrame({...})

# tests/test_preprocess.py - Data processing tests
def test_date_conversion():
    df = preprocess_data(sample_df)
    assert df['eta_dp'].dtype == 'datetime64[ns]'

# tests/test_agent.py - Agent behavior tests
def test_container_milestones():
    result = get_container_milestones("MSBU4522691")
    assert "Milestones" in result
```

---

## 📊 Performance Optimization

### Caching Strategy

1. **Blob Data Cache** ([`services/azure_blob.py`](services/azure_blob.py))
   - ETag-based validation
   - In-memory DataFrame cache
   - Automatic refresh when blob changes

2. **Vector Store Cache** ([`services/vectorstore.py`](services/vectorstore.py))
   - Persistent FAISS index on disk
   - Lazy loading
   - Rebuild only when data changes

3. **LLM Response Cache**
   - LangChain built-in caching
   - Reduces redundant API calls

### Background Refresh

[`services/auto_updater.py`](services/auto_updater.py) provides automatic data updates:

```python
# Configuration in .env
BLOB_REFRESH_INTERVAL_SECONDS=300  # Check every 5 minutes

# Auto-started in FastAPI app
@app.on_event("startup")
def startup():
    # ... agent initialization
    # Background updater started automatically
```

---

## 🔒 Security Considerations

1. **Environment Variables**: Never commit `.env` file. Use Azure Key Vault in production.
2. **API Keys**: Rotate Azure OpenAI keys regularly.
3. **Consignee Authorization**: All queries filtered by consignee_code.
4. **Input Validation**: Pydantic schemas validate all inputs.
5. **Rate Limiting**: Implement rate limiting in production (not included by default).

---

## 🤝 Contributing

### Contribution Guidelines

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make your changes**
4. **Run tests**: `pytest`
5. **Format code**: `black . && isort .`
6. **Commit**: `git commit -m "Add my feature"`
7. **Push**: `git push origin feature/my-feature`
8. **Create Pull Request** to `dev` branch

### Code Style

- Follow PEP 8
- Use type hints
- Document functions with docstrings
- Keep functions < 50 lines
- Add tests for new features

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors & Acknowledgments

**Development Team**: Starlink Development Team

**Special Thanks**:
- Azure OpenAI team for GPT-4 access
- LangChain community
- Contributors and testers

---

## 📞 Support

For issues, questions, or feature requests:

- **GitHub Issues**: [Create an issue](https://github.com/sourimmitramol/StlnkWebNprdChatV2/issues)
- **Documentation**: See inline code documentation
- **Email**: support@yourcompany.com

---

## 🗺️ Roadmap

### Planned Features

- [ ] Multi-language support (Spanish, Chinese)
- [ ] Real-time WebSocket updates
- [ ] Advanced analytics dashboard
- [ ] Mobile app integration
- [ ] Voice command support
- [ ] Export to Excel/PDF
- [ ] Custom report builder
- [ ] Integration with ERP systems

---

## 📈 Monitoring & Observability

### Logging

All components use the centralized logger from [`utils/logger.py`](utils/logger.py):

```python
from utils.logger import logger

logger.info("Processing query")
logger.warning("Slow response detected")
logger.error("Tool execution failed", exc_info=True)
```

### Application Insights (Production)

Configure Azure Application Insights for production monitoring:

```python
# Add to api/main.py
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=your-key'
))
```

---

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Project Architecture Diagram](docs/architecture.png)

---

