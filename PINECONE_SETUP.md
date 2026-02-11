# Pinecone Vector Store Setup

This guide explains how to set up Pinecone as a vector store with automatic fallback from FAISS.

## Features

- **Dual Vector Store Support**: Uses both FAISS (local) and Pinecone (cloud)
- **Automatic Fallback**: If FAISS fails or returns no results, automatically tries Pinecone
- **Agent Integration**: Agent automatically uses vector search when specific tools can't find answers
- **Configurable**: Choose FAISS only, Pinecone only, or both

## Installation

1. Install Pinecone dependencies:
```powershell
pip install pinecone-client==3.0.0 langchain-pinecone==0.1.0
```

Or install all requirements:
```powershell
pip install -r requirements.txt
```

## Configuration

### 1. Get Pinecone Credentials

1. Sign up at [https://www.pinecone.io/](https://www.pinecone.io/)
2. Create a new project
3. Get your API key from the project settings
4. Note your environment/region (e.g., "us-east-1")

### 2. Update .env File

Add these lines to your `.env` file:

```env
# Pinecone Configuration (optional)
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=shipment-index

# Vector Store Type: 'faiss', 'pinecone', or 'both'
VECTOR_STORE_TYPE=both
```

### Vector Store Type Options:

- `faiss` (default): Uses only local FAISS index
- `pinecone`: Uses only Pinecone cloud index
- `both`: Uses FAISS first, falls back to Pinecone if FAISS fails

## Usage

### Automatic Fallback (Recommended)

The system automatically uses vector search with fallback:

1. **Agent tries specific tools first** (e.g., Get Delayed Containers)
2. **If tools return no data**, agent automatically tries Vector Search Tool
3. **Vector Search tries FAISS** (fast, local)
4. **If FAISS fails**, automatically tries Pinecone (cloud, always available)

### Manual Vector Search

You can also ask directly:
- "Search for information about container ABCD1234567"
- "Find data related to PO 5302969319"
- "What do you know about shipments to Los Angeles?"

## How It Works

### Architecture

```
User Query
    ↓
Agent (tries specific tools)
    ↓
If No Results → Vector Search Tool
    ↓
Try FAISS (local, fast)
    ↓
If No Results → Try Pinecone (cloud, reliable)
    ↓
Return Results to User
```

### Vector Search Fallback Logic

```python
from services.vectorstore import search_with_fallback

# This function automatically tries FAISS then Pinecone
results = search_with_fallback(query="delayed containers", k=5)
```

## Benefits

### FAISS (Local)
- ✅ Fast - no network latency
- ✅ Free - no cloud costs
- ✅ Privacy - data stays local
- ❌ Single instance - no sharing across servers

### Pinecone (Cloud)
- ✅ Always available - cloud reliability
- ✅ Scalable - handles large datasets
- ✅ Multi-instance - shared across servers
- ✅ Automatic backups
- ❌ Requires internet connection
- ❌ May have costs for large usage

### Both (Hybrid Approach)
- ✅ Best of both worlds
- ✅ FAISS speed with Pinecone reliability
- ✅ Automatic failover
- ✅ No single point of failure

## Troubleshooting

### Pinecone Connection Issues

If you see `Pinecone not installed` or connection errors:

1. Verify Pinecone is installed:
```powershell
pip show pinecone-client langchain-pinecone
```

2. Check your .env file has correct credentials

3. Verify API key is active in Pinecone dashboard

4. Check index name matches your Pinecone index

### Fall Back to FAISS Only

If Pinecone has issues, set in .env:
```env
VECTOR_STORE_TYPE=faiss
```

The system will work fine with FAISS only.

## Monitoring

Check logs for vector store activity:

```
[INFO] Found 5 results from FAISS
[INFO] Found 5 results from Pinecone (fallback)
[WARNING] FAISS search failed: <error>
[WARNING] No results found from any vector store
```

## Best Practices

1. **Start with FAISS**: Test locally first with `VECTOR_STORE_TYPE=faiss`
2. **Add Pinecone for Production**: Use `VECTOR_STORE_TYPE=both` for production
3. **Monitor Usage**: Check Pinecone dashboard for usage and costs
4. **Index Naming**: Use descriptive index names (e.g., `shipment-prod`, `shipment-dev`)
5. **Rebuild Indexes**: When data changes significantly, rebuild indexes for accuracy

## Rebuilding Indexes

### Rebuild FAISS Index
Delete the local index folder and restart:
```powershell
Remove-Item -Recurse -Force faiss_index
# Restart application - FAISS will rebuild automatically
```

### Rebuild Pinecone Index
Delete and recreate in Pinecone dashboard, or delete all vectors:
```python
from pinecone import Pinecone
pc = Pinecone(api_key="your-key")
index = pc.Index("shipment-index")
index.delete(delete_all=True)
# Restart application - Pinecone will repopulate
```

## Example Queries That Use Vector Search

- "Find containers related to supplier ABC Corp"
- "Show me shipments with unusual destinations"
- "What containers are similar to ABCD1234567?"
- "Find shipments with long transit times"
- "Search for hot containers delayed more than 10 days"

The agent will automatically use vector search if specific tools don't find answers!
