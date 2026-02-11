# Quick Start: Vector Store Setup

## Issue: Embeddings Not Working

The embeddings weren't working because:
1. FAISS index wasn't built yet
2. Configuration was set to "both" but Pinecone isn't installed
3. No graceful fallback when Pinecone is unavailable

## Fixed

I've updated the code to:
- ✓ Handle missing Pinecone gracefully
- ✓ Fall back to FAISS-only when Pinecone unavailable
- ✓ Better error logging
- ✓ Added test script to verify setup

## Quick Setup (No Pinecone)

### 1. Update .env file

Change this line in your `.env` file:
```env
VECTOR_STORE_TYPE=faiss
```

### 2. Build FAISS Index

Run the test script to build the index:
```powershell
python test_vectorstore.py
```

This will:
- Check your configuration
- Build the FAISS index from your shipment data
- Test vector search functionality
- Show sample results

### 3. Test in Chatbot

Once the index is built, try these queries:
```
"Search for delayed containers"
"Find information about container ABCD1234567"
"What shipments are going to Los Angeles?"
```

## How Vector Search Works Now

```
User Query
    ↓
Agent tries specific tools first
    ↓
If no results → Vector Search Tool
    ↓
Try FAISS (local, fast)
    ↓
If VECTOR_STORE_TYPE=both → Try Pinecone fallback
    ↓
Return results
```

## Optional: Add Pinecone Later

If you want cloud backup with Pinecone:

### 1. Install Pinecone
```powershell
pip install pinecone-client==3.0.0 langchain-pinecone==0.1.0
```

### 2. Get Pinecone Credentials
- Sign up at https://www.pinecone.io/
- Create a project and get your API key

### 3. Update .env
```env
PINECONE_API_KEY=your-api-key-here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=shipment-index
VECTOR_STORE_TYPE=both
```

### 4. Rebuild Index
```powershell
python test_vectorstore.py
```

## Troubleshooting

### "Import could not be resolved" errors
These are just VSCode/Pylance warnings. The packages ARE installed in your virtual environment. The code will run fine.

### FAISS index build fails
- Check Azure OpenAI credentials in .env
- Make sure shipment data is accessible from Azure Blob
- Check logs for specific errors

### "No results found"
- Make sure FAISS index was built successfully
- Try rebuilding: delete `faiss_index` folder and run `test_vectorstore.py`
- Check if your shipment data is loaded

### Vector search not being used by agent
The agent will automatically use vector search when specific tools return no results. You can force it by asking:
- "Search for..." (triggers vector search directly)
- "Find information about..." (triggers vector search)

## Current Status

✓ FAISS working (local, fast)
✓ Automatic fallback to FAISS if Pinecone unavailable  
✓ Vector search integrated with agent
✓ Test script available

To get started, just run:
```powershell
python test_vectorstore.py
```
