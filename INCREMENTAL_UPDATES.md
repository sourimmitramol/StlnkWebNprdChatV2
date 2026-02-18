# Vector Store Incremental Update System

## Overview

This system allows **fast daily updates** to the ChromaDB vector index by only processing new/changed data, instead of rebuilding the entire index from scratch.

## 🚀 Quick Start

### Check Index Status
```powershell
D:/chatbot16012026/StlnkWebNprdChatV2/venv/python.exe update_index.py --status
```

### Daily Incremental Update (FAST - 2-5 minutes)
```powershell
D:/chatbot16012026/StlnkWebNprdChatV2/venv/python.exe update_index.py --incremental
```

### Monthly Full Rebuild (SLOW - 3-4 hours)
```powershell
D:/chatbot16012026/StlnkWebNprdChatV2/venv/python.exe update_index.py --full
```

## 📊 Performance Comparison

| Operation | Time | When to Use |
|-----------|------|-------------|
| **First Time Build** | 3-4 hours | Initial setup |
| **Incremental Update** | 2-5 minutes | Daily (only new data) |
| **Full Rebuild** | 3-4 hours | Monthly maintenance |

### Example: Daily Updates
- **Day 1**: Full build → 19,373 records → 3-4 hours
- **Day 2**: +50 new records → **3 minutes** ✅
- **Day 3**: +30 new records → **2 minutes** ✅
- **Day 4**: +100 new records → **5 minutes** ✅

**Savings: Hours → Minutes!**

## 🔄 How It Works

### 1. **Change Detection**
- Tracks MD5 hash of each shipment record
- Compares current data with indexed data
- Identifies: New, Modified, Deleted records

### 2. **Selective Updates**
- Only processes changed data
- Adds new embeddings to existing index
- Preserves existing data

### 3. **Metadata Tracking**
- Stores: `chroma_db/index_metadata.json`
- Tracks: Last update time, record hashes, statistics

## 📅 Recommended Schedule

### Option 1: Manual Daily Updates
Run this every morning:
```powershell
cd D:\chatbot16012026\StlnkWebNprdChatV2
D:/chatbot16012026/StlnkWebNprdChatV2/venv/python.exe update_index.py --incremental
```

### Option 2: Automated with Windows Task Scheduler

1. **Create a PowerShell script**: `daily_update.ps1`
```powershell
# daily_update.ps1
cd D:\chatbot16012026\StlnkWebNprdChatV2
D:/chatbot16012026/StlnkWebNprdChatV2/venv/python.exe update_index.py --incremental

# Send notification (optional)
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Vector index updated successfully"
} else {
    Write-Host "❌ Vector index update failed"
}
```

2. **Schedule with Task Scheduler**:
```powershell
# Create scheduled task (run as Administrator)
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File D:\chatbot16012026\StlnkWebNprdChatV2\daily_update.ps1"
$trigger = New-ScheduledTaskTrigger -Daily -At "6:00AM"
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "ChatbotVectorIndexUpdate" -Description "Daily incremental update of vector search index"
```

### Option 3: Automated with Python Scheduler

Add to your application startup:
```python
from apscheduler.schedulers.background import BackgroundScheduler
from services.vectorstore import update_vectorstore_incremental

scheduler = BackgroundScheduler()
scheduler.add_job(
    update_vectorstore_incremental,
    'cron',
    hour=6,  # 6 AM daily
    minute=0
)
scheduler.start()
```

## 🛠️ Maintenance

### When to Run Full Rebuild

Run a full rebuild in these cases:
- **Monthly maintenance** (recommended)
- After incremental update failures
- After major data structure changes
- When metadata file is corrupted

### Monitoring

Check index status anytime:
```powershell
D:/chatbot16012026/StlnkWebNprdChatV2/venv/python.exe update_index.py --status
```

Output example:
```
================================================================================
VECTORSTORE INDEX STATUS
================================================================================

Index Exists: ✅ Yes
Last Update: 2026-02-18T11:30:00
Total Records Indexed: 19,373
Total Chunks: 38,919
Version: 1.0

⚠️ Pending Changes:
  • New Records: 45
  • Deleted Records: 2

💡 Recommendation: Run 'python update_index.py --incremental'
================================================================================
```

## 🐛 Troubleshooting

### Issue: "429 Too Many Requests"
**Solution**: The script has built-in retry logic. If it fails:
1. Wait 5 minutes
2. Run again - it will resume from where it failed

### Issue: Incremental update not finding changes
**Solution**: Run full rebuild:
```powershell
D:/chatbot16012026/StlnkWebNprdChatV2/venv/python.exe update_index.py --full
```

### Issue: Metadata file corrupted
**Solution**: Delete and rebuild:
```powershell
Remove-Item "D:\chatbot16012026\StlnkWebNprdChatV2\chroma_db\index_metadata.json"
D:/chatbot16012026/StlnkWebNprdChatV2/venv/python.exe update_index.py --full
```

## 📝 Integration with Your App

### Import Functions
```python
from services.vectorstore import (
    update_vectorstore_incremental,  # Fast daily update
    force_rebuild_vectorstore,       # Monthly rebuild
    get_index_status,                # Check status
    get_vectorstore                  # Get vectorstore for queries
)

# Check if update needed
status = get_index_status()
if status.get('update_recommended'):
    update_vectorstore_incremental()

# Use vectorstore
vectorstore = get_vectorstore()
results = vectorstore.similarity_search("delayed containers", k=5)
```

### Auto-update on App Start (Optional)
```python
# In your main.py or startup
import threading
from services.vectorstore import update_vectorstore_incremental

def startup_update():
    """Background update on app start"""
    threading.Thread(
        target=update_vectorstore_incremental,
        daemon=True
    ).start()

# Call on startup
startup_update()
```

## 📈 Benefits Summary

✅ **Fast daily updates** (2-5 min vs 3-4 hours)  
✅ **Automatic change detection**  
✅ **Preserves existing data**  
✅ **Built-in retry logic**  
✅ **Easy scheduling**  
✅ **Status monitoring**  

## 🎯 Best Practices

1. **Daily**: Run incremental update
2. **Monthly**: Run full rebuild (1st of month)
3. **Monitor**: Check status weekly
4. **Automate**: Use Task Scheduler or cron
5. **Backup**: Keep metadata file in version control

---

**Questions?** Check the code in `services/vectorstore.py` or `update_index.py`
