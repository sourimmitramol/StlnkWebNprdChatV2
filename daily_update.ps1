# Daily Vector Index Update Script
# Run this daily at 6 AM to keep your vector search index up to date

$ErrorActionPreference = "Stop"
$ScriptDir = "D:\chatbot16012026\StlnkWebNprdChatV2"
$PythonExe = "$ScriptDir\venv\python.exe"
$UpdateScript = "$ScriptDir\update_index.py"
$LogFile = "$ScriptDir\logs\index_update_$(Get-Date -Format 'yyyyMMdd').log"

# Create logs directory if it doesn't exist
$LogDir = Split-Path $LogFile -Parent
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Start logging
$StartTime = Get-Date
Write-Host "========================================" | Tee-Object -FilePath $LogFile -Append
Write-Host "Vector Index Daily Update" | Tee-Object -FilePath $LogFile -Append
Write-Host "Started: $StartTime" | Tee-Object -FilePath $LogFile -Append
Write-Host "========================================" | Tee-Object -FilePath $LogFile -Append

try {
    # Change to script directory
    Set-Location $ScriptDir
    
    # Run incremental update
    Write-Host "`nRunning incremental update..." | Tee-Object -FilePath $LogFile -Append
    & $PythonExe $UpdateScript --incremental 2>&1 | Tee-Object -FilePath $LogFile -Append
    
    if ($LASTEXITCODE -eq 0) {
        $EndTime = Get-Date
        $Duration = $EndTime - $StartTime
        Write-Host "`n========================================" | Tee-Object -FilePath $LogFile -Append
        Write-Host "✅ SUCCESS" | Tee-Object -FilePath $LogFile -Append
        Write-Host "Duration: $($Duration.TotalMinutes) minutes" | Tee-Object -FilePath $LogFile -Append
        Write-Host "========================================" | Tee-Object -FilePath $LogFile -Append
        
        # Optional: Send success notification
        # Send-MailMessage -To "admin@example.com" -Subject "Vector Index Updated" -Body "Success"
        
        exit 0
    } else {
        throw "Update script exited with code $LASTEXITCODE"
    }
    
} catch {
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime
    Write-Host "`n========================================" | Tee-Object -FilePath $LogFile -Append
    Write-Host "❌ FAILED" | Tee-Object -FilePath $LogFile -Append
    Write-Host "Error: $_" | Tee-Object -FilePath $LogFile -Append
    Write-Host "Duration: $($Duration.TotalMinutes) minutes" | Tee-Object -FilePath $LogFile -Append
    Write-Host "========================================" | Tee-Object -FilePath $LogFile -Append
    
    # Optional: Send failure notification
    # Send-MailMessage -To "admin@example.com" -Subject "Vector Index Update FAILED" -Body "Error: $_"
    
    exit 1
}
