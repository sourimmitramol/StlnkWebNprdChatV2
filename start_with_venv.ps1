# Start the application using the virtual environment

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

Write-Host "Starting FastAPI server on port 8000..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server`n" -ForegroundColor Gray

python -m uvicorn api.main:app --reload --port 8000
