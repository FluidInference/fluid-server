# Test the fluid-server.exe
Write-Host "Testing fluid-server.exe..." -ForegroundColor Cyan

# Start the server with console output for debugging
Write-Host "`nStarting server (console mode)..." -ForegroundColor Yellow
$process = Start-Process -FilePath ".\dist\fluid-server.exe" `
    -ArgumentList "--host", "127.0.0.1", "--port", "8080" `
    -PassThru `
    -NoNewWindow

# Wait for server to start
Write-Host "Waiting for server to initialize (10 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Test the health endpoint
Write-Host "`nTesting /health endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8080/health" -Method Get
    Write-Host "Health check successful!" -ForegroundColor Green
    Write-Host ($response | ConvertTo-Json -Depth 10) -ForegroundColor White
} catch {
    Write-Host "Failed to connect to server: $_" -ForegroundColor Red
}

# Test the root endpoint
Write-Host "`nTesting / endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:8080/" -Method Get
    Write-Host "Root endpoint successful!" -ForegroundColor Green
    Write-Host ($response | ConvertTo-Json -Depth 10) -ForegroundColor White
} catch {
    Write-Host "Failed to connect to root: $_" -ForegroundColor Red
}

# Stop the server
Write-Host "`nStopping server..." -ForegroundColor Yellow
Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
Write-Host "Test complete!" -ForegroundColor Green