# Test fluid server with actual model inference
Write-Host "=== Fluid Server Model Test ===" -ForegroundColor Cyan

# Parameters
$modelPath = if ($args[0]) { $args[0] } else { ".\models" }
$serverHost = "127.0.0.1"
$port = 8080

Write-Host "Model path: $modelPath" -ForegroundColor Cyan
Write-Host "Testing with actual model inference..." -ForegroundColor Yellow

# Check if model directory exists
if (-not (Test-Path $modelPath)) {
    Write-Host "Error: Model path '$modelPath' does not exist." -ForegroundColor Red
    Write-Host "Please create the directory and add models, or specify a different path:" -ForegroundColor Yellow
    Write-Host "  .\scripts\test_with_models.ps1 C:\path\to\models" -ForegroundColor White
    exit 1
}

# Start server in background
Write-Host "`nStarting server..." -ForegroundColor Yellow
if (Test-Path "src\fluid_server") {
    $process = Start-Process -FilePath "uv" -ArgumentList "run", "python", "-m", "fluid_server", "--model-path", $modelPath, "--host", $serverHost, "--port", $port, "--preload" -PassThru -WindowStyle Hidden
} elseif (Test-Path "dist\fluid-server.exe") {
    $process = Start-Process -FilePath ".\dist\fluid-server.exe" -ArgumentList "--model-path", $modelPath, "--host", $serverHost, "--port", $port, "--preload" -PassThru -WindowStyle Hidden
} else {
    Write-Host "Error: No server found to run." -ForegroundColor Red
    exit 1
}

# Wait for server startup
Write-Host "Waiting for server to start (15 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

try {
    # Test health endpoint
    Write-Host "`nTesting health..." -ForegroundColor Yellow
    $health = Invoke-RestMethod -Uri "http://$serverHost`:$port/health" -Method Get
    Write-Host "Health Status: $($health.status)" -ForegroundColor Green
    Write-Host "Device: $($health.device)" -ForegroundColor Cyan
    Write-Host "Models loaded: $($health.models_loaded | ConvertTo-Json)" -ForegroundColor Cyan
    
    # Test available models
    Write-Host "`nTesting models endpoint..." -ForegroundColor Yellow
    $models = Invoke-RestMethod -Uri "http://$serverHost`:$port/v1/models" -Method Get
    Write-Host "Available models: $($models.data.Count)" -ForegroundColor Green
    foreach ($model in $models.data) {
        Write-Host "  - $($model.id) ($($model.model_type))" -ForegroundColor White
    }
    
    # Test chat completion if LLM models are available
    $llmModels = $models.data | Where-Object { $_.model_type -eq "llm" }
    if ($llmModels.Count -gt 0) {
        Write-Host "`nTesting chat completion..." -ForegroundColor Yellow
        $chatRequest = @{
            model = $llmModels[0].id
            messages = @(
                @{
                    role = "user"
                    content = "Hello! Can you respond with just 'Hello back!' to test the connection?"
                }
            )
            max_tokens = 50
            temperature = 0.7
        }
        
        $chatResponse = Invoke-RestMethod -Uri "http://$serverHost`:$port/v1/chat/completions" -Method Post -ContentType "application/json" -Body ($chatRequest | ConvertTo-Json -Depth 10)
        Write-Host "Chat response: $($chatResponse.choices[0].message.content)" -ForegroundColor Green
    } else {
        Write-Host "`nNo LLM models available for chat testing." -ForegroundColor Yellow
    }
    
    Write-Host "`nAll tests completed successfully!" -ForegroundColor Green
    
} catch {
    Write-Host "`nTest failed: $_" -ForegroundColor Red
    Write-Host "Check server logs for details." -ForegroundColor Yellow
} finally {
    # Stop server
    Write-Host "`nStopping server..." -ForegroundColor Yellow
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
    Write-Host "Test complete!" -ForegroundColor Cyan
}