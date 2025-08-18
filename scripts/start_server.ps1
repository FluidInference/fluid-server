# Start fluid server with development options
Write-Host "=== Fluid Server Start Script ===" -ForegroundColor Cyan

# Default parameters
$modelPath = ".\models"
$host = "127.0.0.1"
$port = 8080
$device = "GPU"

# Parse command line arguments
for ($i = 0; $i -lt $args.Length; $i++) {
    switch ($args[$i]) {
        "--model-path" { $modelPath = $args[++$i] }
        "--host" { $host = $args[++$i] }
        "--port" { $port = $args[++$i] }
        "--device" { $device = $args[++$i] }
        "--help" {
            Write-Host "Usage: start_server.ps1 [options]" -ForegroundColor Yellow
            Write-Host "Options:" -ForegroundColor Yellow
            Write-Host "  --model-path PATH    Model directory path (default: .\models)" -ForegroundColor White
            Write-Host "  --host HOST         Host to bind to (default: 127.0.0.1)" -ForegroundColor White
            Write-Host "  --port PORT         Port to bind to (default: 8080)" -ForegroundColor White
            Write-Host "  --device DEVICE     Device (CPU/GPU/NPU) (default: GPU)" -ForegroundColor White
            Write-Host "  --help              Show this help" -ForegroundColor White
            Write-Host "`nExamples:" -ForegroundColor Yellow
            Write-Host "  .\scripts\start_server.ps1" -ForegroundColor White
            Write-Host "  .\scripts\start_server.ps1 --model-path C:\MyModels --device CPU" -ForegroundColor White
            exit 0
        }
    }
}

Write-Host "Starting Fluid Server..." -ForegroundColor Yellow
Write-Host "Model path: $modelPath" -ForegroundColor Cyan
Write-Host "Host: $host" -ForegroundColor Cyan
Write-Host "Port: $port" -ForegroundColor Cyan
Write-Host "Device: $device" -ForegroundColor Cyan

# Check if we're in development mode (src exists)
if (Test-Path "src\fluid_server") {
    Write-Host "`nRunning in development mode..." -ForegroundColor Green
    uv run python -m fluid_server --model-path $modelPath --host $host --port $port --device $device --reload
} elseif (Test-Path "dist\fluid-server.exe") {
    Write-Host "`nRunning executable..." -ForegroundColor Green
    & ".\dist\fluid-server.exe" --model-path $modelPath --host $host --port $port --device $device
} else {
    Write-Host "`nError: Neither development source nor executable found." -ForegroundColor Red
    Write-Host "Please run from project root directory or build the executable first." -ForegroundColor Red
    exit 1
}