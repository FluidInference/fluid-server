# Start fluid server with development options
Write-Host "=== Fluid Server Start Script ===" -ForegroundColor Cyan

# Default parameters
$modelPath = ".\models"
$serverHost = "127.0.0.1"
$port = 3847
$device = "GPU"

# Parse command line arguments
for ($i = 0; $i -lt $args.Length; $i++) {
    switch ($args[$i]) {
        "--model-path" { $modelPath = $args[++$i] }
        "--host" { $serverHost = $args[++$i] }
        "--port" { $port = $args[++$i] }
        "--help" {
            Write-Host "Usage: start_server.ps1 [options]" -ForegroundColor Yellow
            Write-Host "Options:" -ForegroundColor Yellow
            Write-Host "  --model-path PATH    Model directory path (default: .\models)" -ForegroundColor White
            Write-Host "  --host HOST         Host to bind to (default: 127.0.0.1)" -ForegroundColor White
            Write-Host "  --port PORT         Port to bind to (default: 3847)" -ForegroundColor White
            Write-Host "  --help              Show this help" -ForegroundColor White
            Write-Host "`nExamples:" -ForegroundColor Yellow
            Write-Host "  .\scripts\start_server.ps1" -ForegroundColor White
            Write-Host "  .\scripts\start_server.ps1 --model-path C:\MyModels" -ForegroundColor White
            exit 0
        }
    }
}

Write-Host "Starting Fluid Server..." -ForegroundColor Yellow
Write-Host "Model path: $modelPath" -ForegroundColor Cyan
Write-Host "Host: $serverHost" -ForegroundColor Cyan
Write-Host "Port: $port" -ForegroundColor Cyan
Write-Host "Device: $device" -ForegroundColor Cyan

# Check if we're in development mode (src exists)
if (Test-Path "src\fluid_server") {
    Write-Host "`nRunning in development mode..." -ForegroundColor Green
    uv run python -m fluid_server --model-path $modelPath --host $serverHost --port $port --reload
} elseif (Test-Path "dist\fluid-server.exe") {
    Write-Host "`nRunning executable..." -ForegroundColor Green
    & ".\dist\fluid-server.exe" --model-path $modelPath --host $serverHost --port $port
    Write-Host "`nError: Neither development source nor executable found." -ForegroundColor Red
    Write-Host "Please run from project root directory or build the executable first." -ForegroundColor Red
    exit 1
}