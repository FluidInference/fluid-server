# Run type checking with ty
Write-Host "=== Fluid Server Type Check ===" -ForegroundColor Cyan

# Check if ty is available
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Error: uv is not installed. Please install uv first." -ForegroundColor Red
    exit 1
}

Write-Host "Installing ty..." -ForegroundColor Yellow
uv add --dev ty

Write-Host "`nRunning type check with ty..." -ForegroundColor Yellow
uv run ty

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nType check passed!" -ForegroundColor Green
} else {
    Write-Host "`nType check failed!" -ForegroundColor Red
    exit 1
}