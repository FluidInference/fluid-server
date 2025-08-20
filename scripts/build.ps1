# Fluid Server Build Script for Windows ARM64 (Qualcomm)
# Builds a standalone .exe using PyInstaller targeting ARM64 architecture

Write-Host "=== Fluid Server PyInstaller Build ===" -ForegroundColor Cyan

# Check if uv is available
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Error: uv is not installed. Please install uv first." -ForegroundColor Red
    exit 1
}

# Clean previous builds
Write-Host "`nCleaning previous builds..." -ForegroundColor Yellow
if (Test-Path "dist") {
    Remove-Item -Path "dist" -Recurse -Force
}
if (Test-Path "build") {
    Remove-Item -Path "build" -Recurse -Force
}
if (Test-Path "*.spec") {
    Remove-Item -Path "*.spec" -Force
}

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
uv pip install -e .
uv pip install pyinstaller

# Create the executable
Write-Host "`nBuilding executable with PyInstaller..." -ForegroundColor Yellow

# Run PyInstaller using the spec file which is optimized for ARM64
uv run pyinstaller fluid-server.spec --noconfirm --clean --log-level=INFO

# Check if build was successful
if (Test-Path "dist/fluid-server.exe") {
    $size = (Get-Item "dist/fluid-server.exe").Length / 1MB
    Write-Host "`nBuild successful!" -ForegroundColor Green
    Write-Host "Executable: dist/fluid-server.exe" -ForegroundColor Cyan
    Write-Host ("Size: {0:N2} MB" -f $size) -ForegroundColor Cyan
    
    Write-Host "`nYou can test the server with:" -ForegroundColor Yellow
    Write-Host "  .\dist\fluid-server.exe" -ForegroundColor White
    Write-Host "`nThen visit:" -ForegroundColor Yellow
    Write-Host "  http://localhost:8080" -ForegroundColor White
    Write-Host "  http://localhost:8080/health" -ForegroundColor White
    Write-Host "  http://localhost:8080/docs" -ForegroundColor White
} else {
    Write-Host "`nBuild failed!" -ForegroundColor Red
    exit 1
}