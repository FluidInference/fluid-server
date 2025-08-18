# Fluid Server Build Script for Windows
# Builds a standalone .exe using PyInstaller

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

# Run PyInstaller with all necessary options
uv run pyinstaller --onefile `
    --name fluid-server `
    --noconsole `
    --hidden-import=openvino `
    --hidden-import=openvino_genai `
    --hidden-import=openvino.runtime `
    --hidden-import=openvino.properties `
    --hidden-import=uvicorn.logging `
    --hidden-import=uvicorn.loops `
    --hidden-import=uvicorn.loops.auto `
    --hidden-import=uvicorn.protocols `
    --hidden-import=uvicorn.protocols.http `
    --hidden-import=uvicorn.protocols.http.auto `
    --hidden-import=uvicorn.protocols.websockets `
    --hidden-import=uvicorn.protocols.websockets.auto `
    --hidden-import=uvicorn.lifespan `
    --hidden-import=uvicorn.lifespan.on `
    --hidden-import=uvicorn.lifespan.off `
    --collect-all=openvino `
    --collect-all=openvino_genai `
    --log-level=INFO `
    src\fluid_server\__main__.py

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