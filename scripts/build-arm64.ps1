# Fluid Server Build Script for Windows ARM64 (Qualcomm)
# Explicitly builds a standalone .exe for ARM64 architecture

Write-Host "=== Fluid Server PyInstaller Build (ARM64) ===" -ForegroundColor Cyan

# Check architecture
$arch = [System.Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE")
if ($arch -ne "ARM64") {
    Write-Host "Error: This script requires an ARM64 system to build ARM64 executables" -ForegroundColor Red
    Write-Host "Current architecture: $arch" -ForegroundColor Red
    Write-Host "" -ForegroundColor Red
    Write-Host "PyInstaller cannot cross-compile between architectures." -ForegroundColor Yellow
    Write-Host "To build for ARM64, you must run this script on an ARM64 Windows machine." -ForegroundColor Yellow
    exit 1
}

Write-Host "Building on ARM64 architecture" -ForegroundColor Green
Write-Host ""

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

# Create the executable for ARM64
Write-Host "`nBuilding executable with PyInstaller for ARM64..." -ForegroundColor Yellow

# Create a temporary spec file with explicit ARM64 target
$specContent = Get-Content "fluid-server.spec" -Raw
$specContent = $specContent -replace '# target_arch can be set.*\r?\n.*# By default.*', "target_arch='arm64',  # Explicitly target ARM64 architecture"
$specContent | Out-File -FilePath "fluid-server-arm64.spec" -Encoding UTF8

# Run PyInstaller with the ARM64 spec
uv run pyinstaller fluid-server-arm64.spec --noconfirm --clean --log-level=INFO

# Clean up temporary spec
Remove-Item -Path "fluid-server-arm64.spec" -Force

# Check if build was successful
if (Test-Path "dist/fluid-server.exe") {
    $size = (Get-Item "dist/fluid-server.exe").Length / 1MB
    Write-Host "`nBuild successful!" -ForegroundColor Green
    Write-Host "Architecture: ARM64" -ForegroundColor Cyan
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