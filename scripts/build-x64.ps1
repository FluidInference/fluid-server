# Fluid Server Build Script for Windows x64
# Explicitly builds a standalone .exe for x64 architecture

Write-Host "=== Fluid Server PyInstaller Build (x64) ===" -ForegroundColor Cyan

# Check architecture and warn if not x64
$arch = [System.Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE")
if ($arch -ne "AMD64") {
    Write-Host "Warning: Current architecture is $arch, but building for x64" -ForegroundColor Yellow
    Write-Host "Note: Cross-compilation may not work properly with PyInstaller" -ForegroundColor Yellow
    Write-Host ""
}

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

# Create the executable for x64
Write-Host "`nBuilding executable with PyInstaller for x64..." -ForegroundColor Yellow

# Create a temporary spec file with x64 target
$specContent = Get-Content "fluid-server.spec" -Raw
$specContent = $specContent -replace '# target_arch can be set.*\r?\n.*# By default.*', "target_arch='x86_64',  # Explicitly target x64 architecture"
$specContent | Out-File -FilePath "fluid-server-x64.spec" -Encoding UTF8

# Run PyInstaller with the x64 spec
uv run pyinstaller fluid-server-x64.spec --noconfirm --clean --log-level=INFO

# Clean up temporary spec
Remove-Item -Path "fluid-server-x64.spec" -Force

# Check if build was successful
if (Test-Path "dist/fluid-server.exe") {
    $size = (Get-Item "dist/fluid-server.exe").Length / 1MB
    Write-Host "`nBuild successful!" -ForegroundColor Green
    Write-Host "Architecture: x64" -ForegroundColor Cyan
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