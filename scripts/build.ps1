# Fluid Server Build Script for Windows
# Builds a standalone .exe using PyInstaller for the current architecture

Write-Host "=== Fluid Server PyInstaller Build ===" -ForegroundColor Cyan

# Detect system architecture
$arch = [System.Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE")
if ($arch -eq "ARM64") {
    Write-Host "Detected ARM64 architecture" -ForegroundColor Green
    $archName = "arm64"
} elseif ($arch -eq "AMD64") {
    Write-Host "Detected x64 (AMD64) architecture" -ForegroundColor Green
    $archName = "x64"
} else {
    Write-Host "Detected architecture: $arch" -ForegroundColor Yellow
    $archName = $arch.ToLower()
}

Write-Host ""
# Kill any existing fluid-server processes
Write-Host "`nChecking for running fluid-server processes..." -ForegroundColor Yellow
$existingProcesses = Get-Process -Name "fluid-server" -ErrorAction SilentlyContinue
if ($existingProcesses) {
    Write-Host "Found $($existingProcesses.Count) running fluid-server process(es). Terminating..." -ForegroundColor Yellow
    $existingProcesses | Stop-Process -Force
    Start-Sleep -Seconds 2
    Write-Host "Processes terminated." -ForegroundColor Green
} else {
    Write-Host "No running fluid-server processes found." -ForegroundColor Green
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

# Ensure virtual environment exists
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    uv venv
}

# Install dependencies
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
uv sync
uv add --dev pyinstaller

# Create the executable
Write-Host "`nBuilding executable with PyInstaller for $archName..." -ForegroundColor Yellow

<<<<<<< HEAD
# Run PyInstaller using the spec file (builds for current architecture)
uv run pyinstaller fluid-server.spec --noconfirm --clean --log-level=INFO
=======
# Check if source file exists
if (-not (Test-Path "src\fluid_server\__main__.py")) {
    Write-Host "Error: Source file not found: src\fluid_server\__main__.py" -ForegroundColor Red
    exit 1
}

# Run PyInstaller directly from virtual environment
Write-Host "Running PyInstaller with comprehensive packaging options..." -ForegroundColor Green
$pyinstallerResult = & ".venv\Scripts\python.exe" -m PyInstaller --onefile `
    --name fluid-server `
    --noconsole `
    --add-data "src\fluid_server;fluid_server" `
    --hidden-import=openvino `
    --hidden-import=openvino_genai `
    --hidden-import=openvino_tokenizers `
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
    --hidden-import=fluid_server.app `
    --hidden-import=fluid_server.managers `
    --hidden-import=fluid_server.runtimes `
    --hidden-import=fluid_server.api `
    --hidden-import=fluid_server.models `
    --hidden-import=fluid_server.utils `
    --hidden-import=librosa `
    --hidden-import=scipy `
    --hidden-import=scipy.signal `
    --hidden-import=scipy.stats `
    --hidden-import=scipy.stats._distn_infrastructure `
    --hidden-import=scipy.stats._stats `
    --hidden-import=scipy.stats.distributions `
    --hidden-import=scipy.io `
    --hidden-import=scipy.fft `
    --hidden-import=numpy `
    --hidden-import=soundfile `
    --hidden-import=_soundfile_data `
    --hidden-import=multiprocessing `
    --hidden-import=asyncio `
    --collect-all=openvino `
    --collect-all=openvino_genai `
    --collect-all=openvino_tokenizers `
    --collect-all=librosa `
    --collect-all=scipy `
    --collect-all=soundfile `
    --log-level=WARN `
    --distpath=dist `
    --workpath=build `
    --clean `
    "src\fluid_server\__main__.py"

# Check PyInstaller exit code
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

# Verify build results
Write-Host "`nVerifying build results..." -ForegroundColor Yellow
>>>>>>> ec050bf (Kill server)

if (Test-Path "dist/fluid-server.exe") {
<<<<<<< HEAD
    $size = (Get-Item "dist/fluid-server.exe").Length / 1MB
    Write-Host "`nBuild successful!" -ForegroundColor Green
    Write-Host "Architecture: $archName" -ForegroundColor Cyan
    Write-Host "Executable: dist/fluid-server.exe" -ForegroundColor Cyan
    Write-Host ("Size: {0:N2} MB" -f $size) -ForegroundColor Cyan
=======
    $exe = Get-Item "dist/fluid-server.exe"
    $size = $exe.Length / 1MB
>>>>>>> ec050bf (Kill server)
    
    Write-Host "`nBuild successful!" -ForegroundColor Green
    Write-Host "Executable: $($exe.FullName)" -ForegroundColor Cyan
    Write-Host ("Size: {0:N2} MB" -f $size) -ForegroundColor Cyan
    Write-Host "Built: $($exe.LastWriteTime)" -ForegroundColor Cyan
    
    # Test if the executable can be run
    Write-Host "`nTesting executable..." -ForegroundColor Yellow
    $testResult = & "$($exe.FullName)" --help 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Executable runs successfully" -ForegroundColor Green
    } else {
        Write-Host "Warning: Executable test failed" -ForegroundColor Yellow
        Write-Host "Output: $testResult" -ForegroundColor Gray
    }
    
    # Copy executable to ../windows folder
    Write-Host "`nCopying to distribution folder..." -ForegroundColor Yellow
    $windowsDir = "..\windows"
    if (-not (Test-Path $windowsDir)) {
        New-Item -ItemType Directory -Path $windowsDir -Force | Out-Null
        Write-Host "Created directory: $windowsDir" -ForegroundColor Green
    }
    
    $destination = "$windowsDir\fluid-server.exe"
    Copy-Item -Path "$($exe.FullName)" -Destination $destination -Force
    if (Test-Path $destination) {
        Write-Host "Successfully copied to: $destination" -ForegroundColor Green
    } else {
        Write-Host "Warning: Failed to copy to $destination" -ForegroundColor Yellow
    }
    
    Write-Host "`nUsage Instructions:" -ForegroundColor Yellow
    Write-Host "  .\dist\fluid-server.exe --help" -ForegroundColor White
    Write-Host "  ..\windows\fluid-server.exe --help" -ForegroundColor White
    Write-Host "  .\dist\fluid-server.exe --host 127.0.0.1 --port 8080" -ForegroundColor White
    Write-Host "`nThen visit:" -ForegroundColor Yellow
    Write-Host "  http://localhost:8080/health" -ForegroundColor White
    Write-Host "  http://localhost:8080/docs" -ForegroundColor White
    Write-Host "  http://localhost:8080/v1/models" -ForegroundColor White
} else {
    Write-Host "`nBuild failed! Executable not found." -ForegroundColor Red
    if (Test-Path "build\fluid-server\warn-fluid-server.txt") {
        Write-Host "Build warnings:" -ForegroundColor Yellow
        Get-Content "build\fluid-server\warn-fluid-server.txt" | Write-Host -ForegroundColor Gray
    }
    exit 1
}