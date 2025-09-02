# Fluid Server Build Script for Windows
# Builds a standalone .exe using PyInstaller for the current architecture

Write-Host "=== Fluid Server PyInstaller Build ===" -ForegroundColor Cyan

# Detect system architecture - use WMI for accurate detection
$systemType = (Get-WmiObject -Class Win32_ComputerSystem).SystemType
if ($systemType -eq "ARM64-based PC") {
    Write-Host "Detected ARM64 architecture (Snapdragon/ARM processor)" -ForegroundColor Green
    $archName = "arm64"
} elseif ($systemType -eq "x64-based PC") {
    Write-Host "Detected x64 (Intel/AMD) architecture" -ForegroundColor Green
    $archName = "x64"
} else {
    # Fallback to environment variable
    $arch = [System.Environment]::GetEnvironmentVariable("PROCESSOR_ARCHITECTURE")
    Write-Host "Detected architecture via env var: $arch" -ForegroundColor Yellow
    if ($arch -eq "ARM64") {
        $archName = "arm64"
    } elseif ($arch -eq "AMD64") {
        $archName = "x64"
    } else {
        $archName = $arch.ToLower()
    }
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
# Keep the main spec file, only remove generated ones
if (Test-Path "*-x64.spec") {
    Remove-Item -Path "*-x64.spec" -Force
}
if (Test-Path "*-arm64.spec") {
    Remove-Item -Path "*-arm64.spec" -Force
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

# Run PyInstaller using the spec file (builds for current architecture)
uv run pyinstaller fluid-server.spec --noconfirm --clean --log-level=INFO

# Check PyInstaller exit code
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyInstaller failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

# Verify build results
Write-Host "`nVerifying build results..." -ForegroundColor Yellow

if (Test-Path "dist/fluid-server.exe") {
    $exe = Get-Item "dist/fluid-server.exe"
    $size = $exe.Length / 1MB
    Write-Host "`nBuild successful!" -ForegroundColor Green
    Write-Host "Architecture: $archName" -ForegroundColor Cyan
    Write-Host "Executable: dist/fluid-server.exe" -ForegroundColor Cyan
    Write-Host ("Size: {0:N2} MB" -f $size) -ForegroundColor Cyan
    
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