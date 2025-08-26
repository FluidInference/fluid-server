# Kill Fluid Server Processes Script
# Terminates all running fluid-server.exe processes

Write-Host "=== Fluid Server Process Killer ===" -ForegroundColor Cyan

# Find all fluid-server processes
$processes = Get-Process -Name "fluid-server" -ErrorAction SilentlyContinue

if ($processes) {
    Write-Host "`nFound $($processes.Count) fluid-server process(es):" -ForegroundColor Yellow
    
    foreach ($process in $processes) {
        Write-Host "  PID: $($process.Id) - CPU: $($process.CPU.ToString('F2'))s - Memory: $([math]::Round($process.WorkingSet64/1MB, 2))MB" -ForegroundColor Gray
    }
    
    Write-Host "`nTerminating processes..." -ForegroundColor Yellow
    
    try {
        $processes | Stop-Process -Force
        Start-Sleep -Seconds 1
        
        # Verify processes are terminated
        $remainingProcesses = Get-Process -Name "fluid-server" -ErrorAction SilentlyContinue
        if ($remainingProcesses) {
            Write-Host "Warning: Some processes may still be running:" -ForegroundColor Yellow
            foreach ($process in $remainingProcesses) {
                Write-Host "  PID: $($process.Id) - Still running" -ForegroundColor Red
            }
        } else {
            Write-Host "All fluid-server processes terminated successfully." -ForegroundColor Green
        }
        
    } catch {
        Write-Host "Error terminating processes: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
    
} else {
    Write-Host "`nNo fluid-server processes found." -ForegroundColor Green
}

Write-Host "`nDone." -ForegroundColor Cyan