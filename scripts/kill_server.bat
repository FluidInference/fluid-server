@echo off
REM Kill all fluid-server.exe processes
echo === Killing Fluid Server Processes ===

tasklist /FI "IMAGENAME eq fluid-server.exe" 2>NUL | find /I /N "fluid-server.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Found fluid-server processes. Terminating...
    taskkill /F /IM "fluid-server.exe" /T
    if %ERRORLEVEL%==0 (
        echo All fluid-server processes terminated successfully.
    ) else (
        echo Error: Failed to terminate some processes.
    )
) else (
    echo No fluid-server processes found.
)

echo Done.