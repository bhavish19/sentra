@echo off
REM Batch file to start all SENTRA nodes on Windows
REM Usage: start_all_nodes.bat

setlocal enabledelayedexpansion

set N_NODES=5
set BASE_PORT=8000

echo ======================================================================
echo Starting SENTRA Multi-Node Training
echo ======================================================================
echo Total nodes: %N_NODES%
echo Base port: %BASE_PORT%
echo ======================================================================
echo.

REM Start each node in a new window
for /L %%i in (1,1,%N_NODES%) do (
    echo Starting Node %%i...
    start "SENTRA Node %%i" cmd /k "python run_node.py --node-id %%i --n-nodes %N_NODES% --base-port %BASE_PORT%"
    timeout /t 1 /nobreak >nul
)

echo.
echo All nodes started in separate windows.
echo Close each window to stop the corresponding node.
echo.
echo Usage: start_all_nodes.bat
pause

