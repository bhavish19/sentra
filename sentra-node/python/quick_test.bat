@echo off
REM Quick test runner batch file for Windows
cd /d "%~dp0"
python -m pytest tests/ -v
pause

