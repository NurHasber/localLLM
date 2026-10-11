@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" if not exist ".venv\Scripts\python.exe" (
    echo Virtual env missing — run setup.bat first
    pause
    exit /b 1
)
wscript.exe "%~dp0start-local-photo-ai.vbs"
exit /b 0
