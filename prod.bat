@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual env missing — run setup.bat first
    exit /b 1
)

netstat -ano | findstr /R /C:":7860 .*LISTENING" >nul 2>&1
if %errorlevel%==0 (
    echo Stopping previous server on port 7860...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":7860 .*LISTENING"') do taskkill /PID %%a /F >nul 2>&1
    timeout /t 2 /nobreak >nul
)

set "TORCH_LIB=%~dp0.venv\Lib\site-packages\torch\lib"
if exist "%TORCH_LIB%" set "PATH=%TORCH_LIB%;%PATH%"

if not exist "%~dp0models\huggingface" mkdir "%~dp0models\huggingface"
set "HF_HOME=%~dp0models\huggingface"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"

set HF_HUB_DISABLE_XET=1
set PHOTO_EDIT_PORT=7860
set PHOTO_EDIT_LOG=photo-edit.log

echo.
echo === Photo Edit ===
echo Root: %~dp0
echo Open: http://127.0.0.1:7860
echo Log:  logs\photo-edit.log
echo Models cache: models\huggingface
echo.
echo Do not restart during generation.
echo.

start http://127.0.0.1:7860
".venv\Scripts\python.exe" web_app.py
pause
