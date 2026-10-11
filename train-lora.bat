@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Run setup.bat first
    exit /b 1
)

if not exist "tools\ai-toolkit\run.py" (
    echo Cloning ostris/ai-toolkit...
    if not exist "tools" mkdir "tools"
    git clone --recurse-submodules https://github.com/ostris/ai-toolkit.git "tools\ai-toolkit"
    if errorlevel 1 exit /b 1
)

if not exist "tools\ai-toolkit\venv\Scripts\python.exe" (
    echo AI-Toolkit venv is missing. Recreate: python -m venv tools\ai-toolkit\venv
    exit /b 1
)

".venv\Scripts\python.exe" training\check_dataset.py
if errorlevel 1 (
    echo Put paired photos in training\qwen-edit-lora\source and target, then run again.
    exit /b 1
)

set "HF_HOME=%~dp0models\huggingface"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set HF_HUB_DISABLE_XET=1

echo Training uses the GPU — do not generate photos at the same time.
echo Config: training\qwen-edit-lora\config.yaml
echo When finished, copy the .safetensors from training\qwen-edit-lora\output into loras\
echo.

cd /d "%~dp0tools\ai-toolkit"
"venv\Scripts\python.exe" run.py "..\..\training\qwen-edit-lora\config.yaml"
exit /b %errorlevel%
