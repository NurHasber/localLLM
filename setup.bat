@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create .venv — is Python on PATH?
        exit /b 1
    )
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
echo Installing PyTorch with CUDA (cu124)...
".venv\Scripts\python.exe" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 exit /b 1
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if not exist "vendor\ACE-Step-1.5\.git" (
    echo Cloning ACE-Step 1.5...
    git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git "vendor\ACE-Step-1.5"
)
if not exist "vendor\MOSS-TTS\moss_soundeffect_v2\__init__.py" (
    echo Cloning MOSS-SoundEffect v2.0...
    git clone --depth 1 --filter=blob:none --sparse https://github.com/OpenMOSS/MOSS-TTS.git "vendor\MOSS-TTS"
    git -C "vendor\MOSS-TTS" sparse-checkout set moss_soundeffect_v2
)
if errorlevel 1 exit /b 1

if not exist "models\huggingface" mkdir "models\huggingface"

echo.
echo Done. Start: prod.bat
echo Model downloads are not in git. See DEPLOY.md
echo.
