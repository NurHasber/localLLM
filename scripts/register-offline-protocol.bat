@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0.."

set PYTHONW=%CD%\.venv\Scripts\pythonw.exe
set PYTHON=%CD%\.venv\Scripts\python.exe
if not exist "%PYTHONW%" set PYTHONW=%PYTHON%
if not exist "%PYTHON%" (
    echo Virtual env missing — run setup.bat first
    if /I not "%~1"=="silent" pause
    exit /b 1
)

"%PYTHON%" -u "%CD%\protocol_launch.py" register
if errorlevel 1 (
    echo [ОШИБКА] Не удалось зарегистрировать localphotoai://
    if /I not "%~1"=="silent" pause
    exit /b 1
)
echo Готово: протокол localphotoai:// и офлайн-кнопка запуска.
if /I not "%~1"=="silent" (
    echo.
    pause
)
