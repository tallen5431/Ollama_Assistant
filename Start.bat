@echo off
setlocal EnableExtensions

REM ============================================
REM CodeSmith Ollama Helper Starter (Windows)
REM ============================================

REM Resolve app directory to the folder containing this script
set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

REM Virtual environment paths
set "VENV_DIR=%APP_DIR%\.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

REM Create virtual environment if needed
if not exist "%PYTHON_EXE%" (
    echo [SETUP] Creating virtual environment...
    REM Prefer "py -3" if available, fall back to "python"
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv "%VENV_DIR%"
    ) else (
        python -m venv "%VENV_DIR%"
    )
)

echo [SETUP] Installing dependencies...
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel >nul 2>&1

if exist "%APP_DIR%requirements.txt" (
    "%PYTHON_EXE%" -m pip install -r "%APP_DIR%requirements.txt" >nul 2>&1
)

REM Defaults (HTTPS Server Manager will usually override these via program.env)
if not defined HOST set "HOST=0.0.0.0"
if not defined PORT set "PORT=8070"
if not defined OLLAMA_HOST set "OLLAMA_HOST=http://127.0.0.1:11434"
if not defined OLLAMA_MODEL set "OLLAMA_MODEL=qwen2.5-coder:7b"

echo [RUN] Starting CodeSmith Ollama Helper on %HOST%:%PORT%
echo [RUN] OLLAMA_HOST=%OLLAMA_HOST%  OLLAMA_MODEL=%OLLAMA_MODEL%

REM Run the Flask app
"%PYTHON_EXE%" app.py
set "EXITCODE=%ERRORLEVEL%"

endlocal & exit /b %EXITCODE%
