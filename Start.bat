@echo off
setlocal EnableExtensions

REM ============================================
REM Ollama Chat Starter (Windows)
REM The HTTPS Server Manager usually overrides these via program.env.
REM ============================================

REM Resolve app directory to the folder containing this script
set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

REM Virtual environment paths
set "VENV_DIR=%APP_DIR%.venv"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

REM Create virtual environment if needed
if not exist "%PYTHON_EXE%" (
    echo [SETUP] Creating virtual environment...
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

REM Optional voice support — a failure here is not fatal, the mic just stays hidden.
if exist "%APP_DIR%requirements-voice.txt" (
    echo [SETUP] Installing optional voice support...
    "%PYTHON_EXE%" -m pip install -r "%APP_DIR%requirements-voice.txt" >nul 2>&1
    if errorlevel 1 echo [WARN] Voice support ^(vosk^) unavailable - continuing without the mic button.
)

REM Settings file (.env beside this script). config.py reads it too, but the
REM defaults below set OLLAMA_HOST whether or not anyone asked, so a value
REM written in .env would reach Python already overridden. The environment
REM still wins; the file only fills in what nothing else has said.
REM Plain KEY=value lines only here — no "export" prefix on Windows.
set "ENV_FILE=%APP_DIR%.env"
if defined CHAT_ENV_FILE set "ENV_FILE=%CHAT_ENV_FILE%"
if exist "%ENV_FILE%" (
    for /f "usebackq eol=# tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
        if not defined %%A set "%%A=%%~B"
    )
)

REM Defaults (Server Manager will usually override these via program.env)
if not defined HOST set "HOST=0.0.0.0"
if not defined PORT set "PORT=8070"
if not defined OLLAMA_HOST set "OLLAMA_HOST=http://127.0.0.1:11434"
if not defined OLLAMA_MODEL set "OLLAMA_MODEL=llama3.1:8b"

echo [RUN] Starting Ollama Chat on %HOST%:%PORT%
echo [RUN] OLLAMA_HOST=%OLLAMA_HOST%  OLLAMA_MODEL=%OLLAMA_MODEL%

"%PYTHON_EXE%" app.py
set "EXITCODE=%ERRORLEVEL%"

endlocal & exit /b %EXITCODE%
