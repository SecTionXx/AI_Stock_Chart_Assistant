@echo off
REM AI Stock Chart Assistant - Windows Launcher
REM This script makes it easy to run the application on Windows

echo.
echo ========================================
echo   AI Stock Chart Assistant v1.0.0
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

REM Check if main.py exists
if not exist "main.py" (
    echo ERROR: main.py not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import customtkinter" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements_new.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Check for .env file
if not exist ".env" (
    echo.
    echo WARNING: .env file not found
    echo Please create a .env file with your GEMINI_API_KEY
    echo You can copy env_example.txt to .env and edit it
    echo.
    pause
)

REM Run the application
echo Starting AI Stock Chart Assistant...
echo.
python main.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with an error
    pause
)

echo.
echo Application closed
pause
