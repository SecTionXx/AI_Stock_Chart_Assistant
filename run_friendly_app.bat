@echo off
echo ========================================
echo AI Stock Chart Assistant - Friendly GUI
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Starting friendly application...
echo.

REM Check if dependencies are installed
python -c "import customtkinter, PIL, google.generativeai" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo Dependencies OK. Launching Friendly AI Stock Chart Assistant...
echo.

REM Run the friendly version
python friendly_stock_analyzer.py

echo.
echo Application closed. Thanks for using the AI Stock Chart Assistant!
pause
