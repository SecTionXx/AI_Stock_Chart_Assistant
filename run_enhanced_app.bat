@echo off
echo =============================================
echo AI Stock Chart Assistant - Enhanced Version
echo =============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Starting enhanced application...
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

echo Dependencies OK. 
echo.
echo Features in Enhanced Version:
echo - Automatic retry mechanisms
echo - Session recovery
echo - Enhanced error handling
echo - Connection health monitoring
echo - Auto-save functionality
echo.
echo Launching Enhanced AI Stock Chart Assistant...
echo.

REM Run the enhanced version
python enhanced_stock_analyzer.py

echo.
echo Application closed.
if exist "app_recovery_state.json" (
    echo Recovery data saved for next session.
)
pause
