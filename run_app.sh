#!/bin/bash
# AI Stock Chart Assistant - Unix/Linux Launcher
# This script makes it easy to run the application on Unix-like systems

echo ""
echo "========================================"
echo "  AI Stock Chart Assistant v1.0.0"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed or not in PATH"
        echo "Please install Python 3.8+ from your package manager"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "ERROR: Python $PYTHON_VERSION found, but Python $REQUIRED_VERSION+ is required"
    exit 1
fi

echo "Python $PYTHON_VERSION found"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "ERROR: main.py not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment exists and activate it
if [ -f "venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -f "env/bin/activate" ]; then
    echo "Activating virtual environment..."
    source env/bin/activate
fi

# Check if dependencies are installed
echo "Checking dependencies..."
$PYTHON_CMD -c "import customtkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements_new.txt
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies"
        exit 1
    fi
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: .env file not found"
    echo "Please create a .env file with your GEMINI_API_KEY"
    echo "You can copy env_example.txt to .env and edit it"
    echo ""
    read -p "Press Enter to continue anyway..."
fi

# Run the application
echo "Starting AI Stock Chart Assistant..."
echo ""
$PYTHON_CMD main.py

# Check exit status
if [ $? -ne 0 ]; then
    echo ""
    echo "Application exited with an error"
    read -p "Press Enter to continue..."
fi

echo ""
echo "Application closed" 