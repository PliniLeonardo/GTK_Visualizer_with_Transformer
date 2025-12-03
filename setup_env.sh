#!/bin/bash

# filepath: /Users/leonardoplini/Desktop/Visualizations/setup_env.sh

# Name of the virtual environment
VENV_DIR="VisualizationGTK_environment"

# Path to Python 3.11 (modifica questo percorso se necessario)
PYTHON_BIN="/usr/local/bin/python3.11"

# Check if Python 3.11 is installed
if ! [ -x "$(command -v $PYTHON_BIN)" ]; then
    echo "Error: Python 3.11 is not installed or not found at $PYTHON_BIN."
    echo "Please install Python 3.11 and try again."
    exit 1
fi

# Create the virtual environment
echo "Creating the virtual environment with Python 3.11..."
$PYTHON_BIN -m venv $VENV_DIR

# Activate the virtual environment
echo "Activating the virtual environment..."
source $VENV_DIR/bin/activate

# Install required packages from requirements.txt
if [ -f "requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install --upgrade pip  # Upgrade pip to the latest version
    pip install -r requirements.txt
else
    echo "requirements.txt not found."
fi

echo "Setup complete! The virtual environment is now activated."