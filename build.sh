!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Create virtual environment if not exists
echo "Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt