#!/bin/bash

#chmod +x run.sh
# Ensure Poetry is installed, skip if already installed
if ! command -v poetry &>/dev/null; then
	echo "Poetry not found. Installing..."
	curl -sSL https://install.python-poetry.org | python3 -
else
	echo "Poetry is already installed."
fi

# Remove poetry.lock if it exists
if [ -f "poetry.lock" ]; then
	echo "Removing existing poetry.lock..."
	rm poetry.lock
fi

if [ -f "s3vdx_test4_3.pth" ]; then
	echo "Removing existing model..."
	rm s3vdx_test4_3.pth
fi

if [ -r "image" ]; then
	echo "Removing existing images..."
	rm -r image
fi

# Install dependencies
echo "Installing dependencies..."
poetry install

# Execute the specified Python file
#PYTHON_FILE="./vae/model.py" # Change this to the actual script path
# PYTHON_FILE="./vae/generate_data.py" # Change this to the actual script path
PYTHON_FILE="./vae/train.py" # Change this to the actual script path
echo "Running $PYTHON_FILE..."
poetry run python "$PYTHON_FILE"
