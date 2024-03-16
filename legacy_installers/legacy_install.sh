#!/bin/bash
set -e

# Move to the directory one level up from the script's location
cd "$(dirname "$0")"/..

# Check for Python and exit if not found
if ! [ -x "$(command -v python)" ]; then
  echo 'Error: python is not installed.' >&2
  exit 1
fi

# Create a virtual environment in the current directory (now one level up)
if [ ! -d "venv" ]; then
  python -m venv venv
fi

# Upgrade pip before `pip install`
./venv/bin/python -m pip install --upgrade pip

# Install other requirements, assuming requirements.txt is also one level up
./venv/bin/pip install -r requirements.txt

echo "Installation completed. Execute 'run.sh' script to start generating!"
