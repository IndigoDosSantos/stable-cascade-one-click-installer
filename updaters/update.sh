#!/bin/bash

# Move to the directory above the script's location
cd "$(dirname "$0")"/..
git pull

echo "Update finished! Press Enter to celebrate ^_^"
read -p "Press Enter to continue..."
