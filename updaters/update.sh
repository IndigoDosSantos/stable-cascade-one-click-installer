#!/bin/bash

cd "$(dirname "$0")"
git pull

echo "Update finished! Press Enter to celebrate ^_^"
read -p "Press Enter to continue..."  # Wait for Enter
