@echo off
COLOR 0B

cd %~dp0
git pull

echo "Update finished! Press Enter to celebrate ^_^"
pause > nul
COLOR 0F
