@echo off
COLOR 0B

REM Move to the directory above the script's location
cd %~dp0..
git pull

echo "Update finished! Press Enter to celebrate ^_^"
pause > nul
COLOR 0F
