@echo off
setlocal

REM Capture the current script's directory
set "script_dir=%~dp0"

REM Change to the directory above
cd ..

REM Check for Python and exit if not found
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python and retry.
    exit /b
)

REM Create a virtual environment in the current directory (which is now one level up)
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip before `pip install`
python -m pip install --upgrade pip

REM Install other requirements
pip install -r requirements.txt

echo Installation completed. Double-click `run.bat` file next to start generating!
pause
