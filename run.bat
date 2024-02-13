@echo off
REM Check for Python and exit if not found
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python and retry.
    exit /b
)

REM Create a virtual environment
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install requirements
pip install -r requirements.txt

echo Installation completed. Run your script with the virtual environment activated.
pause
