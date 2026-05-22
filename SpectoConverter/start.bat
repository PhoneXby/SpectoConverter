@echo off
cd /d "%~dp0"
python -c "import customtkinter, numpy, PIL, matplotlib, scipy" 2>nul || (
    echo Installing dependencies...
    pip install -r requirements.txt
)
python app.py
if errorlevel 1 pause
