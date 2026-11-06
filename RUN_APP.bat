@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the Python 3.11 virtual environment as described in README.md.
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 -m streamlit run app.py %*
