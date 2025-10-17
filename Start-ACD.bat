@echo off
setlocal
set BASEDIR=%~dp0
set PY=%BASEDIR%\.venv\Scripts\python.exe

REM === Run all agents sequentially + Streamlit UI ===
"%PY%" "%BASEDIR%run_acd.py" --ui

pause
