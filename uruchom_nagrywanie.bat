@echo off
cd /d "%~dp0"
venv\Scripts\python.exe collect_pjm.py
pause
