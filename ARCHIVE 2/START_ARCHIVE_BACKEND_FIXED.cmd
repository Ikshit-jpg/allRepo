@echo off
cd /d "C:\ARCHIVE\ARCHIVE 2"
echo Starting ARCHIVE fixed backend on http://localhost:5052
echo.
"C:\ARCHIVE\.venv\Scripts\python.exe" -m waitress --host=127.0.0.1 --port=5052 archive_backend:app
