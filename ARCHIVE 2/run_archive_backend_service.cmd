@echo off
cd /d "C:\ARCHIVE\ARCHIVE 2"
"C:\ARCHIVE\.venv\Scripts\python.exe" -m waitress --host=0.0.0.0 --port=5000 archive_backend:app
