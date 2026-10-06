@echo off
setlocal
python "%~dp0main.py" %*
exit /b %errorlevel%
