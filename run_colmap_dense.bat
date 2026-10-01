@echo off
setlocal
python "%~dp0run_colmap.py" resume --workspace "%~dp0colmap_workspace" %*
exit /b %errorlevel%
