@echo off
setlocal
python "%~dp0run_colmap.py" sparse --workspace "%~dp0generated\colmap_workspace" %*
exit /b %errorlevel%
