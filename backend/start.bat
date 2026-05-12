@echo off
echo.
echo  =============================================
echo   FreshlyFishy Backend
echo   Loading TF + YOLO models — please wait...
echo   This takes ~30-60s on first run.
echo  =============================================
echo.
cd /d "%~dp0"
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
