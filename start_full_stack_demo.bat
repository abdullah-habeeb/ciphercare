@echo off
echo ========================================================
echo   Starting 5-Hospital FL Full-Stack Dashboard Demo
echo ========================================================

cd /d "%~dp0"

echo [1/3] Starting FastAPI Backend (Port 8000)...
start "FL Backend API" cmd /k "cd fl_dashboard\backend && uvicorn main:app --reload --port 8000"

echo [2/3] Starting React Frontend (Port 5173)...
start "FL Frontend UI" cmd /k "cd fl_dashboard\frontend && npm run dev"

echo [3/3] Waiting for servers to initialize...
timeout /t 5 >nul

echo [INFO] Opening Dashboard in default browser...
start http://localhost:5173

echo ========================================================
echo   Demo is RUNNING! 
echo   - Backend: http://localhost:8000/docs
echo   - Frontend: http://localhost:5173
echo ========================================================
pause
