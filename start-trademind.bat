@echo off
echo ========================================
echo 🚀 TradeMind AI - Professional Trading System
echo ========================================
echo Starting backend and frontend servers...
echo.

REM Start backend in new window
echo 📊 Starting Backend Server...
start "TradeMind Backend" cmd /k "cd /d C:\trademind-ai\backend && venv\Scripts\activate && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait a bit for backend to start
timeout /t 5 /nobreak >nul

REM Start frontend in new window  
echo 🌐 Starting Frontend Server...
start "TradeMind Frontend" cmd /k "cd /d C:\trademind-ai\frontend && npm run dev"

echo.
echo ✅ Both servers are starting up!
echo 📊 Backend: http://localhost:8000
echo 🌐 Frontend: http://localhost:3000
echo.
echo Press any key to open the dashboard in your browser...
pause >nul

REM Open browser
start http://localhost:3000

echo.
echo 🎉 TradeMind AI is now running!
echo.
echo To stop the servers:
echo - Close the "TradeMind Backend" window
echo - Close the "TradeMind Frontend" window
echo.
pause