@echo off
setlocal enabledelayedexpansion
title TradeMind AI Startup Manager

echo ========================================
echo 🚀 TradeMind AI - Professional Trading System
echo ========================================
echo Version 2.0 - Advanced Startup Manager
echo ========================================
echo.

REM Check if we're in the right directory
if not exist "backend" (
    echo ❌ ERROR: backend directory not found!
    echo Make sure you're running this from C:\trademind-ai\
    echo Current directory: %CD%
    pause
    exit /b 1
)

if not exist "frontend" (
    echo ❌ ERROR: frontend directory not found!
    echo Make sure you're running this from C:\trademind-ai\
    pause
    exit /b 1
)

echo 🔍 Pre-flight checks...

REM Check Python virtual environment
if not exist "backend\venv\Scripts\activate.bat" (
    echo ❌ ERROR: Python virtual environment not found!
    echo Please run: cd backend && python -m venv venv
    pause
    exit /b 1
)

REM Check if Node.js is installed
where node >nul 2>nul
if errorlevel 1 (
    echo ❌ ERROR: Node.js not found!
    echo Please install Node.js from https://nodejs.org
    pause
    exit /b 1
)

REM Check if npm packages are installed
if not exist "frontend\node_modules" (
    echo ⚠️  WARNING: Frontend dependencies not found!
    echo Installing npm packages...
    cd frontend
    npm install
    if errorlevel 1 (
        echo ❌ ERROR: Failed to install npm packages!
        pause
        exit /b 1
    )
    cd ..
)

echo ✅ All pre-flight checks passed!
echo.

REM Kill any existing processes on our ports
echo 🔄 Cleaning up existing processes...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do taskkill /f /pid %%a >nul 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :3000') do taskkill /f /pid %%a >nul 2>nul

echo 📊 Starting Backend Server...
echo    - Port: 8000
echo    - API Docs: http://localhost:8000/docs
echo    - Health Check: http://localhost:8000/health
echo.

REM Start backend with enhanced logging
start "🤖 TradeMind AI Backend" cmd /k "title TradeMind Backend Server && cd /d %CD%\backend && echo Activating Python environment... && venv\Scripts\activate && echo Starting FastAPI server... && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

REM Wait for backend to start
echo ⏳ Waiting for backend to initialize...
:wait_backend
timeout /t 2 /nobreak >nul
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:8000' -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo    ⏳ Backend still starting...
    goto wait_backend
)

echo ✅ Backend server is ready!
echo.

echo 🌐 Starting Frontend Server...
echo    - Port: 3000
echo    - Dashboard: http://localhost:3000
echo    - Development Mode: Turbopack enabled
echo.

REM Start frontend
start "⚡ TradeMind AI Frontend" cmd /k "title TradeMind Frontend Server && cd /d %CD%\frontend && echo Starting Next.js development server... && npm run dev"

echo ⏳ Waiting for frontend to initialize...
:wait_frontend
timeout /t 3 /nobreak >nul
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:3000' -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
    echo    ⏳ Frontend still starting...
    goto wait_frontend
)

echo ✅ Frontend server is ready!
echo.

echo ========================================
echo 🎉 TradeMind AI is now fully operational!
echo ========================================
echo.
echo 📊 Backend API: http://localhost:8000
echo 🌐 Dashboard:   http://localhost:3000
echo 📚 API Docs:    http://localhost:8000/docs
echo 🏥 Health:      http://localhost:8000/health
echo.
echo 📱 Expected Features:
echo   ✅ Real-time trading signals every 45 seconds
echo   ✅ Live analytics dashboard
echo   ✅ WebSocket connections
echo   ✅ Telegram notifications (if configured)
echo.

REM Check Telegram configuration
findstr /C:"TELEGRAM_BOT_TOKEN=7025847XXX" backend\.env >nul
if not errorlevel 1 (
    echo ⚠️  Telegram Bot: Not configured yet
    echo    Add your bot token to backend\.env file
) else (
    echo ✅ Telegram Bot: Configured
)

echo.
echo 🚀 Opening TradeMind AI Dashboard...
timeout /t 3 /nobreak >nul
start http://localhost:3000

echo.
echo ========================================
echo 💡 SYSTEM MANAGEMENT COMMANDS
echo ========================================
echo.
echo To view logs:
echo   - Backend logs: Check "TradeMind AI Backend" window
echo   - Frontend logs: Check "TradeMind AI Frontend" window
echo.
echo To stop the system:
echo   - Close both server windows, or
echo   - Press Ctrl+C in each server window, or
echo   - Run: stop-trademind.bat
echo.
echo To restart:
echo   - Run this script again
echo.
echo ========================================
echo Press any key to minimize this window...
pause >nul