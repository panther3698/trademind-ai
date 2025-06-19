@echo off
echo ========================================
echo 🏥 TradeMind AI - System Health Check
echo ========================================
echo.

echo 🔍 Checking system components...
echo.

REM Check directories
echo [DIRECTORIES]
if exist "backend" (echo ✅ Backend directory exists) else (echo ❌ Backend directory missing)
if exist "frontend" (echo ✅ Frontend directory exists) else (echo ❌ Frontend directory missing)
if exist "backend\venv" (echo ✅ Python virtual environment exists) else (echo ❌ Python venv missing)
if exist "frontend\node_modules" (echo ✅ Node.js dependencies installed) else (echo ❌ npm packages missing)
echo.

REM Check configuration
echo [CONFIGURATION]
if exist "backend\.env" (echo ✅ Environment file exists) else (echo ❌ .env file missing)
if exist "backend\app\main.py" (echo ✅ Main application file exists) else (echo ❌ main.py missing)
if exist "frontend\src\app\page.tsx" (echo ✅ Frontend page exists) else (echo ❌ page.tsx missing)
echo.

REM Check ports
echo [SERVICES]
netstat -an | findstr :8000 >nul
if errorlevel 1 (echo ❌ Backend not running on port 8000) else (echo ✅ Backend running on port 8000)

netstat -an | findstr :3000 >nul  
if errorlevel 1 (echo ❌ Frontend not running on port 3000) else (echo ✅ Frontend running on port 3000)
echo.

REM Check web services
echo [CONNECTIVITY]
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:8000' -UseBasicParsing -TimeoutSec 2 | Out-Null; Write-Host '✅ Backend API responding' } catch { Write-Host '❌ Backend API not responding' }"
powershell -Command "try { Invoke-WebRequest -Uri 'http://localhost:3000' -UseBasicParsing -TimeoutSec 2 | Out-Null; Write-Host '✅ Frontend responding' } catch { Write-Host '❌ Frontend not responding' }"
echo.

echo ========================================
echo Health check complete!
echo ========================================
pause