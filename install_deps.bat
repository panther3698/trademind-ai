@echo off
echo ========================================
echo 🔧 Installing Missing TradeMind AI Dependencies
echo ========================================
echo.

cd backend

echo ⚙️ Activating Python environment...
call venv\Scripts\activate

echo ⚙️ Installing core missing dependencies...
pip install aiohttp==3.9.0

echo ⚙️ Installing ML dependencies...
pip install xgboost==2.0.2
pip install scikit-learn==1.3.2
pip install pandas==2.1.3
pip install numpy==1.25.2

echo ⚙️ Installing additional useful dependencies...
pip install ta==0.10.2
pip install requests==2.31.0
pip install python-telegram-bot==20.7

echo.
echo ✅ Dependencies installed!
echo.
echo 🚀 Now restart your server:
echo    1. Stop the current server (Ctrl+C)
echo    2. Run: start-trademind-advanced.bat
echo.
echo Or test manually:
echo    cd backend
echo    venv\Scripts\activate  
echo    python app\main.py
echo.
pause
