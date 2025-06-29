@echo off
echo ================================================================
echo Running TradeMind AI Unit Tests
echo ================================================================

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install test dependencies if not already installed
pip install pytest pytest-asyncio pytest-cov pytest-mock

REM Run tests with coverage
echo.
echo Running unit tests with coverage...
pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html:htmlcov

REM Run specific test categories
echo.
echo Running integration tests...
pytest tests/ -m integration -v

echo.
echo Running performance tests...
pytest tests/ -m performance -v

echo.
echo Running error handling tests...
pytest tests/ -m error_handling -v

echo.
echo ================================================================
echo Test execution completed!
echo Coverage report generated in htmlcov/index.html
echo ================================================================

pause 